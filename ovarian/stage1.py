"""Stage I: normal versus pathological triage on the full image, trained with the clinical CAM-loss.

EfficientNet-B7 sees the whole frame resized to 224 x 224 with ImageNet normalization. During training, the Stage II
detector supplies a box for each pathological image, from which the focus mask of the CAM-loss is built
(ovarian.clinical_cam_loss). Checkpoints are selected on patient-level validation macro-F1 (majority vote).
"""
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ColorJitter, InterpolationMode
from torchvision.transforms import functional as TF
from tqdm import tqdm

from ovarian import data, seeds, stats
from ovarian.clinical_cam_loss import (
    ATTENTION_WEIGHT,
    CONTEXT_WEIGHT,
    ROI_PADDING,
    WARMUP_EPOCHS,
    FeatureCapture,
    attention_weight,
    cam_focus_loss,
    focus_mask,
)
from ovarian.models import EfficientNetClassifier, load_weights

LABELS = {"normal": 0, "benign": 1, "malignant": 1}
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
IMAGE_COLUMNS = ["image", "class", "patient", "split", "evaluation_order", "y_true", "y_pred",
                 "logit_normal", "logit_pathological", "p_pathological"]
PATIENT_COLUMNS = ["patient", "split", "n_images", "y_true", "y_pred_majority", "n_pred_normal",
                   "n_pred_pathological", "vote_tied"]


@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    patience: int = 12
    image_size: int = 224
    seed: int | None = None
    num_workers: int = 4
    pretrained: bool = True
    roi_padding: float = ROI_PADDING
    context_weight: float = CONTEXT_WEIGHT
    attention_weight: float = ATTENTION_WEIGHT
    warmup_epochs: int = WARMUP_EPOCHS
    zoom_min: float = 0.90
    zoom_max: float = 1.10
    detector_imgsz: int = 640
    detector_conf: float = 0.25
    detector_iou: float = 0.45


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_split(limit: int | None = None) -> pd.DataFrame:
    """The Stage I partition with a binary label (pathological = 1); limit keeps the first images per split and class."""
    frame = data.stage1_split()
    if limit:
        frame = frame.groupby(["split", "class"], sort=False).head(limit)
    frame = frame.reset_index(drop=True)
    frame["label"] = frame["class"].map(LABELS).astype(int)
    return frame


def detector_boxes(frame: pd.DataFrame, weights: Path, *, imgsz: int = 640, conf: float = 0.25, iou: float = 0.45,
                   device: str = "cpu", cache: Path | None = None) -> dict:
    """Highest-confidence detector box (x1, y1, x2, y2, pixels) per pathological image, or None without a detection.

    Results are read from and added to the JSON cache when one is given.
    """
    boxes = {}
    if cache is not None and Path(cache).exists():
        boxes = {name: tuple(box) if box else None for name, box in json.loads(Path(cache).read_text()).items()}
    pending = frame[(frame.label == 1) & ~frame.image.isin(list(boxes))]
    if len(pending):
        from ultralytics import YOLO

        detector = YOLO(str(weights))
        target = 0 if torch.device(device).type == "cuda" else "cpu"
        for image, cls in tqdm(list(zip(pending.image, pending["class"], strict=True)), desc="Detector boxes"):
            image_bgr = cv2.imread(str(data.image_path(image, cls)), cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise FileNotFoundError(data.image_path(image, cls))
            result = detector.predict(source=[image_bgr], imgsz=imgsz, conf=conf, iou=iou, device=target,
                                      verbose=False)[0]
            if result.boxes is None or len(result.boxes) == 0:
                boxes[image] = None
                continue
            best = int(np.argmax(result.boxes.conf.cpu().numpy()))
            boxes[image] = tuple(int(v) for v in result.boxes.xyxy.cpu().numpy()[best][:4].astype(int))
        if cache is not None:
            Path(cache).parent.mkdir(parents=True, exist_ok=True)
            Path(cache).write_text(json.dumps({k: list(v) if v else None for k, v in boxes.items()}, indent=1))
    return boxes


class Stage1Dataset(Dataset):
    """Full images with ImageNet normalization; with_mask adds the focus mask, transformed identically.

    Items are (image, label, index), or (image, label, mask, index) with masks. Training adds horizontal flip,
    rotation up to 7 degrees, isotropic scaling and brightness and contrast jitter.
    """

    def __init__(self, frame: pd.DataFrame, boxes: dict | None = None, *, image_size: int = 224,
                 training: bool = False, with_mask: bool = False, roi_padding: float = ROI_PADDING,
                 context_weight: float = CONTEXT_WEIGHT, zoom: tuple[float, float] = (0.90, 1.10)):
        if training and not with_mask:
            raise ValueError("training requires focus masks")
        self.images = frame.image.tolist()
        self.paths = [str(data.image_path(image, cls)) for image, cls in zip(frame.image, frame["class"], strict=True)]
        self.labels = frame.label.astype(int).tolist()
        self.boxes = boxes or {}
        self.image_size = image_size
        self.training = training
        self.with_mask = with_mask
        self.roi_padding = roi_padding
        self.context_weight = context_weight
        self.zoom = zoom
        self.color_jitter = ColorJitter(brightness=0.10, contrast=0.16)

    def __len__(self) -> int:
        return len(self.paths)

    def load(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Original RGB image and its focus mask at full resolution."""
        with Image.open(self.paths[index]) as source:
            image_rgb = np.asarray(source.convert("RGB"))
        pathological = self.labels[index] == 1
        box = self.boxes.get(self.images[index]) if pathological else None
        return image_rgb, focus_mask(image_rgb, pathological, box, self.roi_padding, self.context_weight)

    def __getitem__(self, index: int):
        mask = None
        if self.with_mask:
            image_rgb, mask_array = self.load(index)
            image = Image.fromarray(image_rgb)
            mask = Image.fromarray((mask_array * 255.0).astype(np.uint8))
        else:
            with Image.open(self.paths[index]) as source:
                image = source.convert("RGB")

        if self.training:
            # Two unused draws keep the augmentation random stream aligned with the released training run.
            random.random()
            random.uniform(0.0, 0.2)
            if random.random() < 0.5:
                image, mask = TF.hflip(image), TF.hflip(mask)
            angle = random.uniform(-7.0, 7.0)
            scale = random.uniform(*self.zoom)
            geometry = dict(angle=angle, translate=[0, 0], scale=scale, shear=[0.0, 0.0], fill=0)
            image = TF.affine(image, interpolation=InterpolationMode.BILINEAR, **geometry)
            mask = TF.affine(mask, interpolation=InterpolationMode.NEAREST, **geometry)
            image = self.color_jitter(image)

        size = [self.image_size, self.image_size]
        image = TF.resize(image, size, interpolation=InterpolationMode.BILINEAR, antialias=True)
        tensor = TF.normalize(TF.to_tensor(image), mean=MEAN, std=STD)
        label = torch.tensor(self.labels[index], dtype=torch.long)
        if mask is None:
            return tensor, label, index
        mask = TF.resize(mask, size, interpolation=InterpolationMode.NEAREST)
        return tensor, label, TF.to_tensor(mask).clamp(0.0, 1.0), index


def make_loader(dataset: Dataset, batch_size: int = 8, num_workers: int = 4, shuffle: bool = False) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available(), persistent_workers=num_workers > 0)


def class_weights(labels) -> torch.Tensor:
    """Inverse-frequency weights total / (2 * n_class) for the two classes."""
    labels = np.asarray(labels, dtype=int)
    counts = [max(int((labels == k).sum()), 1) for k in (0, 1)]
    return torch.tensor([sum(counts) / (2.0 * n) for n in counts], dtype=torch.float32)


def load_stage1_model(weights: Path, device: str = "cpu") -> nn.Module:
    model = EfficientNetClassifier("b7", pretrained=False)
    return load_weights(model, weights).to(device)


def train_one_epoch(model: nn.Module, capture: FeatureCapture, loader: DataLoader, criterion: nn.Module,
                    optimizer: torch.optim.Optimizer, device: str, weight: float) -> dict:
    model.train()
    sums = dict(loss=0.0, ce_loss=0.0, attention_loss=0.0, inside_share=0.0)
    y_true, y_pred = [], []
    for images, labels, masks, _ in tqdm(loader, desc="Train"):
        images, labels, masks = images.to(device).float(), labels.to(device), masks.to(device).float()
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        ce = criterion(logits.float(), labels)
        penalty, share = cam_focus_loss(model, capture.features, labels, masks)
        loss = ce + weight * penalty
        loss.backward()
        optimizer.step()
        for key, value in zip(sums, (loss, ce, penalty, share), strict=True):
            sums[key] += float(value.detach())
        y_true += labels.cpu().tolist()
        y_pred += logits.argmax(dim=1).cpu().tolist()
    steps = max(len(loader), 1)
    return {**{k: v / steps for k, v in sums.items()}, "image_accuracy": float(accuracy_score(y_true, y_pred)),
            "image_macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0))}


def evaluate(model: nn.Module, loader: DataLoader, frame: pd.DataFrame, device: str,
             capture: FeatureCapture | None = None, criterion: nn.Module | None = None,
             weight: float = 0.0) -> tuple[pd.DataFrame, dict]:
    """Per-image logits and predictions in loader order, plus mean losses when masks, capture and criterion are given."""
    model.eval()
    sums = dict(loss=0.0, ce_loss=0.0, attention_loss=0.0, inside_share=0.0)
    indices, logits_all, steps = [], [], 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluate"):
            images, labels = batch[0].to(device).float(), batch[1].to(device)
            logits = model(images)
            if len(batch) == 4 and capture is not None and criterion is not None:
                ce = criterion(logits.float(), labels)
                penalty, share = cam_focus_loss(model, capture.features, labels, batch[2].to(device).float())
                for key, value in zip(sums, (ce + weight * penalty, ce, penalty, share), strict=True):
                    sums[key] += float(value)
                steps += 1
            indices += batch[-1].tolist()
            logits_all.append(logits.float().cpu())
    logits = torch.cat(logits_all)
    rows = frame.iloc[indices]
    out = rows[["image", "class", "patient", "split"]].reset_index(drop=True)
    out["evaluation_order"] = np.arange(1, len(out) + 1)
    out["y_true"] = rows.label.to_numpy()
    out["y_pred"] = logits.argmax(dim=1).numpy()
    out["logit_normal"] = logits[:, 0].numpy().astype(float)
    out["logit_pathological"] = logits[:, 1].numpy().astype(float)
    out["p_pathological"] = torch.softmax(logits.double(), dim=1)[:, 1].numpy()
    losses = {k: v / steps for k, v in sums.items()} if steps else {}
    return out, losses


def patient_majority(predictions: pd.DataFrame) -> pd.DataFrame:
    """Majority vote of the image predictions per patient; a tied vote takes the patient's first image in evaluation order."""
    frames = []
    for split, images in predictions.sort_values(["split", "evaluation_order"]).groupby("split", sort=True):
        votes = stats.patient_vote(images, pred="y_pred")
        truth = images.groupby("patient", sort=False).y_true.first()
        votes["split"] = split
        votes["y_true"] = votes.patient.map(truth).to_numpy()
        frames.append(votes.rename(columns={"n_pred_0": "n_pred_normal", "n_pred_1": "n_pred_pathological"}))
    table = pd.concat(frames, ignore_index=True).sort_values(["split", "patient"]).reset_index(drop=True)
    return table[PATIENT_COLUMNS]


def split_metrics(predictions: pd.DataFrame) -> dict:
    """Image and patient accuracy, macro-F1 and confusion matrices (rows true, columns predicted) per split."""
    patients = patient_majority(predictions)
    result = {}
    for split, images in predictions.groupby("split", sort=False):
        votes = patients[patients.split == split]
        result[split] = {
            "n_images": int(len(images)),
            "n_patients": int(len(votes)),
            "image_accuracy": float(accuracy_score(images.y_true, images.y_pred)),
            "image_macro_f1": float(f1_score(images.y_true, images.y_pred, average="macro", zero_division=0)),
            "patient_accuracy": float(accuracy_score(votes.y_true, votes.y_pred_majority)),
            "patient_macro_f1": float(f1_score(votes.y_true, votes.y_pred_majority, average="macro",
                                               zero_division=0)),
            "image_confusion_matrix": confusion_matrix(images.y_true, images.y_pred, labels=[0, 1]).tolist(),
            "patient_confusion_matrix": confusion_matrix(votes.y_true, votes.y_pred_majority, labels=[0, 1]).tolist(),
        }
    return result


def write_outputs(predictions: pd.DataFrame, out_dir: Path, extra: dict | None = None) -> dict:
    """Write stage1_image_predictions.csv, stage1_patient_predictions.csv and stage1_metrics.json."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions[IMAGE_COLUMNS].to_csv(out_dir / "stage1_image_predictions.csv", index=False)
    patient_majority(predictions).to_csv(out_dir / "stage1_patient_predictions.csv", index=False)
    metrics = {**(extra or {}), "splits": split_metrics(predictions)}
    (out_dir / "stage1_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def save_mask_previews(dataset: Stage1Dataset, out_dir: Path, indices, size: int = 320) -> list[Path]:
    """Overlay the focus mask (viridis, 40 %) on the image, downscaled to size pixels, for visual checks."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for index in indices:
        image_rgb, mask = dataset.load(int(index))
        heatmap = cv2.applyColorMap((mask * 255.0).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
        overlay = cv2.addWeighted(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR), 0.60, heatmap, 0.40, 0)
        path = out_dir / f"mask_label{dataset.labels[index]}_{Path(dataset.images[index]).stem}.png"
        cv2.imwrite(str(path), cv2.resize(overlay, (size, size), interpolation=cv2.INTER_AREA))
        written.append(path)
    return written


def train(cfg: TrainConfig, frame: pd.DataFrame, boxes: dict, out_dir: Path, device: str = "cpu") -> dict:
    """Train on the train split, select on validation, then write predictions and metrics for validation and test."""
    out_dir = Path(out_dir)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    splits = {name: frame[frame.split == name].reset_index(drop=True) for name in ("train", "val", "test")}

    def loader(name: str, training: bool) -> DataLoader:
        dataset = Stage1Dataset(splits[name], boxes, image_size=cfg.image_size, training=training, with_mask=True,
                                roi_padding=cfg.roi_padding, context_weight=cfg.context_weight,
                                zoom=(cfg.zoom_min, cfg.zoom_max))
        return make_loader(dataset, cfg.batch_size, cfg.num_workers, shuffle=training)

    train_loader, val_loader, test_loader = loader("train", True), loader("val", False), loader("test", False)

    seed_everything(seeds.resolve(cfg.seed))
    model = EfficientNetClassifier("b7", pretrained=cfg.pretrained).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights(splits["train"].label).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    best_path = out_dir / "checkpoints" / "stage1_best.pt"
    history, best_score, best_epoch, stale = [], -1.0, 0, 0

    with FeatureCapture(model) as capture:
        for epoch in range(1, cfg.epochs + 1):
            weight = attention_weight(epoch, cfg.attention_weight, cfg.warmup_epochs)
            train_stats = train_one_epoch(model, capture, train_loader, criterion, optimizer, device, weight)
            val_pred, val_losses = evaluate(model, val_loader, splits["val"], device, capture, criterion, weight)
            val_stats = {k: v for k, v in split_metrics(val_pred)["val"].items() if not k.endswith("matrix")}
            history.append({"epoch": epoch, "attention_weight": weight,
                            **{f"train_{k}": v for k, v in train_stats.items()},
                            **{f"val_{k}": v for k, v in {**val_losses, **val_stats}.items()}})
            pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
            print(f"epoch {epoch}: " + ", ".join(f"{k}={v:.4f}" for k, v in history[-1].items()
                                                  if isinstance(v, float)))

            if val_stats["patient_macro_f1"] > best_score:
                best_score, best_epoch, stale = val_stats["patient_macro_f1"], epoch, 0
                torch.save(model.state_dict(), best_path.with_suffix(".tmp"))
                best_path.with_suffix(".tmp").replace(best_path)
            else:
                stale += 1
            if stale >= cfg.patience:
                print(f"early stopping at epoch {epoch}; best epoch {best_epoch}")
                break

        load_weights(model, best_path)
        results = [evaluate(model, lo, splits[name], device, capture, criterion, cfg.attention_weight)
                   for name, lo in (("val", val_loader), ("test", test_loader))]

    extra = {"best_epoch": best_epoch, "best_val_patient_macro_f1": best_score, "checkpoint": best_path.name,
             "attention": {name: losses for name, (_, losses) in zip(("val", "test"), results, strict=True)}}
    return write_outputs(pd.concat([pred for pred, _ in results], ignore_index=True), out_dir, extra)
