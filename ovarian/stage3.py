"""Stage III: benign versus malignant classification of the Stage II cyst region of interest.

Configuration 2 classifies the crop alone. Configuration 6 concatenates the eleven morphological
descriptors of the automatic cyst mask with the image embedding (ovarian.models.EfficientNetFusion).

Both configurations see the same input: the crop of the highest-confidence detector box, the full frame
when the detector finds nothing, resized to 224 x 224 and passed as RGB in [0, 1]. Unlike Stage I, no
ImageNet normalization is applied. The descriptors are z-scored with statistics fitted on the training
images only, which are released next to the weights.

Patient decisions are the majority vote of the image predictions, a tied vote counting as malignant.
"""
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm

from ovarian import data, paths, seeds, stats
from ovarian.models import load_weights, stage3_model
from ovarian.stage1 import make_loader, seed_everything

LABELS = {"benign": 0, "malignant": 1}
IMAGE_SIZE = 224
DETECTOR_IMGSZ = 640
DETECTOR_CONF = 0.25
DETECTOR_IOU = 0.45
IMAGE_COLUMNS = ["image", "class", "patient", "split", "evaluation_order", "y_true", "y_pred",
                 "logit_benign", "logit_malignant", "p_malignant"]
PATIENT_COLUMNS = ["patient", "split", "n_images", "y_true", "y_pred_majority", "n_pred_benign",
                   "n_pred_malignant", "vote_tied", "p_malignant_mean"]
AGGREGATION_THRESHOLDS = (0.50, 0.75, 0.85, 0.95)
# Learning rate and batch size of each reported configuration (Supplementary Table S7).
TRAINING = {2: {"learning_rate": 1e-4, "batch_size": 8}, 6: {"learning_rate": 1e-5, "batch_size": 4}}


@dataclass
class TrainConfig:
    """Stage III training protocol: Adam, plain cross-entropy, no augmentation, no early stopping.

    The checkpoint with the highest image-level validation macro-F1 is kept.
    """

    configuration: int = 6
    epochs: int = 250
    batch_size: int = 4
    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    seed: int | None = None
    num_workers: int = 4
    pretrained: bool = True

    @classmethod
    def for_configuration(cls, configuration: int, **overrides) -> "TrainConfig":
        """The reported settings of one configuration, with any keyword overriding them."""
        return cls(configuration=configuration, **{**TRAINING[configuration], **overrides})


def load_split(splits: list[str] | None = None) -> pd.DataFrame:
    """The Stage III partition with a binary label (malignant = 1), optionally restricted to some splits."""
    frame = data.stage3_split()
    if splits is not None:
        frame = frame[frame.split.isin(splits)]
    frame = frame.reset_index(drop=True)
    frame["label"] = frame["class"].map(LABELS).astype(int)
    return frame


def load_boxes(path: str | Path) -> dict:
    """Released detector boxes: image name to (x1, y1, x2, y2), or None where the detector found nothing."""
    stored = json.loads(Path(path).read_text())
    return {name: None if entry.get("full_frame_fallback") else tuple(entry["box"]) for name, entry in stored.items()}


def detector_boxes(frame: pd.DataFrame, weights: Path, *, imgsz: int = DETECTOR_IMGSZ, conf: float = DETECTOR_CONF,
                   iou: float = DETECTOR_IOU, device: str = "cpu", cache: Path | None = None) -> dict:
    """Highest-confidence detector box per image, or None without a detection; cached as JSON when given.

    Box coordinates are truncated to integers, as in the released boxes. Detector output can differ by a
    pixel between platforms, which changes the crop but not the reported metrics.
    """
    boxes = {}
    if cache is not None and Path(cache).exists():
        boxes = {name: tuple(box) if box else None for name, box in json.loads(Path(cache).read_text()).items()}
    pending = frame[~frame.image.isin(list(boxes))]
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


def load_normalization(path: str | Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Descriptor names and the training-set means and standard deviations released with the weights."""
    stored = json.loads(Path(path).read_text())
    names = list(stored["feature_names"])
    return (names, np.array([stored["means"][n] for n in names], dtype=np.float64),
            np.array([stored["stds"][n] for n in names], dtype=np.float64))


def load_features(paths_by_split: dict[str, Path], names: list[str], fallback: bool = False) -> pd.DataFrame:
    """The descriptors of the automatic masks, one row per image, indexed by image name.

    The reported validation and test numbers require the automatic descriptors released per split. With
    `fallback`, images not covered by those files (the training split) take the released annotation-box
    descriptors instead; the two are correlated but not interchangeable, so it is off by default.
    """
    frames = []
    for path in paths_by_split.values():
        if Path(path).exists():
            frames.append(pd.read_csv(path).set_index("image"))
        elif not fallback:
            raise FileNotFoundError(f"{path} is missing; it is part of the data release")
    if fallback:
        released = data.descriptors().set_index("image")
        covered = pd.concat(frames).index if frames else pd.Index([])
        frames.append(released.drop(index=covered, errors="ignore"))
    table = pd.concat(frames)
    missing = [n for n in names if n not in table.columns]
    if missing:
        raise ValueError(f"missing descriptor columns: {missing}")
    return table[names]


def crop(image_rgb: np.ndarray, box: tuple | None, size: int = IMAGE_SIZE) -> np.ndarray:
    """The box region resized to `size`, or the whole frame when there is no box."""
    if box is not None:
        x1, y1, x2, y2 = (int(v) for v in box)
        region = image_rgb[y1:y2, x1:x2]
        if region.size:
            image_rgb = region
    return cv2.resize(image_rgb, (size, size), interpolation=cv2.INTER_LINEAR)


class Stage3Dataset(Dataset):
    """Crops of the detector region, with the normalized descriptors for the fusion configurations.

    Yields (image, label, index) for Configuration 2 and (image, label, descriptors, index) otherwise.
    """

    def __init__(self, frame: pd.DataFrame, boxes: dict, features: pd.DataFrame | None = None,
                 mean: np.ndarray | None = None, std: np.ndarray | None = None, image_size: int = IMAGE_SIZE):
        self.images = frame.image.tolist()
        self.classes = frame["class"].tolist()
        self.labels = frame.label.to_numpy()
        self.boxes = boxes
        self.image_size = image_size
        self.tabular = None
        if features is not None:
            missing = [name for name in self.images if name not in features.index]
            if missing:
                raise KeyError(f"no descriptors for {len(missing)} images, first: {missing[0]}")
            values = features.loc[self.images].to_numpy(dtype=np.float64)
            self.tabular = ((values - mean) / std).astype(np.float32)

    def __len__(self) -> int:
        return len(self.images)

    def load(self, index: int) -> np.ndarray:
        """The 224 x 224 RGB crop of one image, as uint8."""
        path = data.image_path(self.images[index], self.classes[index])
        image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(path)
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        return crop(rgb, self.boxes.get(self.images[index]), self.image_size)

    def __getitem__(self, index: int):
        image = torch.from_numpy(self.load(index)).permute(2, 0, 1).float() / 255.0
        label = int(self.labels[index])
        if self.tabular is None:
            return image, label, index
        return image, label, torch.from_numpy(self.tabular[index]), index


def load_stage3_model(configuration: int, weights: Path, device: str = "cpu") -> nn.Module:
    return load_weights(stage3_model(configuration, pretrained=False), weights).to(device).eval()


def evaluate(model: nn.Module, loader, frame: pd.DataFrame, device: str) -> pd.DataFrame:
    """Per-image logits and predictions in loader order."""
    model.eval()
    indices, logits_all = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluate"):
            images = batch[0].to(device).float()
            logits = model(images) if len(batch) == 3 else model(images, batch[2].to(device).float())
            indices += batch[-1].tolist()
            logits_all.append(logits.float().cpu())
    logits = torch.cat(logits_all)
    rows = frame.iloc[indices]
    out = rows[["image", "class", "patient", "split"]].reset_index(drop=True)
    out["evaluation_order"] = np.arange(1, len(out) + 1)
    out["y_true"] = rows.label.to_numpy()
    out["y_pred"] = logits.argmax(dim=1).numpy()
    out["logit_benign"] = logits[:, 0].numpy().astype(float)
    out["logit_malignant"] = logits[:, 1].numpy().astype(float)
    out["p_malignant"] = torch.softmax(logits.double(), dim=1)[:, 1].numpy()
    return out


def patient_majority(predictions: pd.DataFrame) -> pd.DataFrame:
    """Majority vote of the image predictions per patient; a tied vote is malignant."""
    frames = []
    for split, images in predictions.sort_values(["split", "evaluation_order"]).groupby("split", sort=True):
        votes = stats.patient_vote(images, pred="y_pred", prob="p_malignant", tie_to_positive=True)
        truth = images.groupby("patient", sort=False).y_true.first()
        votes["split"] = split
        votes["y_true"] = votes.patient.map(truth).to_numpy()
        frames.append(votes.rename(columns={"n_pred_0": "n_pred_benign", "n_pred_1": "n_pred_malignant",
                                            "prob_mean": "p_malignant_mean"}))
    table = pd.concat(frames, ignore_index=True).sort_values(["split", "patient"]).reset_index(drop=True)
    return table[PATIENT_COLUMNS]


def _auc(truth, score) -> float | None:
    truth = np.asarray(truth)
    return float(roc_auc_score(truth, score)) if len(np.unique(truth)) == 2 else None


def _rule_scores(truth, prediction, score) -> dict:
    """Accuracy, macro-F1 and AUC of one patient-level aggregation rule."""
    return {"accuracy": float(accuracy_score(truth, prediction)),
            "macro_f1": float(f1_score(truth, prediction, average="macro", zero_division=0)),
            "auc": _auc(truth, score)}


def split_metrics(predictions: pd.DataFrame) -> dict:
    """Image and patient accuracy, macro-F1, AUC and confusion matrices per split."""
    patients = patient_majority(predictions)
    result = {}
    for split, images in predictions.groupby("split", sort=False):
        votes = patients[patients.split == split]
        result[split] = {
            "n_images": int(len(images)),
            "n_patients": int(len(votes)),
            "image_accuracy": float(accuracy_score(images.y_true, images.y_pred)),
            "image_macro_f1": float(f1_score(images.y_true, images.y_pred, average="macro", zero_division=0)),
            "image_auc": _auc(images.y_true, images.p_malignant),
            "patient_accuracy": float(accuracy_score(votes.y_true, votes.y_pred_majority)),
            "patient_macro_f1": float(f1_score(votes.y_true, votes.y_pred_majority, average="macro",
                                               zero_division=0)),
            "patient_auc": _auc(votes.y_true, votes.p_malignant_mean),
            "image_confusion_matrix": confusion_matrix(images.y_true, images.y_pred, labels=[0, 1]).tolist(),
            "patient_confusion_matrix": confusion_matrix(votes.y_true, votes.y_pred_majority,
                                                         labels=[0, 1]).tolist(),
        }
    return result


def aggregation_rules(predictions: pd.DataFrame, thresholds=AGGREGATION_THRESHOLDS) -> dict:
    """Patient-level accuracy, macro-F1 and AUC of the alternative aggregation rules (Supplementary Table S5).

    Majority voting (ties malignant), the mean and the maximum image probability, and confidence-weighted
    voting, which restricts the vote to images whose predicted-class probability reaches the threshold and
    falls back to majority voting for a patient with no such image.
    """
    result = {}
    for split, images in predictions.groupby("split", sort=False):
        truth = images.groupby("patient", sort=False).y_true.first()
        grouped = images.groupby("patient", sort=False)
        n_malignant, n_images = grouped.y_pred.sum(), grouped.y_pred.count()
        majority = (2 * n_malignant >= n_images).astype(int)
        mean_p, max_p = grouped.p_malignant.mean(), grouped.p_malignant.max()

        rules = {"Majority vote": _rule_scores(truth, majority, mean_p),
                 "Mean probability": _rule_scores(truth, (mean_p >= 0.5).astype(int), mean_p),
                 "Max probability": _rule_scores(truth, (max_p >= 0.5).astype(int), max_p)}
        confidence = np.maximum(images.p_malignant, 1 - images.p_malignant)
        for threshold in thresholds:
            kept = images[confidence >= threshold].groupby("patient", sort=False)
            votes, counts = kept.y_pred.sum().reindex(truth.index), kept.y_pred.count().reindex(truth.index)
            prediction, score = majority.copy(), mean_p.copy()
            enough = counts.notna() & (counts > 0)
            prediction[enough] = (2 * votes[enough] >= counts[enough]).astype(int)
            score[enough] = kept.p_malignant.mean().reindex(truth.index)[enough]
            rules[f"Weighted vote >={threshold:.2f}"] = _rule_scores(truth, prediction, score)
        result[split] = rules
    return result


def write_outputs(predictions: pd.DataFrame, out_dir: Path, extra: dict | None = None) -> dict:
    """Write stage3_image_predictions.csv, stage3_patient_predictions.csv and stage3_metrics.json."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions[IMAGE_COLUMNS].to_csv(out_dir / "stage3_image_predictions.csv", index=False)
    patient_majority(predictions).to_csv(out_dir / "stage3_patient_predictions.csv", index=False)
    metrics = {**(extra or {}), "splits": split_metrics(predictions)}
    (out_dir / "stage3_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def train_one_epoch(model: nn.Module, loader, criterion: nn.Module, optimizer, device: str) -> float:
    """One pass over the training split; returns the mean loss."""
    model.train()
    total, steps = 0.0, 0
    for batch in tqdm(loader, desc="Train", leave=False):
        images, labels = batch[0].to(device).float(), batch[1].to(device)
        logits = model(images) if len(batch) == 3 else model(images, batch[2].to(device).float())
        loss = criterion(logits.float(), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += float(loss.detach())
        steps += 1
    return total / max(steps, 1)


def train(cfg: TrainConfig, frame: pd.DataFrame, boxes: dict, out_dir: Path, features: pd.DataFrame | None = None,
          mean: np.ndarray | None = None, std: np.ndarray | None = None, device: str = "cpu") -> dict:
    """Train on the train split, select on image-level validation macro-F1, then evaluate validation and test.

    This is the protocol described in the Methods. The released checkpoints were trained with it on the
    original cluster; a rerun here reaches comparable but not bit-identical weights.
    """
    out_dir = Path(out_dir)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    splits = {name: frame[frame.split == name].reset_index(drop=True) for name in ("train", "val", "test")}

    def loader(name: str, shuffle: bool):
        dataset = Stage3Dataset(splits[name], boxes, features, mean, std)
        return make_loader(dataset, cfg.batch_size, cfg.num_workers, shuffle=shuffle)

    train_loader, val_loader, test_loader = loader("train", True), loader("val", False), loader("test", False)

    seed_everything(seeds.resolve(cfg.seed))
    model = stage3_model(cfg.configuration, pretrained=cfg.pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    best_path = out_dir / "checkpoints" / f"stage3_config{cfg.configuration}_best.pt"
    history, best_score, best_epoch = [], -1.0, 0

    for epoch in range(1, cfg.epochs + 1):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_pred = evaluate(model, val_loader, splits["val"], device)
        val_stats = {k: v for k, v in split_metrics(val_pred)["val"].items() if not k.endswith("matrix")}
        history.append({"epoch": epoch, "train_loss": loss, **{f"val_{k}": v for k, v in val_stats.items()}})
        pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
        print(f"epoch {epoch}: train_loss={loss:.4f}, " +
              ", ".join(f"{k}={v:.4f}" for k, v in val_stats.items() if isinstance(v, float)))
        if val_stats["image_macro_f1"] > best_score:
            best_score, best_epoch = val_stats["image_macro_f1"], epoch
            torch.save(model.state_dict(), best_path.with_suffix(".tmp"))
            best_path.with_suffix(".tmp").replace(best_path)

    load_weights(model, best_path)
    predictions = pd.concat([evaluate(model, val_loader, splits["val"], device),
                             evaluate(model, test_loader, splits["test"], device)], ignore_index=True)
    extra = {"configuration": cfg.configuration, "best_epoch": best_epoch,
             "best_val_image_macro_f1": best_score, "checkpoint": best_path.name}
    return write_outputs(predictions, out_dir, extra)


STAGE3_WEIGHTS = {2: "stage3_config2_baseline.pth", 6: "stage3_config6_morphological_fusion.pth"}


def default_paths(configuration: int) -> dict:
    """Released inputs of one configuration: weights, descriptor normalization and detector boxes."""
    return {"weights": paths.WEIGHTS / STAGE3_WEIGHTS[configuration],
            "normalization": paths.WEIGHTS / "stage3_config6_feature_normalization.json",
            "boxes": paths.DATA / "detector_boxes",
            "features": paths.FEATURES}
