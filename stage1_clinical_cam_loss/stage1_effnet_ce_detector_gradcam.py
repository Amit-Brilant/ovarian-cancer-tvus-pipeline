#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from ultralytics import YOLO

from src.config import DataConfig, resolve_seed
from src.data_loader import create_dataloaders as create_dataloaders_stage1
from models.networksVer3 import EfficientNet_CE


@dataclass
class RunConfig:
    data_root: str = "/path/to/project/Thesis code"
    detector_checkpoint: str = "/path/to/project/mnn/best/det_best.pt"
    output_root: str = "/path/to/project/mnn/stage1_effnet_ce_detector_gradcam_healthy_sick"
    checkpoint_in: str = ""
    variant: str = "b7"
    batch_size: int = 8
    num_workers: int = 0
    epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    patience: int = 12
    image_size: int = 224
    detector_imgsz: int = 640
    detector_conf: float = 0.25
    detector_iou: float = 0.45
    crop_padding: float = 0.10
    missing_box_mode: str = "center_crop"
    healthy_crop_mode: str = "trimmed_center"
    metadata_key: str = "patient_code"
    seed: Optional[int] = None
    pretrained: bool = True
    save_gradcam_each_epoch: bool = True
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(
        description="Stage 1 EfficientNet_CE with detector ROI crop and per-epoch validation Grad-CAM."
    )
    parser.add_argument("--data-root", default=RunConfig.data_root)
    parser.add_argument("--detector-checkpoint", default=RunConfig.detector_checkpoint)
    parser.add_argument("--output-root", default=RunConfig.output_root)
    parser.add_argument("--checkpoint-in", default=RunConfig.checkpoint_in)
    parser.add_argument("--variant", default=RunConfig.variant)
    parser.add_argument("--batch-size", type=int, default=RunConfig.batch_size)
    parser.add_argument("--num-workers", type=int, default=RunConfig.num_workers)
    parser.add_argument("--epochs", type=int, default=RunConfig.epochs)
    parser.add_argument("--learning-rate", type=float, default=RunConfig.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=RunConfig.weight_decay)
    parser.add_argument("--patience", type=int, default=RunConfig.patience)
    parser.add_argument("--image-size", type=int, default=RunConfig.image_size)
    parser.add_argument("--detector-imgsz", type=int, default=RunConfig.detector_imgsz)
    parser.add_argument("--detector-conf", type=float, default=RunConfig.detector_conf)
    parser.add_argument("--detector-iou", type=float, default=RunConfig.detector_iou)
    parser.add_argument("--crop-padding", type=float, default=RunConfig.crop_padding)
    parser.add_argument("--missing-box-mode", choices=["center_crop", "skip"], default=RunConfig.missing_box_mode)
    parser.add_argument(
        "--healthy-crop-mode",
        choices=["trimmed_center", "center_crop", "full_image"],
        default=RunConfig.healthy_crop_mode,
    )
    parser.add_argument("--metadata-key", choices=["patient_code", "patient_visit_id"], default=RunConfig.metadata_key)
    parser.add_argument("--seed", type=int, default=None, help="random seed (or set SEED)")
    parser.add_argument("--train-ratio", type=float, default=RunConfig.train_ratio)
    parser.add_argument("--val-ratio", type=float, default=RunConfig.val_ratio)
    parser.add_argument("--test-ratio", type=float, default=RunConfig.test_ratio)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-gradcam", dest="save_gradcam_each_epoch", action="store_false")
    args = parser.parse_args()
    return RunConfig(
        data_root=str(args.data_root),
        detector_checkpoint=str(args.detector_checkpoint),
        output_root=str(args.output_root),
        checkpoint_in=str(args.checkpoint_in),
        variant=str(args.variant),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        epochs=int(args.epochs),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        patience=int(args.patience),
        image_size=int(args.image_size),
        detector_imgsz=int(args.detector_imgsz),
        detector_conf=float(args.detector_conf),
        detector_iou=float(args.detector_iou),
        crop_padding=float(args.crop_padding),
        missing_box_mode=str(args.missing_box_mode),
        healthy_crop_mode=str(args.healthy_crop_mode),
        metadata_key=str(args.metadata_key),
        seed=resolve_seed(args.seed),
        pretrained=not bool(args.no_pretrained),
        save_gradcam_each_epoch=bool(args.save_gradcam_each_epoch),
        train_ratio=float(args.train_ratio),
        val_ratio=float(args.val_ratio),
        test_ratio=float(args.test_ratio),
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_name(raw: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(raw))


def resolve_checkpoint_path(
    checkpoint_in: str,
    *,
    data_root: str | None = None,
    fallback_search_root: str | None = None,
) -> Path:
    raw = str(checkpoint_in).strip()
    candidates: List[Path] = []
    seen: set[str] = set()

    def add_candidate(path: Path | None) -> None:
        if path is None:
            return
        normalized = str(path.expanduser())
        if normalized in seen:
            return
        seen.add(normalized)
        candidates.append(Path(normalized))

    if raw:
        raw_path = Path(raw).expanduser()
        add_candidate(raw_path)
        if not raw_path.is_absolute():
            add_candidate(Path.cwd() / raw_path)
            add_candidate(Path(__file__).resolve().parent / raw_path)
            if data_root:
                data_root_path = Path(data_root).expanduser()
                add_candidate(data_root_path / raw_path)
                add_candidate(data_root_path.parent / raw_path)
            if fallback_search_root:
                search_root_path = Path(fallback_search_root).expanduser()
                add_candidate(search_root_path / raw_path)
                add_candidate(search_root_path.parent / raw_path)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    search_roots: List[Path] = []
    if fallback_search_root:
        search_roots.append(Path(fallback_search_root).expanduser())
    if data_root:
        data_root_path = Path(data_root).expanduser()
        search_roots.append(data_root_path.parent / "mnn" / "stage1_effnet_ce_detector_gradcam_healthy_sick")
        search_roots.append(data_root_path.parent / "stage1_effnet_ce_detector_gradcam_healthy_sick")

    best_matches: List[Path] = []
    for search_root in search_roots:
        if not search_root.exists():
            continue
        best_matches.extend(search_root.glob("run_*/checkpoints/best_effnet_ce_detector_gradcam.h"))
        best_matches.extend(search_root.glob("best_effnet_ce_detector_gradcam.h"))

    best_matches = [path for path in best_matches if path.exists()]
    if best_matches:
        best_matches.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        return best_matches[0].resolve()

    tried = "\n".join(f"- {path}" for path in candidates) if candidates else "- <no explicit checkpoint path provided>"
    searched = "\n".join(f"- {path}" for path in search_roots) if search_roots else "- <no fallback search roots provided>"
    raise FileNotFoundError(
        "Checkpoint not found.\n"
        f"Tried:\n{tried}\n"
        f"Searched for latest checkpoint under:\n{searched}"
    )


def make_output_dirs(root: Path) -> Dict[str, Path]:
    run_id = time.strftime("%Y%m%d-%H%M%S")
    run_root = root / f"run_{run_id}"
    checkpoints = run_root / "checkpoints"
    gradcams = run_root / "gradcam_val"
    reports = run_root / "reports"
    for path in [run_root, checkpoints, gradcams, reports]:
        path.mkdir(parents=True, exist_ok=True)
    return {
        "run_root": run_root,
        "checkpoints": checkpoints,
        "gradcams": gradcams,
        "reports": reports,
    }


def build_dataframes(cfg: RunConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_cfg = DataConfig()
    data_cfg.data_root = cfg.data_root
    data_cfg.batch_size = cfg.batch_size
    data_cfg.num_workers = cfg.num_workers
    data_cfg.seed = cfg.seed
    data_cfg.train_ratio = cfg.train_ratio
    data_cfg.val_ratio = cfg.val_ratio
    data_cfg.test_ratio = cfg.test_ratio
    data_cfg.image_size = (cfg.image_size, cfg.image_size)
    data_cfg.use_augmentation = True
    _train_loader, _val_loader, _test_loader, train_df, val_df, test_df = create_dataloaders_stage1(
        data_cfg,
        flag_patient_visit_id=(cfg.metadata_key == "patient_code"),
    )
    return train_df.copy(), val_df.copy(), test_df.copy()


def central_clinical_crop_with_bounds(
    image_rgb: np.ndarray,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    height, width = image_rgb.shape[:2]
    x_margin = int(round(width * 0.08))
    y_top = int(round(height * 0.10))
    y_bottom = int(round(height * 0.04))
    x2 = max(x_margin + 1, width - x_margin)
    y2 = max(y_top + 1, height - y_bottom)
    crop = image_rgb[y_top:y2, x_margin:x2]
    if crop.size:
        return crop, (x_margin, y_top, x2, y2)
    return image_rgb, (0, 0, width, height)


def central_clinical_crop(image_rgb: np.ndarray) -> np.ndarray:
    crop, _bounds = central_clinical_crop_with_bounds(image_rgb)
    return crop


def trim_dark_border_with_bounds(
    image_rgb: np.ndarray,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    mask = gray > 8
    coords = np.argwhere(mask)
    height, width = image_rgb.shape[:2]
    if coords.size == 0:
        return image_rgb, (0, 0, width, height)
    y1, x1 = coords.min(axis=0)
    y2, x2 = coords.max(axis=0) + 1
    pad_x = int(round(width * 0.015))
    pad_y = int(round(height * 0.015))
    x1 = max(0, int(x1) - pad_x)
    y1 = max(0, int(y1) - pad_y)
    x2 = min(width, int(x2) + pad_x)
    y2 = min(height, int(y2) + pad_y)
    crop = image_rgb[y1:y2, x1:x2]
    if crop.size:
        return crop, (x1, y1, x2, y2)
    return image_rgb, (0, 0, width, height)


def trim_dark_border(image_rgb: np.ndarray) -> np.ndarray:
    crop, _bounds = trim_dark_border_with_bounds(image_rgb)
    return crop


def crop_box_with_padding_and_bounds(
    image_rgb: np.ndarray,
    box: Tuple[int, int, int, int],
    padding: float,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    height, width = image_rgb.shape[:2]
    x1, y1, x2, y2 = box
    box_w = max(1, int(x2) - int(x1))
    box_h = max(1, int(y2) - int(y1))
    pad_x = int(round(box_w * padding))
    pad_y = int(round(box_h * padding))
    x1 = max(0, int(x1) - pad_x)
    y1 = max(0, int(y1) - pad_y)
    x2 = min(width, int(x2) + pad_x)
    y2 = min(height, int(y2) + pad_y)
    crop = image_rgb[y1:y2, x1:x2]
    if crop.size:
        return crop, (x1, y1, x2, y2)
    return image_rgb, (0, 0, width, height)


def crop_box_with_padding(image_rgb: np.ndarray, box: Tuple[int, int, int, int], padding: float) -> np.ndarray:
    crop, _bounds = crop_box_with_padding_and_bounds(image_rgb, box, padding)
    return crop


def focus_image_with_bounds(
    image_rgb: np.ndarray,
    *,
    label_value: int,
    box: Optional[Tuple[int, int, int, int]],
    crop_padding: float,
    healthy_crop_mode: str,
    use_crop: bool = True,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    height, width = image_rgb.shape[:2]
    full_bounds = (0, 0, width, height)

    if not use_crop:
        return image_rgb, full_bounds

    if label_value == 0 and healthy_crop_mode == "full_image":
        return image_rgb, full_bounds

    if label_value == 1 and box is not None:
        return crop_box_with_padding_and_bounds(image_rgb, box, crop_padding)

    if label_value == 0 and healthy_crop_mode == "center_crop":
        return central_clinical_crop_with_bounds(image_rgb)

    trimmed, (trim_x1, trim_y1, _trim_x2, _trim_y2) = trim_dark_border_with_bounds(image_rgb)
    focused, (crop_x1, crop_y1, crop_x2, crop_y2) = central_clinical_crop_with_bounds(trimmed)
    bounds = (
        trim_x1 + crop_x1,
        trim_y1 + crop_y1,
        trim_x1 + crop_x2,
        trim_y1 + crop_y2,
    )
    return focused, bounds


def get_transforms(image_size: int, is_training: bool) -> transforms.Compose:
    if is_training:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=7),
                transforms.ColorJitter(brightness=0.10, contrast=0.16),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.RandomErasing(p=0.18, scale=(0.01, 0.08), ratio=(0.4, 2.5)),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def precompute_detector_boxes(
    detector: YOLO,
    dataframe: pd.DataFrame,
    *,
    imgsz: int,
    conf: float,
    iou: float,
    device: str | int,
) -> Dict[str, Tuple[int, int, int, int]]:
    boxes: Dict[str, Tuple[int, int, int, int]] = {}
    for path in tqdm(dataframe["filepath"].astype(str).tolist(), desc="Precompute detector boxes"):
        try:
            image_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
            if image_bgr is None:
                continue
            result = detector.predict(
                source=[image_bgr],
                imgsz=int(imgsz),
                conf=conf,
                iou=iou,
                device=device,
                verbose=False,
            )[0]
            if result.boxes is None or len(result.boxes) == 0:
                continue
            xyxy = result.boxes.xyxy.detach().cpu().numpy()
            scores = result.boxes.conf.detach().cpu().numpy()
            best_idx = int(np.argmax(scores))
            x1, y1, x2, y2 = xyxy[best_idx][:4].astype(int).tolist()
            boxes[path] = (int(x1), int(y1), int(x2), int(y2))
        except Exception:
            continue
    return boxes


class DetectorCropDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        *,
        transform: transforms.Compose,
        crop_boxes: Dict[str, Tuple[int, int, int, int]],
        metadata_key: str,
        crop_padding: float,
        missing_box_mode: str,
        healthy_crop_mode: str,
        use_crop: bool = True,
    ) -> None:
        self.transform = transform
        self.crop_boxes = crop_boxes
        self.metadata_key = metadata_key
        self.crop_padding = float(crop_padding)
        self.missing_box_mode = str(missing_box_mode)
        self.healthy_crop_mode = str(healthy_crop_mode)
        self.use_crop = bool(use_crop)
        df = dataframe.reset_index(drop=True).copy()
        if self.use_crop and self.missing_box_mode == "skip":
            df = df[df["filepath"].astype(str).isin(self.crop_boxes.keys())].reset_index(drop=True)
        self.dataframe = df
        self.image_paths = self.dataframe["filepath"].astype(str).tolist()

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int):
        row = self.dataframe.iloc[idx]
        filepath = str(row["filepath"])
        label_value = int(row["label"])
        image_rgb = np.asarray(Image.open(filepath).convert("RGB"))
        box = self.crop_boxes.get(filepath)

        # Stage 1 is healthy vs sick. Keep the crop bounds so Grad-CAM can later
        # be projected back onto the full-resolution source image.
        focused, _focus_bounds = focus_image_with_bounds(
            image_rgb,
            label_value=label_value,
            box=box,
            crop_padding=self.crop_padding,
            healthy_crop_mode=self.healthy_crop_mode,
            use_crop=self.use_crop,
        )
        tensor = self.transform(Image.fromarray(focused.astype(np.uint8), mode="RGB"))
        label = torch.tensor(label_value, dtype=torch.long)
        metadata = str(row[self.metadata_key])
        return tensor, label, metadata, filepath


def build_loaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    crop_boxes: Dict[str, Tuple[int, int, int, int]],
    cfg: RunConfig,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset = DetectorCropDataset(
        train_df,
        transform=get_transforms(cfg.image_size, is_training=True),
        crop_boxes=crop_boxes,
        metadata_key=cfg.metadata_key,
        crop_padding=cfg.crop_padding,
        missing_box_mode=cfg.missing_box_mode,
        healthy_crop_mode=cfg.healthy_crop_mode,
    )
    val_dataset = DetectorCropDataset(
        val_df,
        transform=get_transforms(cfg.image_size, is_training=False),
        crop_boxes=crop_boxes,
        metadata_key=cfg.metadata_key,
        crop_padding=cfg.crop_padding,
        missing_box_mode=cfg.missing_box_mode,
        healthy_crop_mode=cfg.healthy_crop_mode,
    )
    test_dataset = DetectorCropDataset(
        test_df,
        transform=get_transforms(cfg.image_size, is_training=False),
        crop_boxes=crop_boxes,
        metadata_key=cfg.metadata_key,
        crop_padding=cfg.crop_padding,
        missing_box_mode=cfg.missing_box_mode,
        healthy_crop_mode=cfg.healthy_crop_mode,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_class_weights(train_df: pd.DataFrame, device: torch.device) -> torch.Tensor:
    counts = train_df["label"].astype(int).value_counts().to_dict()
    n0 = max(int(counts.get(0, 0)), 1)
    n1 = max(int(counts.get(1, 0)), 1)
    total = n0 + n1
    weights = torch.tensor([total / (2.0 * n0), total / (2.0 * n1)], dtype=torch.float32, device=device)
    return weights


def majority_vote(y_true: List[int], y_pred: List[int], ids: List[str]) -> Dict[str, object]:
    grouped_pred: Dict[str, List[int]] = defaultdict(list)
    grouped_true: Dict[str, int] = {}
    for truth, pred, item_id in zip(y_true, y_pred, ids):
        grouped_pred[str(item_id)].append(int(pred))
        if str(item_id) not in grouped_true:
            grouped_true[str(item_id)] = int(truth)

    maj_true: List[int] = []
    maj_pred: List[int] = []
    for item_id, preds in grouped_pred.items():
        maj_pred.append(int(Counter(preds).most_common(1)[0][0]))
        maj_true.append(int(grouped_true[item_id]))

    if len(maj_true) == 0:
        return {"acc": 0.0, "mf1": 0.0, "y_true": np.array([], dtype=int), "y_pred": np.array([], dtype=int)}

    y_true_np = np.asarray(maj_true, dtype=int)
    y_pred_np = np.asarray(maj_pred, dtype=int)
    return {
        "acc": float(accuracy_score(y_true_np, y_pred_np) * 100.0),
        "mf1": float(f1_score(y_true_np, y_pred_np, average="macro", zero_division=0) * 100.0),
        "y_true": y_true_np,
        "y_pred": y_pred_np,
    }


def find_gradcam_target_layer(model: nn.Module) -> nn.Module:
    if hasattr(model, "backbone") and hasattr(model.backbone, "features"):
        for module in reversed(list(model.backbone.features.modules())):
            if isinstance(module, nn.Conv2d):
                return module
    raise RuntimeError("Could not find an EfficientNet conv layer for Grad-CAM.")


class GradCAMHelper:
    def __init__(self, model: nn.Module):
        self.model = model
        self.target_layer = find_gradcam_target_layer(model)
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self.handle_forward = self.target_layer.register_forward_hook(self._save_activation)
        self.handle_backward = self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _module, _inputs, output):
        self.activations = output.detach()

    def _save_gradient(self, _module, _grad_input, grad_output):
        if grad_output and grad_output[0] is not None:
            self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, target_class: int) -> Tuple[np.ndarray, bool]:
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        self.activations = None
        self.gradients = None

        with torch.enable_grad():
            logits = self.model(input_tensor)
            score = logits[:, int(target_class)].sum()
            score.backward()

        if self.activations is None or self.gradients is None:
            blank = np.zeros((input_tensor.shape[-2], input_tensor.shape[-1]), dtype=np.float32)
            return blank, False

        activation = self.activations[0]
        gradient = self.gradients[0]
        weights = gradient.mean(dim=(1, 2), keepdim=True)
        raw_cam = (weights * activation).sum(dim=0)
        cam = torch.relu(raw_cam)
        if float(cam.max().item()) <= 0.0:
            cam = torch.abs(raw_cam)
            if float(cam.max().item()) <= 0.0:
                blank = np.zeros((input_tensor.shape[-2], input_tensor.shape[-1]), dtype=np.float32)
                return blank, False

        cam = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0),
            size=input_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        cam = cam - cam.min()
        cam = cam / max(float(cam.max().item()), 1e-8)
        return cam.detach().cpu().numpy().astype(np.float32), True

    def generate_batch(self, input_tensor: torch.Tensor) -> Tuple[torch.Tensor, List[np.ndarray], List[bool]]:
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        self.activations = None
        self.gradients = None

        with torch.enable_grad():
            logits = self.model(input_tensor)
            target_classes = torch.argmax(logits.detach(), dim=1)
            score = logits.gather(1, target_classes.unsqueeze(1)).sum()
            score.backward()

        batch_size = int(input_tensor.shape[0])
        height, width = int(input_tensor.shape[-2]), int(input_tensor.shape[-1])
        blank = np.zeros((height, width), dtype=np.float32)
        if self.activations is None or self.gradients is None:
            return logits.detach(), [blank.copy() for _ in range(batch_size)], [False] * batch_size

        activation = self.activations
        gradient = self.gradients
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        raw_cams = (weights * activation).sum(dim=1)
        cams = torch.relu(raw_cams)

        non_positive = cams.flatten(1).amax(dim=1) <= 0.0
        if bool(non_positive.any()):
            cams[non_positive] = torch.abs(raw_cams[non_positive])

        cams = F.interpolate(
            cams.unsqueeze(1),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )[:, 0]

        result_cams: List[np.ndarray] = []
        availability: List[bool] = [False] * batch_size
        for index in range(batch_size):
            cam = cams[index]
            cam = cam - cam.min()
            max_value = float(cam.max().item())
            if max_value <= 0.0:
                result_cams.append(blank.copy())
                continue
            cam = cam / max(max_value, 1e-8)
            result_cams.append(cam.detach().cpu().numpy().astype(np.float32))
            availability[index] = True

        return logits.detach(), result_cams, availability

    def close(self) -> None:
        self.handle_forward.remove()
        self.handle_backward.remove()


def denorm_image(image_tensor: torch.Tensor) -> np.ndarray:
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    img = image_tensor.detach().cpu().numpy()
    img = np.clip(img * std + mean, 0.0, 1.0)
    img = (img.transpose(1, 2, 0) * 255.0).astype(np.uint8)
    return img


def _wrap_text_to_width(
    text: str,
    *,
    font: int,
    font_scale: float,
    thickness: int,
    max_width: int,
    max_lines: int = 3,
) -> List[str]:
    words = str(text).split()
    if not words:
        return [""]

    lines: List[str] = []
    current = words[0]

    for word in words[1:]:
        candidate = f"{current} {word}"
        candidate_width = cv2.getTextSize(candidate, font, font_scale, thickness)[0][0]
        if candidate_width <= max_width:
            current = candidate
            continue
        lines.append(current)
        current = word

    lines.append(current)

    if len(lines) <= max_lines:
        return lines

    trimmed = lines[: max_lines - 1]
    remainder = " ".join(lines[max_lines - 1 :])
    while remainder:
        candidate = remainder + "..."
        candidate_width = cv2.getTextSize(candidate, font, font_scale, thickness)[0][0]
        if candidate_width <= max_width:
            trimmed.append(candidate)
            return trimmed
        remainder = remainder[:-1]

    trimmed.append("...")
    return trimmed


def write_gradcam_overlay(
    image_rgb: np.ndarray,
    cam: np.ndarray,
    title: str,
    save_path: Path,
    available: bool,
    overlay_bounds: Optional[Tuple[int, int, int, int]] = None,
) -> None:
    height, width = image_rgb.shape[:2]
    cam = np.nan_to_num(cam.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    cam = np.clip(cam, 0.0, 1.0)
    if cam.shape != (height, width):
        cam = cv2.resize(cam, (width, height), interpolation=cv2.INTER_LINEAR)
    base_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    overlay = base_bgr.copy()
    if available:
        heatmap = cv2.applyColorMap((cam * 255.0).astype(np.uint8), cv2.COLORMAP_JET)
        blended = cv2.addWeighted(base_bgr, 0.55, heatmap, 0.45, 0)
        if overlay_bounds is None:
            overlay = blended
        else:
            x1, y1, x2, y2 = overlay_bounds
            x1 = max(0, min(width, int(x1)))
            y1 = max(0, min(height, int(y1)))
            x2 = max(x1, min(width, int(x2)))
            y2 = max(y1, min(height, int(y2)))
            overlay[y1:y2, x1:x2] = blended[y1:y2, x1:x2]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.34, min(width / 520.0, 0.62))
    thickness = 1 if width < 320 else 2
    pad_x = max(8, int(round(width * 0.04)))
    pad_y = max(8, int(round(height * 0.04)))
    max_text_width = max(40, width - (2 * pad_x))

    raw_lines = title.splitlines() or [title]
    if not available:
        raw_lines.append("Grad-CAM unavailable")

    lines: List[str] = []
    for raw_line in raw_lines:
        clean_line = raw_line.strip()
        if not clean_line:
            continue
        remaining = 3 - len(lines)
        if remaining <= 0:
            break
        wrapped = _wrap_text_to_width(
            clean_line,
            font=font,
            font_scale=font_scale,
            thickness=thickness,
            max_width=max_text_width,
            max_lines=remaining,
        )
        lines.extend(wrapped[:remaining])
    if not lines:
        lines = [""]

    text_height = cv2.getTextSize("Ag", font, font_scale, thickness)[0][1]
    line_gap = max(6, int(round(text_height * 0.55)))
    strip_height = min(
        height,
        pad_y + (len(lines) * text_height) + ((len(lines) - 1) * line_gap) + pad_y,
    )
    cv2.rectangle(overlay, (0, 0), (width, strip_height), (0, 0, 0), thickness=-1)

    y = pad_y + text_height
    for line in lines:
        cv2.putText(
            overlay,
            line,
            (pad_x, y),
            font,
            font_scale,
            (0, 0, 0),
            thickness + 3,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            line,
            (pad_x, y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        y += text_height + line_gap

    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), overlay)


def save_val_gradcams(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    epoch: int,
    root: Path,
) -> List[Dict[str, object]]:
    model.eval()
    gradcam = GradCAMHelper(model)
    epoch_dir = root / f"epoch_{int(epoch):03d}"
    records: List[Dict[str, object]] = []
    label_names = {0: "healthy", 1: "sick"}
    sample_number = 0
    dataset = loader.dataset
    crop_boxes = getattr(dataset, "crop_boxes", {})
    crop_padding = float(getattr(dataset, "crop_padding", 0.0))
    healthy_crop_mode = str(getattr(dataset, "healthy_crop_mode", "full_image"))
    use_crop = bool(getattr(dataset, "use_crop", True))

    try:
        for batch in tqdm(loader, total=len(loader), desc=f"GradCAM val epoch {epoch}"):
            images, labels, metadata, filepaths = batch
            logits, cams, availability = gradcam.generate_batch(images.to(device).float())
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            for index in range(images.shape[0]):
                sample_number += 1
                true_cls = int(labels[index].item())
                pred_cls = int(preds[index].item())
                prob = float(probs[index, pred_cls].detach().cpu().item())
                cam = cams[index]
                available = availability[index]

                filepath = str(filepaths[index])
                full_image_rgb = np.asarray(Image.open(filepath).convert("RGB"))
                _focused, focus_bounds = focus_image_with_bounds(
                    full_image_rgb,
                    label_value=true_cls,
                    box=crop_boxes.get(filepath),
                    crop_padding=crop_padding,
                    healthy_crop_mode=healthy_crop_mode,
                    use_crop=use_crop,
                )
                full_height, full_width = full_image_rgb.shape[:2]
                x1, y1, x2, y2 = focus_bounds
                projected_cam = np.zeros((full_height, full_width), dtype=np.float32)
                if x2 > x1 and y2 > y1:
                    projected_cam[y1:y2, x1:x2] = cv2.resize(
                        cam,
                        (x2 - x1, y2 - y1),
                        interpolation=cv2.INTER_LINEAR,
                    )
                sample_id = safe_name(str(metadata[index]))
                file_stem = safe_name(Path(filepath).stem)
                file_name = (
                    f"epoch_{int(epoch):03d}_{sample_number:05d}_"
                    f"id-{sample_id}_true-{label_names.get(true_cls, str(true_cls))}_"
                    f"pred-{label_names.get(pred_cls, str(pred_cls))}_"
                    f"score-{prob:.3f}_{file_stem}.png"
                )
                save_path = epoch_dir / file_name
                title = (
                    f"E{int(epoch)} | ID {sample_id}\n"
                    f"T: {label_names.get(true_cls, true_cls)} | P: {label_names.get(pred_cls, pred_cls)}\n"
                    f"S: {prob:.3f}"
                )
                write_gradcam_overlay(
                    full_image_rgb,
                    projected_cam,
                    title,
                    save_path,
                    available,
                    overlay_bounds=focus_bounds,
                )

                print(
                    "VAL_GRADCAM | "
                    f"epoch={epoch}, sample={sample_number}, id={sample_id}, "
                    f"true={true_cls}, pred={pred_cls}, score={prob:.6f}, "
                    f"available={available}, path={save_path}"
                )

                records.append(
                    {
                        "epoch": int(epoch),
                        "sample_number": int(sample_number),
                        "metadata": str(metadata[index]),
                        "filepath": str(filepaths[index]),
                        "focus_bounds_xyxy": [int(value) for value in focus_bounds],
                        "gradcam_space": "full_image",
                        "true_cls": int(true_cls),
                        "pred_cls": int(pred_cls),
                        "score": float(prob),
                        "gradcam_available": bool(available),
                        "path": str(save_path),
                    }
                )
    finally:
        gradcam.close()

    pd.DataFrame(records).to_csv(epoch_dir / "summary.csv", index=False)
    print(f"VAL_GRADCAM_DONE | epoch={epoch} | saved={len(records)} | dir={epoch_dir}")
    return records


def evaluate_classifier(
    model: nn.Module,
    loader: DataLoader,
    *,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, object]:
    model.eval()
    total_loss = 0.0
    steps = 0
    y_true: List[int] = []
    y_pred: List[int] = []
    ids: List[str] = []

    with torch.no_grad():
        for images, labels, metadata, _filepaths in tqdm(loader, total=len(loader), desc="Evaluate"):
            images = images.to(device).float()
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits.float(), labels)
            preds = torch.argmax(logits, dim=1)

            total_loss += float(loss.detach().cpu().item())
            steps += 1
            y_true.extend(labels.detach().cpu().numpy().astype(int).tolist())
            y_pred.extend(preds.detach().cpu().numpy().astype(int).tolist())
            ids.extend([str(item) for item in metadata])

    image_acc = 100.0 * accuracy_score(y_true, y_pred) if y_true else 0.0
    image_mf1 = 100.0 * f1_score(y_true, y_pred, average="macro", zero_division=0) if y_true else 0.0
    patient = majority_vote(y_true, y_pred, ids)

    return {
        "loss": total_loss / max(steps, 1),
        "image": {
            "acc": float(image_acc),
            "mf1": float(image_mf1),
            "y_true": np.asarray(y_true, dtype=int),
            "y_pred": np.asarray(y_pred, dtype=int),
        },
        "patient_majority": patient,
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    *,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    y_true: List[int] = []
    y_pred: List[int] = []
    steps = 0

    for images, labels, _metadata, _filepaths in tqdm(loader, total=len(loader), desc="Train"):
        images = images.to(device).float()
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits.float(), labels)
        loss.backward()
        optimizer.step()

        preds = torch.argmax(logits, dim=1)
        total_loss += float(loss.detach().cpu().item())
        steps += 1
        y_true.extend(labels.detach().cpu().numpy().astype(int).tolist())
        y_pred.extend(preds.detach().cpu().numpy().astype(int).tolist())

    acc = 100.0 * accuracy_score(y_true, y_pred) if y_true else 0.0
    mf1 = 100.0 * f1_score(y_true, y_pred, average="macro", zero_division=0) if y_true else 0.0
    return {
        "loss": total_loss / max(steps, 1),
        "acc": float(acc),
        "mf1": float(mf1),
    }


def dump_split_stats(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    crop_boxes: Dict[str, Tuple[int, int, int, int]],
) -> Dict[str, object]:
    stats: Dict[str, object] = {}
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        paths = df["filepath"].astype(str).tolist()
        with_box = sum(1 for path in paths if path in crop_boxes)
        healthy_df = df[df["label"].astype(int) == 0]
        sick_df = df[df["label"].astype(int) == 1]
        stats[name] = {
            "rows": int(len(df)),
            "with_box": int(with_box),
            "without_box": int(len(df) - with_box),
            "healthy_rows": int(len(healthy_df)),
            "healthy_with_box": int(sum(1 for path in healthy_df["filepath"].astype(str).tolist() if path in crop_boxes)),
            "sick_rows": int(len(sick_df)),
            "sick_with_box": int(sum(1 for path in sick_df["filepath"].astype(str).tolist() if path in crop_boxes)),
        }
    return stats


def save_final_reports(
    reports_dir: Path,
    *,
    split_name: str,
    metrics: Dict[str, object],
) -> None:
    image_true = metrics["image"]["y_true"]
    image_pred = metrics["image"]["y_pred"]
    patient_true = metrics["patient_majority"]["y_true"]
    patient_pred = metrics["patient_majority"]["y_pred"]

    label_names = ["healthy", "sick"]
    payload = {
        "split_name": split_name,
        "image_acc_percent": float(metrics["image"]["acc"]),
        "image_mf1_percent": float(metrics["image"]["mf1"]),
        "patient_acc_percent": float(metrics["patient_majority"]["acc"]),
        "patient_mf1_percent": float(metrics["patient_majority"]["mf1"]),
        "image_confusion_matrix": confusion_matrix(image_true, image_pred).tolist() if len(image_true) else [],
        "patient_confusion_matrix": confusion_matrix(patient_true, patient_pred).tolist() if len(patient_true) else [],
    }
    (reports_dir / f"{split_name}_metrics.json").write_text(json.dumps(payload, indent=2))

    lines: List[str] = []
    lines.append(f"{split_name.upper()} IMAGE-LEVEL")
    lines.append(f"ACC: {metrics['image']['acc'] / 100.0:.6f}")
    lines.append(f"mF1: {metrics['image']['mf1'] / 100.0:.6f}")
    if len(image_true):
        lines.append(classification_report(image_true, image_pred, target_names=label_names, digits=4, zero_division=0))
    lines.append("")
    lines.append(f"{split_name.upper()} {str('PATIENT/VISIT').replace('/', '-')}-LEVEL")
    lines.append(f"ACC: {metrics['patient_majority']['acc'] / 100.0:.6f}")
    lines.append(f"mF1: {metrics['patient_majority']['mf1'] / 100.0:.6f}")
    if len(patient_true):
        lines.append(classification_report(patient_true, patient_pred, target_names=label_names, digits=4, zero_division=0))
    (reports_dir / f"{split_name}_report.txt").write_text("\n".join(lines))


def main() -> int:
    cfg = parse_args()
    seed_everything(cfg.seed)
    device = get_device()
    output_dirs = make_output_dirs(Path(cfg.output_root))

    print("Run configuration:")
    print(json.dumps(asdict(cfg), indent=2))
    print(f"device={device}")

    train_df, val_df, test_df = build_dataframes(cfg)

    detector = YOLO(cfg.detector_checkpoint)
    det_device: str | int = 0 if device.type == "cuda" else "cpu"
    all_df = pd.concat([train_df, val_df, test_df], axis=0).drop_duplicates(subset=["filepath"]).reset_index(drop=True)
    crop_boxes = precompute_detector_boxes(
        detector,
        all_df,
        imgsz=cfg.detector_imgsz,
        conf=cfg.detector_conf,
        iou=cfg.detector_iou,
        device=det_device,
    )
    split_stats = dump_split_stats(train_df, val_df, test_df, crop_boxes)
    print("Crop coverage:")
    print(json.dumps(split_stats, indent=2))

    train_loader, val_loader, test_loader = build_loaders(
        train_df,
        val_df,
        test_df,
        crop_boxes=crop_boxes,
        cfg=cfg,
    )

    model = EfficientNet_CE(
        variant=cfg.variant,
        num_classes=2,
        dropout=0.2,
        pretrained=cfg.pretrained,
    ).to(device)

    if str(cfg.checkpoint_in).strip():
        checkpoint_path = Path(cfg.checkpoint_in)
        if checkpoint_path.exists():
            state_dict = torch.load(str(checkpoint_path), map_location=device)
            model.load_state_dict(state_dict, strict=True)
            print(f"Loaded checkpoint: {checkpoint_path}")

    class_weights = compute_class_weights(train_df, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    history: List[Dict[str, object]] = []
    best_val_score = -1.0
    best_epoch = 0
    best_state: Optional[Dict[str, torch.Tensor]] = None
    epochs_without_improvement = 0

    for epoch in range(1, cfg.epochs + 1):
        print(f"\n{'=' * 80}")
        print(f"Epoch {epoch}/{cfg.epochs}")
        print(f"{'=' * 80}")

        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        val_metrics = evaluate_classifier(
            model,
            val_loader,
            criterion=criterion,
            device=device,
        )

        gradcam_records: List[Dict[str, object]] = []
        if cfg.save_gradcam_each_epoch:
            gradcam_records = save_val_gradcams(
                model,
                val_loader,
                device=device,
                epoch=epoch,
                root=output_dirs["gradcams"],
            )

        record = {
            "epoch": int(epoch),
            "train_loss": float(train_metrics["loss"]),
            "train_acc": float(train_metrics["acc"]),
            "train_mf1": float(train_metrics["mf1"]),
            "val_loss": float(val_metrics["loss"]),
            "val_image_acc": float(val_metrics["image"]["acc"]),
            "val_image_mf1": float(val_metrics["image"]["mf1"]),
            "val_patient_acc": float(val_metrics["patient_majority"]["acc"]),
            "val_patient_mf1": float(val_metrics["patient_majority"]["mf1"]),
            "gradcam_files": [item["path"] for item in gradcam_records],
        }
        history.append(record)

        print(
            "EPOCH_METRICS | "
            f"epoch={epoch}, "
            f"train_loss={record['train_loss']:.6f}, "
            f"train_acc={record['train_acc']:.6f}, "
            f"train_mf1={record['train_mf1']:.6f}, "
            f"val_loss={record['val_loss']:.6f}, "
            f"val_image_acc={record['val_image_acc']:.6f}, "
            f"val_image_mf1={record['val_image_mf1']:.6f}, "
            f"val_patient_acc={record['val_patient_acc']:.6f}, "
            f"val_patient_mf1={record['val_patient_mf1']:.6f}"
        )

        current_score = float(val_metrics["patient_majority"]["mf1"])
        if current_score > best_val_score:
            best_val_score = current_score
            best_epoch = epoch
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            best_path = output_dirs["checkpoints"] / "best_effnet_ce_detector_gradcam.h"
            torch.save(best_state, str(best_path))
            print(f"BEST_MODEL_UPDATED | epoch={epoch} | val_patient_mf1={best_val_score:.6f} | path={best_path}")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            print(f"NO_IMPROVEMENT | patience_counter={epochs_without_improvement}/{cfg.patience}")

        pd.DataFrame(history).to_csv(output_dirs["reports"] / "history.csv", index=False)

        if epochs_without_improvement >= cfg.patience:
            print(f"EARLY_STOPPING | epoch={epoch} | patience={cfg.patience}")
            break

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)
        model.to(device).eval()

    val_best = evaluate_classifier(model, val_loader, criterion=criterion, device=device)
    test_best = evaluate_classifier(model, test_loader, criterion=criterion, device=device)

    save_final_reports(output_dirs["reports"], split_name="val_best", metrics=val_best)
    save_final_reports(output_dirs["reports"], split_name="test_best", metrics=test_best)

    summary = {
        "config": asdict(cfg),
        "device": str(device),
        "best_epoch": int(best_epoch),
        "best_val_patient_mf1_percent": float(best_val_score),
        "val_best": {
            "image_acc_percent": float(val_best["image"]["acc"]),
            "image_mf1_percent": float(val_best["image"]["mf1"]),
            "patient_acc_percent": float(val_best["patient_majority"]["acc"]),
            "patient_mf1_percent": float(val_best["patient_majority"]["mf1"]),
        },
        "test_best": {
            "image_acc_percent": float(test_best["image"]["acc"]),
            "image_mf1_percent": float(test_best["image"]["mf1"]),
            "patient_acc_percent": float(test_best["patient_majority"]["acc"]),
            "patient_mf1_percent": float(test_best["patient_majority"]["mf1"]),
        },
        "split_crop_coverage": split_stats,
        "run_root": str(output_dirs["run_root"]),
    }
    summary_path = output_dirs["reports"] / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print("\nFINAL_SUMMARY")
    print(json.dumps(summary, indent=2))
    print(f"history_csv={output_dirs['reports'] / 'history.csv'}")
    print(f"summary_json={summary_path}")
    print(f"gradcam_root={output_dirs['gradcams']}")
    print(f"best_checkpoint={output_dirs['checkpoints'] / 'best_effnet_ce_detector_gradcam.h'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
