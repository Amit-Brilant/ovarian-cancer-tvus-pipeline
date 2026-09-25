#!/usr/bin/env python3
"""Train Stage-1 EfficientNet on full images with clinical-attention supervision.

The classifier always receives the complete source image. Detector boxes are
used only to build soft supervision masks; they are never used to crop model
inputs. A class-activation loss penalizes evidence outside the clinical mask.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ColorJitter, InterpolationMode
from torchvision.transforms import functional as TF
from tqdm import tqdm
from ultralytics import YOLO

from models.networksVer3 import EfficientNet_CE
from src.config import resolve_seed
from models.cam_features import FeatureMapCapture
from utils.clinical_cam_loss import clinical_attention_loss
from stage1_persistent_checkpoint_io import save_checkpoint_atomic
from stage1_effnet_ce_detector_gradcam import (
    DetectorCropDataset,
    RunConfig,
    build_dataframes,
    compute_class_weights,
    dump_split_stats,
    get_device,
    get_transforms,
    majority_vote,
    precompute_detector_boxes,
    resolve_checkpoint_path,
    safe_name,
    save_final_reports,
    save_val_gradcams,
    seed_everything,
)


@dataclass
class ClinicalAttentionConfig:
    data_root: str = "/path/to/project/Thesis code"
    detector_checkpoint: str = "/path/to/project/mnn/best/det_best.pt"
    output_root: str = "/path/to/project/mnn/stage1_effnet_ce_full_image_clinical_attention"
    checkpoint_in: str = ""
    variant: str = "b7"
    batch_size: int = 8
    num_workers: int = 4
    epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    patience: int = 12
    image_size: int = 224
    detector_imgsz: int = 640
    detector_conf: float = 0.25
    detector_iou: float = 0.45
    attention_roi_padding: float = 0.30
    clinical_context_weight: float = 0.35
    attention_loss_weight: float = 0.25
    attention_warmup_epochs: int = 5
    artifact_suppression_prob: float = 0.0
    artifact_residual_max: float = 0.20
    suppress_artifacts_at_eval: bool = False
    eval_artifact_mask_source: str = "image_field_v1"
    canonical_field_min_occupancy: float = 0.0
    field_bbox_normalization: bool = False
    zoom_min: float = 0.9
    zoom_max: float = 1.1
    metadata_key: str = "patient_code"
    seed: Optional[int] = None
    split_seed: Optional[int] = None
    attention_mask_mode: str = "correct"
    detector_boxes_cache: str = ""
    pretrained: bool = True
    use_detector_attention: bool = True
    save_gradcam_each_epoch: bool = False
    save_final_gradcam: bool = True
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    input_mode: str = "full_image_clinical_attention"


def parse_args(
    argv: Optional[List[str]] = None,
    *,
    defaults: Optional[Dict[str, object]] = None,
) -> ClinicalAttentionConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Train Stage-1 EfficientNet on full images while penalizing class "
            "activation outside clinically relevant image regions."
        )
    )
    parser.add_argument("--data-root", default=ClinicalAttentionConfig.data_root)
    parser.add_argument("--detector-checkpoint", default=ClinicalAttentionConfig.detector_checkpoint)
    parser.add_argument("--output-root", default=ClinicalAttentionConfig.output_root)
    parser.add_argument("--checkpoint-in", default=ClinicalAttentionConfig.checkpoint_in)
    parser.add_argument("--variant", default=ClinicalAttentionConfig.variant)
    parser.add_argument("--batch-size", type=int, default=ClinicalAttentionConfig.batch_size)
    parser.add_argument("--num-workers", type=int, default=ClinicalAttentionConfig.num_workers)
    parser.add_argument("--epochs", type=int, default=ClinicalAttentionConfig.epochs)
    parser.add_argument("--learning-rate", type=float, default=ClinicalAttentionConfig.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=ClinicalAttentionConfig.weight_decay)
    parser.add_argument("--patience", type=int, default=ClinicalAttentionConfig.patience)
    parser.add_argument("--image-size", type=int, default=ClinicalAttentionConfig.image_size)
    parser.add_argument("--detector-imgsz", type=int, default=ClinicalAttentionConfig.detector_imgsz)
    parser.add_argument("--detector-conf", type=float, default=ClinicalAttentionConfig.detector_conf)
    parser.add_argument("--detector-iou", type=float, default=ClinicalAttentionConfig.detector_iou)
    parser.add_argument("--attention-roi-padding", type=float, default=ClinicalAttentionConfig.attention_roi_padding)
    parser.add_argument("--clinical-context-weight", type=float, default=ClinicalAttentionConfig.clinical_context_weight)
    parser.add_argument("--attention-loss-weight", type=float, default=ClinicalAttentionConfig.attention_loss_weight)
    parser.add_argument("--attention-warmup-epochs", type=int, default=ClinicalAttentionConfig.attention_warmup_epochs)
    parser.add_argument(
        "--artifact-suppression-prob",
        type=float,
        default=ClinicalAttentionConfig.artifact_suppression_prob,
    )
    parser.add_argument(
        "--artifact-residual-max",
        type=float,
        default=ClinicalAttentionConfig.artifact_residual_max,
        help="Maximum residual outside-field intensity multiplier when suppression is applied.",
    )
    parser.add_argument(
        "--suppress-artifacts-at-eval",
        action="store_true",
        help="Zero pixels outside the image-derived field in every split, without consulting the class label or detector ROI.",
    )
    parser.add_argument(
        "--canonical-field-min-occupancy",
        type=float,
        default=ClinicalAttentionConfig.canonical_field_min_occupancy,
        help=(
            "If >0, retain only pixels inside a fixed train-derived ultrasound mask present in at least this "
            "fraction of training images; the identical silhouette is used for every split."
        ),
    )
    parser.add_argument(
        "--field-bbox-normalization",
        action="store_true",
        help=(
            "Normalize the complete detected ultrasound field bounding box to the input canvas. "
            "This is an acquisition-field normalization, not a detector/lesion crop."
        ),
    )
    parser.add_argument(
        "--zoom-min",
        type=float,
        default=ClinicalAttentionConfig.zoom_min,
        help="Minimum synchronized random affine scale used during training.",
    )
    parser.add_argument(
        "--zoom-max",
        type=float,
        default=ClinicalAttentionConfig.zoom_max,
        help="Maximum synchronized random affine scale used during training.",
    )
    parser.add_argument(
        "--metadata-key",
        choices=["patient_code", "patient_visit_id"],
        default=ClinicalAttentionConfig.metadata_key,
    )
    parser.add_argument("--seed", type=int, default=None, help="random seed (or set SEED)")
    parser.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help="Patient-split seed, separated from the model/augmentation seed (default: same as --seed).",
    )
    parser.add_argument(
        "--attention-mask-mode",
        choices=["correct", "shifted", "random", "permuted"],
        default=ClinicalAttentionConfig.attention_mask_mode,
        help="Negative-control modes alter only the high-weight detector ROI; correct is the scientific model.",
    )
    parser.add_argument(
        "--detector-boxes-cache",
        default=ClinicalAttentionConfig.detector_boxes_cache,
        help="Optional JSON cache of correct detector boxes shared across repeated trainings.",
    )
    parser.add_argument("--train-ratio", type=float, default=ClinicalAttentionConfig.train_ratio)
    parser.add_argument("--val-ratio", type=float, default=ClinicalAttentionConfig.val_ratio)
    parser.add_argument("--test-ratio", type=float, default=ClinicalAttentionConfig.test_ratio)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-detector-attention", action="store_true")
    parser.add_argument("--gradcam-each-epoch", action="store_true")
    parser.add_argument("--no-final-gradcam", action="store_true")
    # RunOptions may supply a preset without changing standalone defaults.
    if defaults:
        known_destinations = {action.dest for action in parser._actions}
        unknown = set(defaults) - known_destinations
        if unknown:
            raise ValueError(f"Unknown parser defaults: {sorted(unknown)}")
        parser.set_defaults(**defaults)
    args = parser.parse_args(argv)

    cfg = ClinicalAttentionConfig(
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
        attention_roi_padding=float(args.attention_roi_padding),
        clinical_context_weight=float(args.clinical_context_weight),
        attention_loss_weight=float(args.attention_loss_weight),
        attention_warmup_epochs=int(args.attention_warmup_epochs),
        artifact_suppression_prob=float(args.artifact_suppression_prob),
        artifact_residual_max=float(args.artifact_residual_max),
        suppress_artifacts_at_eval=bool(args.suppress_artifacts_at_eval),
        canonical_field_min_occupancy=float(args.canonical_field_min_occupancy),
        field_bbox_normalization=bool(args.field_bbox_normalization),
        zoom_min=float(args.zoom_min),
        zoom_max=float(args.zoom_max),
        metadata_key=str(args.metadata_key),
        seed=resolve_seed(args.seed),
        split_seed=resolve_seed(args.split_seed if args.split_seed is not None else args.seed),
        attention_mask_mode=str(args.attention_mask_mode),
        detector_boxes_cache=str(args.detector_boxes_cache),
        pretrained=not bool(args.no_pretrained),
        use_detector_attention=not bool(args.no_detector_attention),
        save_gradcam_each_epoch=bool(args.gradcam_each_epoch),
        save_final_gradcam=not bool(args.no_final_gradcam),
        train_ratio=float(args.train_ratio),
        val_ratio=float(args.val_ratio),
        test_ratio=float(args.test_ratio),
    )
    validate_config(cfg)
    return cfg


def validate_config(cfg: ClinicalAttentionConfig) -> None:
    if cfg.epochs < 1:
        raise ValueError("--epochs must be at least 1")
    if cfg.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if cfg.image_size < 32:
        raise ValueError("--image-size must be at least 32")
    if not 0.0 <= cfg.clinical_context_weight <= 1.0:
        raise ValueError("--clinical-context-weight must be between 0 and 1")
    if cfg.attention_loss_weight < 0.0:
        raise ValueError("--attention-loss-weight cannot be negative")
    if cfg.attention_warmup_epochs < 0:
        raise ValueError("--attention-warmup-epochs cannot be negative")
    if not 0.0 <= cfg.artifact_suppression_prob <= 1.0:
        raise ValueError("--artifact-suppression-prob must be between 0 and 1")
    if not 0.0 <= cfg.artifact_residual_max <= 1.0:
        raise ValueError("--artifact-residual-max must be between 0 and 1")
    if not 0.0 <= cfg.canonical_field_min_occupancy <= 1.0:
        raise ValueError("--canonical-field-min-occupancy must be between 0 and 1")
    if cfg.zoom_min <= 0.0 or cfg.zoom_max <= 0.0:
        raise ValueError("--zoom-min and --zoom-max must be positive")
    if cfg.zoom_min > cfg.zoom_max:
        raise ValueError("--zoom-min cannot exceed --zoom-max")


def make_output_dirs(root: Path) -> Dict[str, Path]:
    run_root = root / f"run_{time.strftime('%Y%m%d-%H%M%S')}"
    paths = {
        "run_root": run_root,
        "checkpoints": run_root / "checkpoints",
        "gradcam_val": run_root / "gradcam_val_full_image",
        "gradcam_test": run_root / "gradcam_test_full_image",
        "reports": run_root / "reports",
        "mask_previews": run_root / "clinical_mask_previews",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _central_field_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.float32)
    x_margin = int(round(width * 0.08))
    y_top = int(round(height * 0.10))
    y_bottom = int(round(height * 0.04))
    mask[y_top:max(y_top + 1, height - y_bottom), x_margin:max(x_margin + 1, width - x_margin)] = 1.0
    return mask


def clinical_foreground_mask(image_rgb: np.ndarray) -> np.ndarray:
    """Estimate the ultrasound field while rejecting detached text and borders."""
    height, width = image_rgb.shape[:2]
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = (blurred > 8).astype(np.uint8) * 255

    kernel_size = max(3, int(round(min(height, width) * 0.02)))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    connected = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours_result = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours_result[-2]
    if not contours:
        return _central_field_mask(height, width)

    largest = max(contours, key=cv2.contourArea)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.drawContours(mask, [largest], contourIdx=-1, color=1, thickness=-1)

    dilation_size = max(3, int(round(min(height, width) * 0.01)))
    if dilation_size % 2 == 0:
        dilation_size += 1
    dilation_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (dilation_size, dilation_size),
    )
    mask = cv2.dilate(mask, dilation_kernel, iterations=1)
    coverage = float(mask.mean())
    if coverage < 0.03:
        return _central_field_mask(height, width)
    return mask.astype(np.float32)


def build_canonical_field_mask(
    train_df: pd.DataFrame,
    *,
    image_size: int,
    minimum_occupancy: float,
    field_bbox_normalization: bool = False,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Build one label-independent field silhouette from training images only."""
    occupancy = np.zeros((image_size, image_size), dtype=np.float64)
    for filepath in train_df["filepath"].astype(str):
        with Image.open(filepath) as source:
            image_rgb = np.asarray(source.convert("RGB"))
        field = clinical_foreground_mask(image_rgb)
        if field_bbox_normalization:
            coordinates = np.column_stack(np.where(field > 0.5))
            if len(coordinates):
                y1, x1 = coordinates.min(axis=0)
                y2, x2 = coordinates.max(axis=0) + 1
                field = field[int(y1):int(y2), int(x1):int(x2)]
        resized = cv2.resize(
            (field > 0.5).astype(np.uint8),
            (image_size, image_size),
            interpolation=cv2.INTER_NEAREST,
        )
        occupancy += resized
    occupancy /= max(len(train_df), 1)
    canonical = (occupancy >= float(minimum_occupancy)).astype(np.uint8)
    count, labels, statistics, _centroids = cv2.connectedComponentsWithStats(canonical, connectivity=8)
    if count > 1:
        largest = 1 + int(np.argmax(statistics[1:, cv2.CC_STAT_AREA]))
        canonical = (labels == largest).astype(np.uint8)
    if int(canonical.sum()) == 0:
        raise ValueError(
            f"canonical field mask is empty at minimum occupancy={minimum_occupancy}"
        )
    return canonical, {
        "training_images": int(len(train_df)),
        "minimum_occupancy": float(minimum_occupancy),
        "field_bbox_normalization": bool(field_bbox_normalization),
        "canonical_area_percent": float(100.0 * canonical.mean()),
        "mean_pixel_occupancy_inside": float(occupancy[canonical > 0].mean()),
        "minimum_pixel_occupancy_inside": float(occupancy[canonical > 0].min()),
    }


def padded_box_bounds(
    image_rgb: np.ndarray,
    box: Tuple[int, int, int, int],
    padding: float,
) -> Tuple[int, int, int, int]:
    height, width = image_rgb.shape[:2]
    x1, y1, x2, y2 = (int(value) for value in box)
    box_width = max(1, x2 - x1)
    box_height = max(1, y2 - y1)
    pad_x = int(round(box_width * float(padding)))
    pad_y = int(round(box_height * float(padding)))
    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(width, x2 + pad_x),
        min(height, y2 + pad_y),
    )


def build_clinical_attention_mask(
    image_rgb: np.ndarray,
    *,
    label_value: int,
    detector_box: Optional[Tuple[int, int, int, int]],
    roi_padding: float,
    context_weight: float,
) -> np.ndarray:
    """Build a soft mask: lesion ROI=1, ultrasound context=weight, artifacts=0."""
    foreground = clinical_foreground_mask(image_rgb)
    if label_value != 1 or detector_box is None:
        return foreground

    mask = foreground * float(context_weight)
    x1, y1, x2, y2 = padded_box_bounds(image_rgb, detector_box, roi_padding)
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = 1.0
    return np.clip(mask, 0.0, 1.0).astype(np.float32)


def _image_hw(filepath: str) -> Tuple[int, int]:
    with Image.open(filepath) as source:
        width, height = source.size
    return int(height), int(width)


def _box_iou(left: Tuple[int, int, int, int], right: Tuple[int, int, int, int]) -> float:
    lx1, ly1, lx2, ly2 = left
    rx1, ry1, rx2, ry2 = right
    intersection = max(0, min(lx2, rx2) - max(lx1, rx1)) * max(0, min(ly2, ry2) - max(ly1, ry1))
    union = max(1, (lx2 - lx1) * (ly2 - ly1) + (rx2 - rx1) * (ry2 - ry1) - intersection)
    return float(intersection / union)


def _box_field_coverage(box: Tuple[int, int, int, int], field: np.ndarray) -> float:
    x1, y1, x2, y2 = box
    area = max(1, (x2 - x1) * (y2 - y1))
    return float((field[y1:y2, x1:x2] > 0).sum() / area)


def _box_at_center(
    center_x: float,
    center_y: float,
    box_width: int,
    box_height: int,
    image_width: int,
    image_height: int,
) -> Tuple[int, int, int, int]:
    x1 = int(round(center_x - box_width / 2.0))
    y1 = int(round(center_y - box_height / 2.0))
    x1 = min(max(0, x1), max(0, image_width - box_width))
    y1 = min(max(0, y1), max(0, image_height - box_height))
    return x1, y1, min(image_width, x1 + box_width), min(image_height, y1 + box_height)


def make_attention_box_control(
    correct_boxes: Dict[str, Tuple[int, int, int, int]],
    *,
    mode: str,
    seed: int,
) -> Tuple[Dict[str, Tuple[int, int, int, int]], pd.DataFrame]:
    """Create wrong-mask negative controls while preserving ROI size.

    Only the detector-derived high-weight ROI is changed.  The full ultrasound
    context and the classifier input are unchanged.
    """
    paths = sorted(correct_boxes)
    if mode == "correct" or not paths:
        rows = [
            {
                "filepath": path,
                "mode": mode,
                "source_filepath": path,
                "correct_box": list(correct_boxes[path]),
                "used_box": list(correct_boxes[path]),
                "iou_with_correct": 1.0,
            }
            for path in paths
        ]
        return dict(correct_boxes), pd.DataFrame(rows)

    rng = np.random.default_rng(seed)
    shapes = {path: _image_hw(path) for path in paths}
    fields: Dict[str, np.ndarray] = {}
    for path in paths:
        with Image.open(path) as source:
            fields[path] = clinical_foreground_mask(np.asarray(source.convert("RGB")))
    result: Dict[str, Tuple[int, int, int, int]] = {}
    rows: List[Dict[str, object]] = []
    donors = list(paths)
    if mode == "permuted":
        rng.shuffle(donors)
        if len(donors) > 1:
            # Rotate until there are no fixed points.  A deterministic fallback
            # shift guarantees a derangement if random attempts do not.
            for _ in range(20):
                if all(target != donor for target, donor in zip(paths, donors)):
                    break
                rng.shuffle(donors)
            if any(target == donor for target, donor in zip(paths, donors)):
                donors = paths[1:] + paths[:1]

    for index, path in enumerate(paths):
        correct = tuple(int(value) for value in correct_boxes[path])
        image_height, image_width = shapes[path]
        box_width = max(1, min(image_width, correct[2] - correct[0]))
        box_height = max(1, min(image_height, correct[3] - correct[1]))
        source_path = path
        if mode == "permuted":
            source_path = donors[index]
            donor = tuple(int(value) for value in correct_boxes[source_path])
            donor_height, donor_width = shapes[source_path]
            normalized = (
                donor[0] / max(donor_width, 1),
                donor[1] / max(donor_height, 1),
                donor[2] / max(donor_width, 1),
                donor[3] / max(donor_height, 1),
            )
            used = (
                int(round(normalized[0] * image_width)),
                int(round(normalized[1] * image_height)),
                int(round(normalized[2] * image_width)),
                int(round(normalized[3] * image_height)),
            )
            used = (
                min(max(0, used[0]), image_width - 1),
                min(max(0, used[1]), image_height - 1),
                min(max(1, used[2]), image_width),
                min(max(1, used[3]), image_height),
            )
            if used[2] <= used[0] or used[3] <= used[1]:
                used = correct
        else:
            if mode == "shifted":
                centers = [
                    (0.20 * image_width, 0.25 * image_height),
                    (0.80 * image_width, 0.25 * image_height),
                    (0.20 * image_width, 0.70 * image_height),
                    (0.80 * image_width, 0.70 * image_height),
                    (0.50 * image_width, 0.78 * image_height),
                ]
            elif mode == "random":
                centers = [
                    (float(rng.uniform(0, image_width)), float(rng.uniform(0, image_height)))
                    for _ in range(64)
                ]
            else:
                raise ValueError(f"Unsupported attention mask mode: {mode}")
            candidates = [
                _box_at_center(cx, cy, box_width, box_height, image_width, image_height)
                for cx, cy in centers
            ]
            original_center = ((correct[0] + correct[2]) / 2.0, (correct[1] + correct[3]) / 2.0)
            used = max(
                candidates,
                key=lambda candidate: (
                    _box_field_coverage(candidate, fields[path]) - (2.0 * _box_iou(correct, candidate)),
                    math.hypot(
                        (candidate[0] + candidate[2]) / 2.0 - original_center[0],
                        (candidate[1] + candidate[3]) / 2.0 - original_center[1],
                    ),
                ),
            )
        result[path] = tuple(int(value) for value in used)
        rows.append(
            {
                "filepath": path,
                "mode": mode,
                "source_filepath": source_path,
                "correct_box": list(correct),
                "used_box": list(result[path]),
                "iou_with_correct": _box_iou(correct, result[path]),
                "used_box_field_coverage": _box_field_coverage(result[path], fields[path]),
            }
        )
    return result, pd.DataFrame(rows)


class FullImageClinicalAttentionDataset(Dataset):
    """Return full images and synchronized soft clinical masks."""

    use_crop = False

    def __init__(
        self,
        dataframe: pd.DataFrame,
        *,
        crop_boxes: Dict[str, Tuple[int, int, int, int]],
        metadata_key: str,
        image_size: int,
        training: bool,
        roi_padding: float,
        context_weight: float,
        artifact_suppression_prob: float,
        artifact_residual_max: float,
        suppress_artifacts_at_eval: bool,
        canonical_field_mask: Optional[np.ndarray],
        field_bbox_normalization: bool,
        zoom_min: float,
        zoom_max: float,
    ) -> None:
        self.dataframe = dataframe.reset_index(drop=True).copy()
        self.crop_boxes = crop_boxes
        self.metadata_key = str(metadata_key)
        self.image_size = int(image_size)
        self.training = bool(training)
        self.roi_padding = float(roi_padding)
        self.context_weight = float(context_weight)
        self.artifact_suppression_prob = float(artifact_suppression_prob)
        self.artifact_residual_max = float(artifact_residual_max)
        self.suppress_artifacts_at_eval = bool(suppress_artifacts_at_eval)
        self.field_bbox_normalization = bool(field_bbox_normalization)
        self.canonical_field_mask = None
        if canonical_field_mask is not None:
            fixed = (np.asarray(canonical_field_mask) > 0).astype(np.float32)
            if fixed.shape != (self.image_size, self.image_size):
                raise ValueError(
                    f"canonical field mask shape {fixed.shape} != {(self.image_size, self.image_size)}"
                )
            self.canonical_field_mask = torch.from_numpy(fixed)[None]
        self.zoom_min = float(zoom_min)
        self.zoom_max = float(zoom_max)
        self.color_jitter = ColorJitter(brightness=0.10, contrast=0.16)

    def __len__(self) -> int:
        return len(self.dataframe)

    def load_raw(self, idx: int) -> Tuple[np.ndarray, np.ndarray, int, str, str]:
        row = self.dataframe.iloc[idx]
        filepath = str(row["filepath"])
        label_value = int(row["label"])
        with Image.open(filepath) as source:
            image_rgb = np.asarray(source.convert("RGB"))
        detector_box = self.crop_boxes.get(filepath) if label_value == 1 else None
        clinical_mask = build_clinical_attention_mask(
            image_rgb,
            label_value=label_value,
            detector_box=detector_box,
            roi_padding=self.roi_padding,
            context_weight=self.context_weight,
        )
        if self.field_bbox_normalization or self.suppress_artifacts_at_eval:
            field = clinical_foreground_mask(image_rgb) > 0.5
        if self.suppress_artifacts_at_eval:
            # Deployment preprocessing must not depend on the true label or
            # its lesion-focused supervision mask. Apply it in every split.
            image_rgb = image_rgb.copy()
            image_rgb[~field] = 0
        if self.field_bbox_normalization:
            coordinates = np.column_stack(np.where(field))
            if len(coordinates):
                y1, x1 = coordinates.min(axis=0)
                y2, x2 = coordinates.max(axis=0) + 1
                if y2 > y1 and x2 > x1:
                    image_rgb = image_rgb[int(y1):int(y2), int(x1):int(x2)]
                    clinical_mask = clinical_mask[int(y1):int(y2), int(x1):int(x2)]
        metadata = str(row[self.metadata_key])
        return image_rgb, clinical_mask, label_value, metadata, filepath

    def __getitem__(self, idx: int):
        image_rgb, clinical_mask, label_value, metadata, filepath = self.load_raw(idx)

        if self.training:
            # Draw both values for every factorial cell so changing suppression
            # does not shift the RNG stream used by geometric augmentation.
            apply_suppression = random.random() < self.artifact_suppression_prob
            dim_factor = random.uniform(0.0, self.artifact_residual_max)
        else:
            apply_suppression = self.suppress_artifacts_at_eval
            dim_factor = 0.0
        if apply_suppression and not self.suppress_artifacts_at_eval:
            image_rgb = image_rgb.copy()
            outside = clinical_mask <= 0.01
            image_rgb[outside] = np.clip(
                image_rgb[outside].astype(np.float32) * dim_factor,
                0,
                255,
            ).astype(np.uint8)

        image = Image.fromarray(image_rgb, mode="RGB")
        mask = Image.fromarray((clinical_mask * 255.0).astype(np.uint8), mode="L")

        if self.training:
            if random.random() < 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            angle = random.uniform(-7.0, 7.0)
            zoom_scale = random.uniform(self.zoom_min, self.zoom_max)
            image = TF.affine(
                image,
                angle=angle,
                translate=[0, 0],
                scale=zoom_scale,
                shear=[0.0, 0.0],
                interpolation=InterpolationMode.BILINEAR,
                fill=0,
            )
            mask = TF.affine(
                mask,
                angle=angle,
                translate=[0, 0],
                scale=zoom_scale,
                shear=[0.0, 0.0],
                interpolation=InterpolationMode.NEAREST,
                fill=0,
            )
            image = self.color_jitter(image)

        image = TF.resize(
            image,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        mask = TF.resize(
            mask,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.NEAREST,
        )
        image_tensor = TF.to_tensor(image)
        if self.canonical_field_mask is not None:
            image_tensor = image_tensor * self.canonical_field_mask
        image_tensor = TF.normalize(
            image_tensor,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        mask_tensor = TF.to_tensor(mask).clamp(0.0, 1.0)
        if self.canonical_field_mask is not None:
            mask_tensor = mask_tensor * self.canonical_field_mask
        label = torch.tensor(label_value, dtype=torch.long)
        return image_tensor, label, metadata, filepath, mask_tensor


class GradcamDatasetView(Dataset):
    """Expose the attention dataset as the four-tuple expected by Grad-CAM export."""

    use_crop = False
    crop_padding = 0.0
    healthy_crop_mode = "full_image"

    def __init__(self, base: FullImageClinicalAttentionDataset) -> None:
        self.base = base
        self.crop_boxes: Dict[str, Tuple[int, int, int, int]] = {}

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        image, label, metadata, filepath, _mask = self.base[index]
        return image, label, metadata, filepath


def train_one_attention_epoch(
    model: EfficientNet_CE,
    capture: FeatureMapCapture,
    loader: DataLoader,
    *,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    attention_weight: float,
) -> Dict[str, float]:
    model.train()
    totals = {"loss": 0.0, "ce_loss": 0.0, "attention_loss": 0.0, "inside_ratio": 0.0}
    y_true: List[int] = []
    y_pred: List[int] = []
    steps = 0

    for images, labels, _metadata, _filepaths, masks in tqdm(loader, total=len(loader), desc="Train full image"):
        images = images.to(device, non_blocking=True).float()
        labels = labels.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True).float()
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        if capture.features is None:
            raise RuntimeError("EfficientNet feature hook did not capture activations")
        ce_loss = criterion(logits.float(), labels)
        attention_loss, inside_ratio = clinical_attention_loss(model, capture.features, labels, masks)
        loss = ce_loss + float(attention_weight) * attention_loss
        loss.backward()
        optimizer.step()

        preds = torch.argmax(logits, dim=1)
        totals["loss"] += float(loss.detach().cpu().item())
        totals["ce_loss"] += float(ce_loss.detach().cpu().item())
        totals["attention_loss"] += float(attention_loss.detach().cpu().item())
        totals["inside_ratio"] += float(inside_ratio.detach().cpu().item())
        steps += 1
        y_true.extend(labels.detach().cpu().numpy().astype(int).tolist())
        y_pred.extend(preds.detach().cpu().numpy().astype(int).tolist())

    return {
        "loss": totals["loss"] / max(steps, 1),
        "ce_loss": totals["ce_loss"] / max(steps, 1),
        "attention_loss": totals["attention_loss"] / max(steps, 1),
        "attention_inside_ratio": totals["inside_ratio"] / max(steps, 1),
        "acc": float(100.0 * accuracy_score(y_true, y_pred)) if y_true else 0.0,
        "mf1": float(100.0 * f1_score(y_true, y_pred, average="macro", zero_division=0)) if y_true else 0.0,
    }


def evaluate_attention_classifier(
    model: EfficientNet_CE,
    capture: FeatureMapCapture,
    loader: DataLoader,
    *,
    criterion: nn.Module,
    device: torch.device,
    attention_weight: float,
) -> Dict[str, object]:
    model.eval()
    totals = {"loss": 0.0, "ce_loss": 0.0, "attention_loss": 0.0, "inside_ratio": 0.0}
    y_true: List[int] = []
    y_pred: List[int] = []
    ids: List[str] = []
    steps = 0

    with torch.no_grad():
        for images, labels, metadata, _filepaths, masks in tqdm(loader, total=len(loader), desc="Evaluate full image"):
            images = images.to(device, non_blocking=True).float()
            labels = labels.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True).float()
            logits = model(images)
            if capture.features is None:
                raise RuntimeError("EfficientNet feature hook did not capture activations")
            ce_loss = criterion(logits.float(), labels)
            attention_loss, inside_ratio = clinical_attention_loss(model, capture.features, labels, masks)
            loss = ce_loss + float(attention_weight) * attention_loss
            preds = torch.argmax(logits, dim=1)

            totals["loss"] += float(loss.detach().cpu().item())
            totals["ce_loss"] += float(ce_loss.detach().cpu().item())
            totals["attention_loss"] += float(attention_loss.detach().cpu().item())
            totals["inside_ratio"] += float(inside_ratio.detach().cpu().item())
            steps += 1
            y_true.extend(labels.detach().cpu().numpy().astype(int).tolist())
            y_pred.extend(preds.detach().cpu().numpy().astype(int).tolist())
            ids.extend([str(item) for item in metadata])

    patient = majority_vote(y_true, y_pred, ids)
    return {
        "loss": totals["loss"] / max(steps, 1),
        "ce_loss": totals["ce_loss"] / max(steps, 1),
        "attention_loss": totals["attention_loss"] / max(steps, 1),
        "attention_inside_ratio": totals["inside_ratio"] / max(steps, 1),
        "image": {
            "acc": float(100.0 * accuracy_score(y_true, y_pred)) if y_true else 0.0,
            "mf1": float(100.0 * f1_score(y_true, y_pred, average="macro", zero_division=0)) if y_true else 0.0,
            "y_true": np.asarray(y_true, dtype=int),
            "y_pred": np.asarray(y_pred, dtype=int),
        },
        "patient_majority": patient,
    }


def build_attention_loader(
    dataframe: pd.DataFrame,
    *,
    crop_boxes: Dict[str, Tuple[int, int, int, int]],
    cfg: ClinicalAttentionConfig,
    training: bool,
    canonical_field_mask: Optional[np.ndarray] = None,
) -> DataLoader:
    dataset = FullImageClinicalAttentionDataset(
        dataframe,
        crop_boxes=crop_boxes,
        metadata_key=cfg.metadata_key,
        image_size=cfg.image_size,
        training=training,
        roi_padding=cfg.attention_roi_padding,
        context_weight=cfg.clinical_context_weight,
        artifact_suppression_prob=cfg.artifact_suppression_prob if training else 0.0,
        artifact_residual_max=cfg.artifact_residual_max,
        suppress_artifacts_at_eval=cfg.suppress_artifacts_at_eval,
        canonical_field_mask=canonical_field_mask,
        field_bbox_normalization=cfg.field_bbox_normalization,
        zoom_min=cfg.zoom_min if training else 1.0,
        zoom_max=cfg.zoom_max if training else 1.0,
    )
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=training,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
    )


def build_full_image_gradcam_loader(
    dataframe: pd.DataFrame,
    *,
    cfg: ClinicalAttentionConfig,
    crop_boxes: Dict[str, Tuple[int, int, int, int]],
    canonical_field_mask: Optional[np.ndarray] = None,
) -> DataLoader:
    base = FullImageClinicalAttentionDataset(
        dataframe,
        crop_boxes=crop_boxes,
        metadata_key=cfg.metadata_key,
        image_size=cfg.image_size,
        training=False,
        roi_padding=cfg.attention_roi_padding,
        context_weight=cfg.clinical_context_weight,
        artifact_suppression_prob=0.0,
        artifact_residual_max=cfg.artifact_residual_max,
        suppress_artifacts_at_eval=cfg.suppress_artifacts_at_eval,
        canonical_field_mask=canonical_field_mask,
        field_bbox_normalization=cfg.field_bbox_normalization,
        zoom_min=1.0,
        zoom_max=1.0,
    )
    dataset = GradcamDatasetView(base)
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
    )


def save_mask_previews(
    dataset: FullImageClinicalAttentionDataset,
    root: Path,
    *,
    max_items: int = 24,
) -> None:
    if len(dataset) == 0:
        return
    root.mkdir(parents=True, exist_ok=True)
    indices = np.linspace(0, len(dataset) - 1, num=min(max_items, len(dataset)), dtype=int)
    for sample_number, idx in enumerate(np.unique(indices), start=1):
        image_rgb, mask, label, metadata, filepath = dataset.load_raw(int(idx))
        base_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        heatmap = cv2.applyColorMap((mask * 255.0).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
        overlay = cv2.addWeighted(base_bgr, 0.60, heatmap, 0.40, 0)
        name = (
            f"{sample_number:03d}_label-{label}_id-{safe_name(metadata)}_"
            f"{safe_name(Path(filepath).stem)}.png"
        )
        cv2.imwrite(str(root / name), overlay)


def _load_state_dict(path: Path, device: torch.device) -> Dict[str, torch.Tensor]:
    payload = torch.load(str(path), map_location=device)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        payload = payload["model_state_dict"]
    if not isinstance(payload, dict):
        raise TypeError(f"unsupported checkpoint payload type: {type(payload).__name__}")
    return payload


def main(cfg: Optional[ClinicalAttentionConfig] = None) -> int:
    """Train from a supplied config, or retain the original command-line entry."""
    cfg = parse_args() if cfg is None else cfg
    validate_config(cfg)
    seed_everything(cfg.seed)
    device = get_device()
    output_dirs = make_output_dirs(Path(cfg.output_root))
    (output_dirs["reports"] / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    print("Full-image clinical-attention training configuration:")
    print(json.dumps(asdict(cfg), indent=2))
    print(f"device={device}")
    print(f"input_crop={'ultrasound_field_bbox' if cfg.field_bbox_normalization else 'disabled'}; lesion_crop=disabled")

    base_cfg = RunConfig(
        data_root=cfg.data_root,
        detector_checkpoint=cfg.detector_checkpoint,
        output_root=cfg.output_root,
        checkpoint_in=cfg.checkpoint_in,
        variant=cfg.variant,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        patience=cfg.patience,
        image_size=cfg.image_size,
        detector_imgsz=cfg.detector_imgsz,
        detector_conf=cfg.detector_conf,
        detector_iou=cfg.detector_iou,
        metadata_key=cfg.metadata_key,
        seed=cfg.split_seed,
        pretrained=cfg.pretrained,
        save_gradcam_each_epoch=False,
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
        test_ratio=cfg.test_ratio,
    )
    train_df, val_df, test_df = build_dataframes(base_cfg)

    crop_boxes: Dict[str, Tuple[int, int, int, int]] = {}
    if cfg.use_detector_attention:
        all_df = pd.concat([train_df, val_df, test_df], axis=0).drop_duplicates(subset=["filepath"])
        cache_path = Path(cfg.detector_boxes_cache) if cfg.detector_boxes_cache.strip() else None
        if cache_path is not None and cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            crop_boxes = {
                str(path): tuple(int(value) for value in box)
                for path, box in cached.items()
            }
            print(f"Loaded detector boxes cache: {cache_path}")
        else:
            detector = YOLO(cfg.detector_checkpoint)
            detector_device: str | int = 0 if device.type == "cuda" else "cpu"
            crop_boxes = precompute_detector_boxes(
                detector,
                all_df.reset_index(drop=True),
                imgsz=cfg.detector_imgsz,
                conf=cfg.detector_conf,
                iou=cfg.detector_iou,
                device=detector_device,
            )
            del detector
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(
                    json.dumps({path: list(box) for path, box in crop_boxes.items()}, indent=2) + "\n",
                    encoding="utf-8",
                )
                print(f"Saved detector boxes cache: {cache_path}")
        crop_boxes, box_assignments = make_attention_box_control(
            crop_boxes,
            mode=cfg.attention_mask_mode,
            seed=cfg.split_seed,
        )
        box_assignments.to_csv(output_dirs["reports"] / "attention_box_assignments.csv", index=False)
        print(
            f"attention_mask_mode={cfg.attention_mask_mode} | "
            f"mean_box_iou_with_correct={box_assignments['iou_with_correct'].mean():.6f}"
        )
    else:
        print("Detector attention disabled; using ultrasound-field masks only.")

    split_stats = dump_split_stats(train_df, val_df, test_df, crop_boxes)
    (output_dirs["reports"] / "attention_mask_coverage.json").write_text(
        json.dumps(split_stats, indent=2)
    )
    print("Attention box coverage:")
    print(json.dumps(split_stats, indent=2))

    canonical_field_mask: Optional[np.ndarray] = None
    if cfg.canonical_field_min_occupancy > 0.0:
        canonical_field_mask, canonical_stats = build_canonical_field_mask(
            train_df,
            image_size=cfg.image_size,
            minimum_occupancy=cfg.canonical_field_min_occupancy,
            field_bbox_normalization=cfg.field_bbox_normalization,
        )
        cv2.imwrite(
            str(output_dirs["reports"] / "canonical_field_mask.png"),
            canonical_field_mask * 255,
        )
        (output_dirs["reports"] / "canonical_field_mask_summary.json").write_text(
            json.dumps(canonical_stats, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Canonical field mask: {json.dumps(canonical_stats)}")

    train_loader = build_attention_loader(
        train_df,
        crop_boxes=crop_boxes,
        cfg=cfg,
        training=True,
        canonical_field_mask=canonical_field_mask,
    )
    val_loader = build_attention_loader(
        val_df,
        crop_boxes=crop_boxes,
        cfg=cfg,
        training=False,
        canonical_field_mask=canonical_field_mask,
    )
    test_loader = build_attention_loader(
        test_df,
        crop_boxes=crop_boxes,
        cfg=cfg,
        training=False,
        canonical_field_mask=canonical_field_mask,
    )
    save_mask_previews(train_loader.dataset, output_dirs["mask_previews"] / "train")
    save_mask_previews(val_loader.dataset, output_dirs["mask_previews"] / "val")

    # Dataset construction deliberately uses split_seed.  The data utility
    # resets global RNGs to that value, so restore the independent training
    # seed immediately before initialization and DataLoader iteration.
    seed_everything(cfg.seed)
    model = EfficientNet_CE(
        variant=cfg.variant,
        num_classes=2,
        dropout=0.2,
        pretrained=cfg.pretrained,
    ).to(device)
    if cfg.checkpoint_in.strip():
        checkpoint_path = resolve_checkpoint_path(
            cfg.checkpoint_in,
            data_root=cfg.data_root,
            fallback_search_root=cfg.output_root,
        )
        model.load_state_dict(_load_state_dict(checkpoint_path, device), strict=True)
        print(f"Loaded initial checkpoint: {checkpoint_path}")

    criterion = nn.CrossEntropyLoss(weight=compute_class_weights(train_df, device))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    capture = FeatureMapCapture(model)
    history: List[Dict[str, object]] = []
    best_score = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    best_path = output_dirs["checkpoints"] / "best_effnet_ce_full_image_clinical_attention.h"
    val_gradcam_loader = None

    try:
        for epoch in range(1, cfg.epochs + 1):
            warmup = 1.0 if cfg.attention_warmup_epochs == 0 else min(
                1.0,
                epoch / float(cfg.attention_warmup_epochs),
            )
            effective_attention_weight = cfg.attention_loss_weight * warmup
            print(f"\n{'=' * 80}")
            print(f"Epoch {epoch}/{cfg.epochs}")
            print(f"attention_weight={effective_attention_weight:.6f}")
            print(f"{'=' * 80}")

            train_metrics = train_one_attention_epoch(
                model,
                capture,
                train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                attention_weight=effective_attention_weight,
            )
            val_metrics = evaluate_attention_classifier(
                model,
                capture,
                val_loader,
                criterion=criterion,
                device=device,
                attention_weight=effective_attention_weight,
            )

            gradcam_files: List[str] = []
            if cfg.save_gradcam_each_epoch:
                if val_gradcam_loader is None:
                    val_gradcam_loader = build_full_image_gradcam_loader(
                        val_df,
                        cfg=cfg,
                        crop_boxes=crop_boxes,
                        canonical_field_mask=canonical_field_mask,
                    )
                records = save_val_gradcams(
                    model,
                    val_gradcam_loader,
                    device=device,
                    epoch=epoch,
                    root=output_dirs["gradcam_val"],
                )
                gradcam_files = [str(item["path"]) for item in records]

            record = {
                "epoch": int(epoch),
                "attention_weight": float(effective_attention_weight),
                "train_loss": float(train_metrics["loss"]),
                "train_ce_loss": float(train_metrics["ce_loss"]),
                "train_attention_loss": float(train_metrics["attention_loss"]),
                "train_attention_inside_ratio": float(train_metrics["attention_inside_ratio"]),
                "train_acc": float(train_metrics["acc"]),
                "train_mf1": float(train_metrics["mf1"]),
                "val_loss": float(val_metrics["loss"]),
                "val_ce_loss": float(val_metrics["ce_loss"]),
                "val_attention_loss": float(val_metrics["attention_loss"]),
                "val_attention_inside_ratio": float(val_metrics["attention_inside_ratio"]),
                "val_image_acc": float(val_metrics["image"]["acc"]),
                "val_image_mf1": float(val_metrics["image"]["mf1"]),
                "val_patient_acc": float(val_metrics["patient_majority"]["acc"]),
                "val_patient_mf1": float(val_metrics["patient_majority"]["mf1"]),
                "gradcam_files": gradcam_files,
            }
            history.append(record)
            print("EPOCH_METRICS | " + ", ".join(
                f"{key}={value:.6f}" for key, value in record.items()
                if isinstance(value, float)
            ))

            current_score = float(val_metrics["patient_majority"]["mf1"])
            if current_score > best_score:
                best_score = current_score
                best_epoch = epoch
                epochs_without_improvement = 0
                save_checkpoint_atomic(model.state_dict(), best_path, torch.save)
                print(f"Saved best checkpoint: {best_path}")
            else:
                epochs_without_improvement += 1

            (output_dirs["reports"] / "history.json").write_text(json.dumps(history, indent=2))
            pd.DataFrame(history).drop(columns=["gradcam_files"], errors="ignore").to_csv(
                output_dirs["reports"] / "history.csv",
                index=False,
            )
            if epochs_without_improvement >= cfg.patience:
                print(f"Early stopping at epoch {epoch}; best epoch={best_epoch}")
                break

        model.load_state_dict(_load_state_dict(best_path, device), strict=True)
        final_val_metrics = evaluate_attention_classifier(
            model,
            capture,
            val_loader,
            criterion=criterion,
            device=device,
            attention_weight=cfg.attention_loss_weight,
        )
        final_test_metrics = evaluate_attention_classifier(
            model,
            capture,
            test_loader,
            criterion=criterion,
            device=device,
            attention_weight=cfg.attention_loss_weight,
        )
    finally:
        capture.close()

    save_final_reports(output_dirs["reports"], split_name="val", metrics=final_val_metrics)
    save_final_reports(output_dirs["reports"], split_name="test", metrics=final_test_metrics)

    gradcam_counts = {"val": 0, "test": 0}
    if cfg.save_final_gradcam:
        final_val_gradcam_loader = build_full_image_gradcam_loader(
            val_df,
            cfg=cfg,
            crop_boxes=crop_boxes,
            canonical_field_mask=canonical_field_mask,
        )
        final_test_gradcam_loader = build_full_image_gradcam_loader(
            test_df,
            cfg=cfg,
            crop_boxes=crop_boxes,
            canonical_field_mask=canonical_field_mask,
        )
        val_records = save_val_gradcams(
            model,
            final_val_gradcam_loader,
            device=device,
            epoch=best_epoch,
            root=output_dirs["gradcam_val"],
        )
        test_records = save_val_gradcams(
            model,
            final_test_gradcam_loader,
            device=device,
            epoch=best_epoch,
            root=output_dirs["gradcam_test"],
        )
        gradcam_counts = {"val": len(val_records), "test": len(test_records)}

    summary = {
        "run_root": str(output_dirs["run_root"]),
        "input_mode": cfg.input_mode,
        "input_crop": bool(cfg.field_bbox_normalization),
        "input_crop_kind": "ultrasound_field_bbox" if cfg.field_bbox_normalization else "none",
        "lesion_crop": False,
        "best_epoch": int(best_epoch),
        "best_checkpoint": str(best_path),
        "best_val_patient_mf1_percent": float(best_score),
        "final_val_attention_inside_ratio": float(final_val_metrics["attention_inside_ratio"]),
        "final_test_attention_inside_ratio": float(final_test_metrics["attention_inside_ratio"]),
        "final_test_image_acc_percent": float(final_test_metrics["image"]["acc"]),
        "final_test_image_mf1_percent": float(final_test_metrics["image"]["mf1"]),
        "final_test_patient_acc_percent": float(final_test_metrics["patient_majority"]["acc"]),
        "final_test_patient_mf1_percent": float(final_test_metrics["patient_majority"]["mf1"]),
        "gradcam_counts": gradcam_counts,
        "mask_previews": str(output_dirs["mask_previews"]),
    }
    (output_dirs["reports"] / "final_summary.json").write_text(json.dumps(summary, indent=2))
    print("\nFULL_IMAGE_CLINICAL_ATTENTION_SUMMARY")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
