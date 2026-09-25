"""Clinical CAM-loss: a penalty on class-activation mass that falls outside a soft focus mask.

    loss = weighted cross-entropy + lambda(epoch) * mean over the batch of [1 - sum(A * M) / max(sum(A), eps)]

A is the softplus of the class-activation map (CAM) of the true class, computed from the last backbone feature
map and the classifier weights. M is the focus mask: for a pathological image 1.0 inside the detector box padded
by 30 %, 0.35 on the rest of the estimated ultrasound field and 0 outside it; for a normal image the estimated
field at 1.0. lambda ramps linearly over the first epochs. The mask enters the loss only, never the network input.
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

ROI_PADDING = 0.30
CONTEXT_WEIGHT = 0.35
ATTENTION_WEIGHT = 0.25
WARMUP_EPOCHS = 5
EPS = 1e-8

Box = tuple[int, int, int, int]


def _odd(size: int) -> int:
    return size if size % 2 else size + 1


def _central_field(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.float32)
    x_margin = int(round(width * 0.08))
    y_top = int(round(height * 0.10))
    y_bottom = int(round(height * 0.04))
    mask[y_top:max(y_top + 1, height - y_bottom), x_margin:max(x_margin + 1, width - x_margin)] = 1.0
    return mask


def estimate_field(image_rgb: np.ndarray) -> np.ndarray:
    """Binary float mask of the ultrasound field: the largest bright connected region, so detached text is excluded.

    Falls back to a fixed central rectangle when no region covering at least 3 % of the image is found.
    """
    height, width = image_rgb.shape[:2]
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    binary = (cv2.GaussianBlur(gray, (5, 5), 0) > 8).astype(np.uint8) * 255

    close_size = _odd(max(3, int(round(min(height, width) * 0.02))))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
    connected = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    contours = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    if not contours:
        return _central_field(height, width)

    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.drawContours(mask, [max(contours, key=cv2.contourArea)], contourIdx=-1, color=1, thickness=-1)
    dilate_size = _odd(max(3, int(round(min(height, width) * 0.01))))
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_size, dilate_size))
    mask = cv2.dilate(mask, dilate_kernel, iterations=1)
    if float(mask.mean()) < 0.03:
        return _central_field(height, width)
    return mask.astype(np.float32)


def padded_box(box: Box, height: int, width: int, padding: float = ROI_PADDING) -> Box:
    """Box (x1, y1, x2, y2) enlarged on each side by a fraction of its width and height, clipped to the image."""
    x1, y1, x2, y2 = (int(value) for value in box)
    pad_x = int(round(max(1, x2 - x1) * float(padding)))
    pad_y = int(round(max(1, y2 - y1) * float(padding)))
    return max(0, x1 - pad_x), max(0, y1 - pad_y), min(width, x2 + pad_x), min(height, y2 + pad_y)


def focus_mask(image_rgb: np.ndarray, pathological: bool, box: Box | None = None,
               padding: float = ROI_PADDING, context_weight: float = CONTEXT_WEIGHT) -> np.ndarray:
    """Soft focus mask in [0, 1] at image resolution.

    Normal images, and pathological images without a detection, receive the estimated field at weight 1.0.
    """
    field = estimate_field(image_rgb)
    if not pathological or box is None:
        return field
    mask = field * float(context_weight)
    x1, y1, x2, y2 = padded_box(box, *image_rgb.shape[:2], padding=padding)
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = 1.0
    return np.clip(mask, 0.0, 1.0).astype(np.float32)


class FeatureCapture:
    """Forward hook that keeps the last backbone feature map, with its autograd graph, for the CAM."""

    def __init__(self, model: nn.Module):
        self.features: torch.Tensor | None = None
        self._handle = model.backbone.features[-1].register_forward_hook(self._hook)

    def _hook(self, _module, _inputs, output: torch.Tensor) -> None:
        self.features = output

    def close(self) -> None:
        self._handle.remove()

    def __enter__(self) -> "FeatureCapture":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


def class_activation_map(model: nn.Module, feature_map: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """CAM of the given classes, shape [B, h, w]: feature channels weighted by the final linear layer."""
    classifier = model.backbone.classifier[-1]
    if not isinstance(classifier, nn.Linear):
        raise TypeError("the classifier must end with nn.Linear")
    if feature_map.ndim != 4 or feature_map.shape[1] != classifier.in_features:
        raise ValueError(f"feature map of shape {tuple(feature_map.shape)} does not match the classifier")
    weights = classifier.weight.index_select(0, labels)
    return (feature_map * weights[:, :, None, None]).sum(dim=1)


def cam_focus_loss(model: nn.Module, feature_map: torch.Tensor, labels: torch.Tensor,
                   masks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (penalty averaged over the batch, mean share of activation mass inside the mask).

    masks have shape [B, 1, H, W] and are resized bilinearly to the feature-map grid.
    """
    attention = F.softplus(class_activation_map(model, feature_map, labels))
    resized = F.interpolate(masks.float(), size=attention.shape[-2:], mode="bilinear",
                            align_corners=False)[:, 0].clamp(0.0, 1.0)
    total = attention.flatten(1).sum(dim=1).clamp_min(EPS)
    inside = (attention * resized).flatten(1).sum(dim=1)
    share = (inside / total).clamp(0.0, 1.0)
    return (1.0 - share).mean(), share.mean()


def attention_weight(epoch: int, final: float = ATTENTION_WEIGHT, warmup_epochs: int = WARMUP_EPOCHS) -> float:
    """lambda for a 1-based epoch: final * min(1, epoch / warmup_epochs)."""
    if epoch < 1:
        raise ValueError("epochs start at 1")
    ramp = 1.0 if warmup_epochs == 0 else min(1.0, epoch / float(warmup_epochs))
    return float(final) * ramp
