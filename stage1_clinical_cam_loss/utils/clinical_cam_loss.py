"""The exact differentiable CAM regularizer used in the Stage I experiments.

L_total = weighted_cross_entropy + lambda(epoch) * L_CAM
L_CAM = mean(1 - sum(softplus(CAM_true) * soft_mask) / sum(softplus(CAM_true)))

The mask supervises class-activation mass; it is not an input crop, a hard
attention gate, a segmentation target, or a claim of clinical validation.
"""
from typing import Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def clinical_attention_loss(
    model: nn.Module,
    feature_map: torch.Tensor,
    labels: torch.Tensor,
    clinical_masks: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (CAM penalty, mean weighted inside ratio), retaining gradients.

    Labels are true class indices; masks have shape [B, 1, H, W]. This keeps
    the historical function name for existing training and audit imports.
    """
    classifier = model.backbone.classifier[-1]
    if not isinstance(classifier, nn.Linear):
        raise TypeError("EfficientNet_CE classifier must end with nn.Linear")
    if feature_map.ndim != 4:
        raise ValueError(f"expected a 4D feature map, got shape={tuple(feature_map.shape)}")
    if feature_map.shape[1] != classifier.in_features:
        raise ValueError(
            "feature channels do not match classifier input: "
            f"{feature_map.shape[1]} != {classifier.in_features}"
        )

    class_weights = classifier.weight.index_select(0, labels)
    raw_cam = (feature_map * class_weights[:, :, None, None]).sum(dim=1)
    attention = F.softplus(raw_cam)
    resized_masks = F.interpolate(
        clinical_masks.float(),
        size=attention.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )[:, 0].clamp(0.0, 1.0)

    total_mass = attention.flatten(1).sum(dim=1).clamp_min(1e-8)
    inside_mass = (attention * resized_masks).flatten(1).sum(dim=1)
    inside_ratio = (inside_mass / total_mass).clamp(0.0, 1.0)
    loss = (1.0 - inside_ratio).mean()
    return loss, inside_ratio.mean()


class ClinicalCAMLoss(nn.Module):
    """Weighted CE + the experiment's CAM penalty, for ModelHandler runs."""

    def __init__(self, attention_weight=0.25, warmup_epochs=5, class_weights=None):
        super().__init__()
        if not math.isfinite(float(attention_weight)) or attention_weight < 0:
            raise ValueError("attention_weight must be finite and nonnegative")
        if int(warmup_epochs) != warmup_epochs or warmup_epochs < 0:
            raise ValueError("warmup_epochs must be a nonnegative integer")
        self.attention_weight = float(attention_weight)
        self.warmup_epochs = int(warmup_epochs)
        self.ce = nn.CrossEntropyLoss(weight=class_weights)

    def set_class_weights(self, weights):
        self.ce.weight = weights.detach().clone()
        return self

    def weight_for_epoch(self, epoch):
        if epoch < 1:
            raise ValueError("CAM epochs start at 1")
        warmup = 1.0 if self.warmup_epochs == 0 else min(1.0, epoch / float(self.warmup_epochs))
        return self.attention_weight * warmup

    def components(self, logits, labels, *, model, feature_map, clinical_masks, epoch,
                   attention_weight=None):
        ce = self.ce(logits.float(), labels)
        cam, inside = clinical_attention_loss(model, feature_map, labels, clinical_masks)
        weight = self.weight_for_epoch(epoch) if attention_weight is None else float(attention_weight)
        return dict(loss=ce + weight * cam, ce_loss=ce, attention_loss=cam,
                    attention_inside_ratio=inside)

    def forward(self, logits, labels, *, model, feature_map, clinical_masks, epoch):
        return self.components(logits, labels, model=model, feature_map=feature_map,
                               clinical_masks=clinical_masks, epoch=epoch)["loss"]

    def extra_repr(self):
        return f"attention_weight={self.attention_weight}, warmup_epochs={self.warmup_epochs}"
