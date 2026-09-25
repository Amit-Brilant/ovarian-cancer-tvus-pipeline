import numpy as np
import pytest
import torch
from torch import nn

from ovarian.clinical_cam_loss import (
    ATTENTION_WEIGHT,
    FeatureCapture,
    attention_weight,
    cam_focus_loss,
    class_activation_map,
    estimate_field,
    focus_mask,
    padded_box,
)
from ovarian.models import EfficientNetClassifier


class LinearHead(nn.Module):
    """Minimal stand-in exposing backbone.classifier[-1] with one feature channel and unit weights."""

    def __init__(self):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(1, 2))
        with torch.no_grad():
            self.backbone.classifier[-1].weight.fill_(1.0)


def features_for_attention(attention: np.ndarray) -> torch.Tensor:
    """Feature map whose softplus CAM equals the given attention (inverse softplus)."""
    a = torch.tensor(attention, dtype=torch.float64)
    return torch.log(torch.expm1(a))[None, None]


def region_grid():
    """10 x 10 grid: 20 cells in the padded box (1.0), 30 in the field (0.35), 50 outside (0)."""
    mask = np.zeros((10, 10))
    mask[:, :5] = 0.35
    mask[:4, :5] = 1.0
    return mask


def test_worked_example_pathological():
    mask = region_grid()
    attention = np.where(mask == 1.0, 0.80 / 20, np.where(mask == 0.35, 0.15 / 30, 0.05 / 50))
    assert attention.sum() == pytest.approx(1.0)
    labels = torch.tensor([1])
    model = LinearHead().double()
    loss, share = cam_focus_loss(model, features_for_attention(attention), labels, torch.tensor(mask)[None, None])
    assert share.item() == pytest.approx(0.8525, abs=1e-9)
    assert loss.item() == pytest.approx(0.1475, abs=1e-9)
    penalty = attention_weight(epoch=6) * loss.item()
    assert penalty == pytest.approx(0.25 * 0.1475, abs=1e-9)


def test_normal_image_field_mask():
    mask = np.zeros((10, 10))
    mask[1:9, 1:9] = 1.0
    attention = np.where(mask == 1.0, 0.90 / 64, 0.10 / 36)
    model = LinearHead().double()
    loss, share = cam_focus_loss(model, features_for_attention(attention), torch.tensor([0]),
                                 torch.tensor(mask)[None, None])
    assert share.item() == pytest.approx(0.90, abs=1e-9)
    assert loss.item() == pytest.approx(0.10, abs=1e-9)


def test_batch_mean_and_gradient():
    mask = torch.tensor(region_grid())[None, None].repeat(2, 1, 1, 1)
    features = torch.zeros(2, 1, 10, 10, dtype=torch.float64, requires_grad=True)
    loss, share = cam_focus_loss(LinearHead().double(), features, torch.tensor([1, 0]), mask)
    uniform_share = (20 * 1.0 + 30 * 0.35) / 100
    assert share.item() == pytest.approx(uniform_share)
    loss.backward()
    assert features.grad is not None and torch.isfinite(features.grad).all()
    assert features.grad[0, 0, 9, 9] > 0 > features.grad[0, 0, 0, 0]


def test_warmup_schedule():
    assert [attention_weight(e) for e in (1, 2, 5, 6, 50)] == pytest.approx(
        [0.05, 0.10, 0.25, 0.25, 0.25])
    assert attention_weight(1, warmup_epochs=0) == ATTENTION_WEIGHT
    with pytest.raises(ValueError):
        attention_weight(0)


def test_padded_box_clipping():
    assert padded_box((100, 200, 200, 300), 1000, 1000, 0.3) == (70, 170, 230, 330)
    assert padded_box((0, 0, 50, 40), 100, 100, 0.3) == (0, 0, 65, 52)
    assert padded_box((900, 900, 1000, 1000), 1000, 1000, 0.3) == (870, 870, 1000, 1000)


def synthetic_frame(size: int = 400) -> np.ndarray:
    """Dark frame with a bright fan-like field and a detached text blob in a corner."""
    image = np.zeros((size, size, 3), dtype=np.uint8)
    yy, xx = np.mgrid[:size, :size]
    image[((yy - 60) ** 2 + (xx - 200) ** 2 <= 300**2) & (yy > 60)] = 120
    image[5:20, 5:60] = 255
    return image


def test_field_rejects_detached_text():
    field = estimate_field(synthetic_frame())
    assert set(np.unique(field)) == {0.0, 1.0}
    assert field[200, 200] == 1.0
    assert field[10, 30] == 0.0


def test_focus_masks():
    image = synthetic_frame()
    box = (150, 150, 250, 250)
    pathological = focus_mask(image, pathological=True, box=box)
    assert np.unique(pathological) == pytest.approx([0.0, 0.35, 1.0])
    assert pathological[140, 140] == 1.0
    assert pathological[200, 60] == pytest.approx(0.35)
    assert pathological[10, 30] == 0.0
    normal = focus_mask(image, pathological=False, box=box)
    assert np.array_equal(normal, estimate_field(image))
    assert np.array_equal(focus_mask(image, pathological=True, box=None), normal)


def test_capture_matches_logits():
    torch.manual_seed(0)
    model = EfficientNetClassifier("b0", pretrained=False).eval()
    x = torch.randn(2, 3, 64, 64)
    with FeatureCapture(model) as capture, torch.no_grad():
        logits = model(x)
        features = capture.features
    assert features.shape[1] == model.backbone.classifier[-1].in_features
    bias = model.backbone.classifier[-1].bias
    for k in (0, 1):
        cam = class_activation_map(model, features, torch.tensor([k, k]))
        assert torch.allclose(cam.mean(dim=(1, 2)) + bias[k], logits[:, k], atol=1e-4)
    assert not model.backbone.features[-1]._forward_hooks
