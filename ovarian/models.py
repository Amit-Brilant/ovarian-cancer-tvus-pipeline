"""Network definitions and weight loading.

Attribute names and layer order match the released checkpoints, which load with strict=True.
"""
from pathlib import Path

import torch
from torch import nn
from torchvision import models as tvm

# Tabular input width of each Stage III configuration that fuses an image with tabular features.
FUSION_TABULAR_DIM = {3: 15, 4: 1, 5: 16, 6: 11}


def _efficientnet(variant: str, pretrained: bool) -> nn.Module:
    index = int(variant.lstrip("b"))
    weights = getattr(tvm, f"EfficientNet_B{index}_Weights").DEFAULT if pretrained else None
    return getattr(tvm, f"efficientnet_b{index}")(weights=weights)


class EfficientNetClassifier(nn.Module):
    """EfficientNet backbone with a two-logit head (Stage I, Stage III Configuration 2)."""

    def __init__(self, variant: str = "b7", num_classes: int = 2, dropout: float = 0.2, pretrained: bool = True):
        super().__init__()
        self.backbone = _efficientnet(variant, pretrained)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(nn.Dropout(p=dropout, inplace=True), nn.Linear(in_features, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x.float())


class EfficientNetFusion(nn.Module):
    """EfficientNet image embedding concatenated with a tabular embedding (Stage III Configurations 3 to 6).

    Tabular branch: Linear, LayerNorm, ReLU, dropout, twice (128 and 64 units). Head: Linear to 256 units,
    ReLU, dropout, Linear to the logits.
    """

    def __init__(self, tabular_dim: int, variant: str = "b7", num_classes: int = 2, dropout: float = 0.2,
                 pretrained: bool = True, tab_hidden: tuple = (128, 64), fusion_hidden: int = 256):
        super().__init__()
        self.tabular_dim = tabular_dim
        self.backbone = _efficientnet(variant, pretrained)
        image_dim = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        layers, previous = [], tabular_dim
        for hidden in tab_hidden:
            layers += [nn.Linear(previous, hidden), nn.LayerNorm(hidden), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            previous = hidden
        self.tab_mlp = nn.Sequential(*layers)
        self.fusion_head = nn.Sequential(nn.Linear(image_dim + previous, fusion_hidden), nn.ReLU(inplace=True),
                                         nn.Dropout(dropout), nn.Linear(fusion_hidden, num_classes))

    def forward(self, x_img: torch.Tensor, x_tab: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([self.backbone(x_img.float()), self.tab_mlp(x_tab.float())], dim=1)
        return self.fusion_head(fused)


def load_weights(model: nn.Module, path: Path) -> nn.Module:
    """Load a released checkpoint (a plain state dict, optionally wrapped or saved from DataParallel)."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    for key in ("model_state_dict", "state_dict"):
        if isinstance(payload, dict) and key in payload:
            payload = payload[key]
    state = {k.removeprefix("module."): v for k, v in payload.items()}
    model.load_state_dict(state, strict=True)
    return model.eval()


def stage3_model(configuration: int, pretrained: bool = True) -> nn.Module:
    if configuration == 2:
        return EfficientNetClassifier(pretrained=pretrained)
    return EfficientNetFusion(FUSION_TABULAR_DIM[configuration], pretrained=pretrained)
