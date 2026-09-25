"""Graph-preserving feature capture for the existing EfficientNet_CE model."""
from typing import Optional

import torch
from torch import nn


class FeatureMapCapture:
    """Capture the final feature map without changing logits or checkpoint keys.

    This is the hook used in the Stage I CAM-loss experiments. It is not a
    Transformer attention layer and adds no trainable parameters.
    """

    def __init__(self, model: nn.Module) -> None:
        self.features: Optional[torch.Tensor] = None
        self.handle = model.backbone.features[-1].register_forward_hook(self._capture)

    def _capture(self, _module, _inputs, output) -> None:
        self.features = output

    def close(self) -> None:
        self.handle.remove()
