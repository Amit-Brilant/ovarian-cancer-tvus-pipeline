"""EfficientNet classifier used by the Stage I trainer.

Only ``EfficientNet_CE`` is used by the published Stage I code; the original
file also contained unrelated model definitions, which were removed for release.
The class body is unchanged, so the released checkpoints load with strict=True.
"""
import torch
import torch.nn as nn
from torchvision.models import (
    efficientnet_b0, efficientnet_b1, efficientnet_b2,
    efficientnet_b3, efficientnet_b4, efficientnet_b5,
    efficientnet_b6, efficientnet_b7,
    EfficientNet_B0_Weights, EfficientNet_B1_Weights,
    EfficientNet_B2_Weights, EfficientNet_B3_Weights,
    EfficientNet_B4_Weights, EfficientNet_B5_Weights,
    EfficientNet_B6_Weights, EfficientNet_B7_Weights,
)


class EfficientNet_CE(nn.Module):
    """EfficientNet backbone with a plain cross-entropy head (two logits)."""
    def __init__(self, variant="b7", num_classes=2, dropout=0.2, pretrained=True, in_channels=3):
        super(EfficientNet_CE, self).__init__()

        self.variant = variant
        self.num_classes = num_classes
        self.dropout = dropout
        self.pretrained = pretrained
        self.in_channels = in_channels

        if variant == "b0":
            weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
            self.backbone = efficientnet_b0(weights=weights)
        elif variant == "b1":
            weights = EfficientNet_B1_Weights.DEFAULT if pretrained else None
            self.backbone = efficientnet_b1(weights=weights)
        elif variant == "b2":
            weights = EfficientNet_B2_Weights.DEFAULT if pretrained else None
            self.backbone = efficientnet_b2(weights=weights)
        elif variant == "b3":
            weights = EfficientNet_B3_Weights.DEFAULT if pretrained else None
            self.backbone = efficientnet_b3(weights=weights)
        elif variant == "b4":
            weights = EfficientNet_B4_Weights.DEFAULT if pretrained else None
            self.backbone = efficientnet_b4(weights=weights)
        elif variant == "b5":
            weights = EfficientNet_B5_Weights.DEFAULT if pretrained else None
            self.backbone = efficientnet_b5(weights=weights)
        elif variant == "b6":
            weights = EfficientNet_B6_Weights.DEFAULT if pretrained else None
            self.backbone = efficientnet_b6(weights=weights)
        elif variant == "b7":
            weights = EfficientNet_B7_Weights.DEFAULT if pretrained else None
            self.backbone = efficientnet_b7(weights=weights)
        else:
            raise ValueError("variant must be one of: b0,b1,b2,b3,b4,b5,b6,b7")

        if in_channels != 3:
            first_conv = self.backbone.features[0][0]
            new_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=first_conv.out_channels,
                kernel_size=first_conv.kernel_size,
                stride=first_conv.stride,
                padding=first_conv.padding,
                dilation=first_conv.dilation,
                groups=first_conv.groups,
                bias=(first_conv.bias is not None),
                padding_mode=first_conv.padding_mode,
            )
            with torch.no_grad():
                if pretrained and first_conv.weight.shape[1] == 3 and in_channels > 3:
                    new_conv.weight[:, :3, :, :] = first_conv.weight
                    nn.init.kaiming_normal_(new_conv.weight[:, 3:, :, :], mode="fan_out", nonlinearity="relu")
                elif pretrained and first_conv.weight.shape[1] == 3 and in_channels == 1:
                    new_conv.weight[:, 0:1, :, :] = first_conv.weight.mean(dim=1, keepdim=True)
                else:
                    nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
                if new_conv.bias is not None:
                    nn.init.zeros_(new_conv.bias)
            self.backbone.features[0][0] = new_conv

        # replace the classifier with a plain logits head
        in_feats = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_feats, num_classes)
        )

    def forward(self, x):
        x = x.float()
        out = self.backbone(x)  # logits [B, num_classes]
        return out
