"""model.py
Defines PyTorch models for chicken age regression.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torchvision.models as models

__all__ = [
    "ResNetRegressor",
    "LateFusionRegressor",
    "FeatureFusionRegressor",
    "build_resnet_regressor",
]


def _get_resnet_backbone(name: str = "resnet50", pretrained: bool = True) -> Tuple[nn.Module, int]:
    model_fn = getattr(models, name)
    resnet = model_fn(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
    in_features = resnet.fc.in_features
    backbone = nn.Sequential(*list(resnet.children())[:-1])
    return backbone, in_features


class ResNetRegressor(nn.Module):
    def __init__(self, backbone_name: str = "resnet50", pretrained: bool = True, freeze_backbone: bool = False):
        super().__init__()
        self.backbone, num_feats = _get_resnet_backbone(backbone_name, pretrained)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.head = nn.Linear(num_feats, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = x.flatten(1)
        x = self.head(x)
        return x.squeeze(1)


class LateFusionRegressor(nn.Module):
    def __init__(self, backbone_name: str = "resnet50", pretrained: bool = True, freeze_backbone: bool = False):
        super().__init__()
        self.model_top = ResNetRegressor(backbone_name, pretrained, freeze_backbone)
        self.model_side = ResNetRegressor(backbone_name, pretrained, freeze_backbone)

    def forward(self, x_top: torch.Tensor, x_side: torch.Tensor) -> torch.Tensor:
        return 0.5 * (self.model_top(x_top) + self.model_side(x_side))


# -----------------------------------------------------------------------------
# Feature-level fusion model (concatenate pooled features)
# -----------------------------------------------------------------------------


class FeatureFusionRegressor(nn.Module):
    """Two ResNet backbones → concatenate features → regression head."""

    def __init__(
        self,
        backbone_name: str = "resnet50",
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.backbone_top, num_feats = _get_resnet_backbone(backbone_name, pretrained)
        self.backbone_side, _ = _get_resnet_backbone(backbone_name, pretrained)

        if freeze_backbone:
            for p in list(self.backbone_top.parameters()) + list(self.backbone_side.parameters()):
                p.requires_grad = False

        self.head = nn.Sequential(
            nn.Linear(num_feats * 2, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1),
        )

    def forward(self, x_top: torch.Tensor, x_side: torch.Tensor) -> torch.Tensor:  # noqa: D401
        feat_top = self.backbone_top(x_top).flatten(1)  # (B, C)
        feat_side = self.backbone_side(x_side).flatten(1)
        fused = torch.cat([feat_top, feat_side], dim=1)
        out = self.head(fused)
        return out.squeeze(1)


def build_resnet_regressor(**kwargs) -> ResNetRegressor:
    return ResNetRegressor(**kwargs) 