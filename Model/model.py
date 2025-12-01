"""model_enhanced.py
Enhanced PyTorch models supporting multiple backbones for chicken age regression.

Supported backbones:
- CNNs: ResNet-18/50/101, EfficientNet-B0, ConvNeXt-Tiny/Base
- Transformers: ViT-B/16, Swin-Tiny/Base

Fusion strategies:
- Late Fusion: Average predictions from two views
- Feature Fusion: Concatenate features before prediction
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torchvision.models as models

__all__ = [
    "ResNetRegressor",
    "BaselineFusionRegressor",
    "FeatureFusionRegressor",
    "LateFusionRegressor",
    "get_backbone_info",
]


# Helper modules for ViT backbone
class Transpose(nn.Module):
    """Transpose tensor dimensions."""
    def __init__(self, dim1: int, dim2: int):
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.transpose(self.dim1, self.dim2)


class ExtractCLSToken(nn.Module):
    """Extract CLS token from ViT output."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, 0]  # Extract first token (CLS token)


class ViTFeatureExtractor(nn.Module):
    """
    Wrapper for ViT that extracts features before classification head.
    Properly handles CLS token and positional embeddings.
    """
    def __init__(self, vit_model):
        super().__init__()
        self.vit = vit_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape and permute the input tensor
        x = self.vit._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.vit.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        # Add positional embedding
        x = self.vit.encoder(x)

        # Extract CLS token
        x = x[:, 0]

        return x


class SwinFeatureExtractor(nn.Module):
    """
    Wrapper for Swin Transformer that extracts features before classification head.
    """
    def __init__(self, swin_model):
        super().__init__()
        self.features = swin_model.features
        self.norm = swin_model.norm
        self.permute = swin_model.permute
        self.avgpool = swin_model.avgpool
        self.flatten = swin_model.flatten

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.norm(x)
        x = self.permute(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        return x


class ConvNeXtFeatureExtractor(nn.Module):
    """
    Wrapper for ConvNeXt that extracts features before classification head.
    """
    def __init__(self, convnext_model):
        super().__init__()
        self.features = convnext_model.features
        self.avgpool = convnext_model.avgpool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        return x


def get_backbone_info(backbone_name: str) -> dict:
    """Get information about available backbones."""
    backbone_registry = {
        "resnet18": {
            "model_fn": models.resnet18,
            "weights": models.ResNet18_Weights.DEFAULT,
            "features": 512,
        },
        "resnet50": {
            "model_fn": models.resnet50,
            "weights": models.ResNet50_Weights.DEFAULT,
            "features": 2048,
        },
        "resnet101": {
            "model_fn": models.resnet101,
            "weights": models.ResNet101_Weights.DEFAULT,
            "features": 2048,
        },
        "efficientnet_b0": {
            "model_fn": models.efficientnet_b0,
            "weights": models.EfficientNet_B0_Weights.DEFAULT,
            "features": 1280,
        },
        "vit_b_16": {
            "model_fn": models.vit_b_16,
            "weights": models.ViT_B_16_Weights.DEFAULT,
            "features": 768,
        },
        "swin_t": {
            "model_fn": models.swin_t,
            "weights": models.Swin_T_Weights.DEFAULT,
            "features": 768,
        },
        "swin_b": {
            "model_fn": models.swin_b,
            "weights": models.Swin_B_Weights.DEFAULT,
            "features": 1024,
        },
        "convnext_t": {
            "model_fn": models.convnext_tiny,
            "weights": models.ConvNeXt_Tiny_Weights.DEFAULT,
            "features": 768,
        },
        "convnext_b": {
            "model_fn": models.convnext_base,
            "weights": models.ConvNeXt_Base_Weights.DEFAULT,
            "features": 1024,
        },
    }

    if backbone_name not in backbone_registry:
        raise ValueError(
            f"Unknown backbone: {backbone_name}. "
            f"Available: {list(backbone_registry.keys())}"
        )

    return backbone_registry[backbone_name]


def _get_backbone(backbone_name: str = "resnet50", pretrained: bool = True) -> Tuple[nn.Module, int]:
    """
    Build a feature extractor backbone.

    Args:
        backbone_name: One of ['resnet18', 'resnet50', 'resnet101', 'efficientnet_b0',
                               'vit_b_16', 'swin_t', 'swin_b', 'convnext_t', 'convnext_b']
        pretrained: Whether to use ImageNet pretrained weights

    Returns:
        (backbone_module, num_features)
    """
    info = get_backbone_info(backbone_name)
    model_fn = info["model_fn"]
    weights = info["weights"] if pretrained else None
    num_features = info["features"]

    if "resnet" in backbone_name:
        # ResNet models
        full_model = model_fn(weights=weights)
        # Remove the final FC layer and avgpool
        backbone = nn.Sequential(*list(full_model.children())[:-1])
    elif "efficientnet" in backbone_name:
        # EfficientNet models
        full_model = model_fn(weights=weights)
        # EfficientNet structure: features -> avgpool -> classifier
        backbone = nn.Sequential(
            full_model.features,
            full_model.avgpool,
        )
    elif "vit" in backbone_name:
        # Vision Transformer models
        full_model = model_fn(weights=weights)
        # Use custom wrapper that properly handles ViT's forward pass
        backbone = ViTFeatureExtractor(full_model)
    elif "swin" in backbone_name:
        # Swin Transformer models
        full_model = model_fn(weights=weights)
        # Use custom wrapper for Swin
        backbone = SwinFeatureExtractor(full_model)
    elif "convnext" in backbone_name:
        # ConvNeXt models
        full_model = model_fn(weights=weights)
        # Use custom wrapper for ConvNeXt
        backbone = ConvNeXtFeatureExtractor(full_model)
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")

    return backbone, num_features


class ResNetRegressor(nn.Module):
    """
    Single-view regressor supporting multiple backbones.

    Args:
        backbone_name: One of ['resnet18', 'resnet50', 'resnet101', 'efficientnet_b0',
                               'vit_b_16', 'swin_t', 'swin_b', 'convnext_t', 'convnext_b']
        pretrained: Whether to use ImageNet pretrained weights
        freeze_backbone: Whether to freeze backbone weights
    """

    def __init__(
        self,
        backbone_name: str = "resnet50",
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.backbone, num_feats = _get_backbone(backbone_name, pretrained)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.head = nn.Linear(num_feats, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = x.flatten(1)
        x = self.head(x)
        return x.squeeze(1)


class BaselineFusionRegressor(nn.Module):
    """
    Dual-view baseline regressor - view-agnostic approach.

    Training: Model is trained on ALL images (TOP + SIDE mixed together) without
    knowing which view each image is from. This is the simplest baseline.

    Testing: Both views are passed through the model independently and predictions
    are averaged (like late fusion but with a single view-agnostic model).

    This creates a clear progression:
    - Single-view (ResNetRegressor): Uses only one view (e.g., TOP only)
    - Baseline (this class): Uses both views, but model is view-agnostic (no view labels)
    - Late Fusion: Uses both views with view awareness (separate models per view)
    - Feature Fusion: Uses both views with learned fusion (concatenate features)

    Args:
        backbone_name: One of ['resnet18', 'resnet50', 'resnet101', 'efficientnet_b0',
                               'vit_b_16', 'swin_t', 'swin_b', 'convnext_t', 'convnext_b']
        pretrained: Whether to use ImageNet pretrained weights
        freeze_backbone: Whether to freeze backbone weights
    """

    def __init__(
        self,
        backbone_name: str = "resnet50",
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        # Single model trained on all views (view-agnostic)
        self.model = ResNetRegressor(backbone_name, pretrained, freeze_backbone)

    def forward(self, x_top: torch.Tensor, x_side: torch.Tensor) -> torch.Tensor:
        # Pass both views through same model (view-agnostic)
        # Average predictions (not features)
        pred_top = self.model(x_top)
        pred_side = self.model(x_side)
        return (pred_top + pred_side) / 2


class LateFusionRegressor(nn.Module):
    """
    Two-view late fusion regressor supporting multiple backbones.
    Averages predictions from TOP and SIDE views.

    Args:
        backbone_name: One of ['resnet18', 'resnet50', 'resnet101', 'efficientnet_b0',
                               'vit_b_16', 'swin_t', 'swin_b', 'convnext_t', 'convnext_b']
        pretrained: Whether to use ImageNet pretrained weights
        freeze_backbone: Whether to freeze backbone weights
    """

    def __init__(
        self,
        backbone_name: str = "resnet50",
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.model_top = ResNetRegressor(backbone_name, pretrained, freeze_backbone)
        self.model_side = ResNetRegressor(backbone_name, pretrained, freeze_backbone)

    def forward(self, x_top: torch.Tensor, x_side: torch.Tensor) -> torch.Tensor:
        return 0.5 * (self.model_top(x_top) + self.model_side(x_side))


class FeatureFusionRegressor(nn.Module):
    """
    Two-view feature fusion regressor supporting multiple backbones.
    Concatenates features from TOP and SIDE views before prediction.

    Args:
        backbone_name: One of ['resnet18', 'resnet50', 'resnet101', 'efficientnet_b0',
                               'vit_b_16', 'swin_t', 'swin_b', 'convnext_t', 'convnext_b']
        pretrained: Whether to use ImageNet pretrained weights
        freeze_backbone: Whether to freeze backbone weights
    """

    def __init__(
        self,
        backbone_name: str = "resnet50",
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.backbone_top, num_feats = _get_backbone(backbone_name, pretrained)
        self.backbone_side, _ = _get_backbone(backbone_name, pretrained)

        if freeze_backbone:
            for p in list(self.backbone_top.parameters()) + list(self.backbone_side.parameters()):
                p.requires_grad = False

        # Fusion head: concatenate features from both views
        self.head = nn.Sequential(
            nn.Linear(num_feats * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 1),
        )

    def forward(self, x_top: torch.Tensor, x_side: torch.Tensor) -> torch.Tensor:
        feat_top = self.backbone_top(x_top).flatten(1)  # (B, C)
        feat_side = self.backbone_side(x_side).flatten(1)
        fused = torch.cat([feat_top, feat_side], dim=1)
        out = self.head(fused)
        return out.squeeze(1)


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in millions."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


if __name__ == "__main__":
    # Test all backbones
    print("Testing all backbone architectures:\n")

    backbones = [
        "resnet18", "resnet50", "resnet101",
        "efficientnet_b0",
        "vit_b_16",
        "swin_t",
        "convnext_t",
    ]

    for backbone in backbones:
        print(f"\n{'='*60}")
        print(f"Backbone: {backbone.upper()}")
        print(f"{'='*60}")

        # Test single-view model
        model_single = ResNetRegressor(backbone_name=backbone)
        params_single = count_parameters(model_single)
        print(f"Single-view model: {params_single:.2f}M parameters")

        # Test baseline fusion model (dual-view, averaged features)
        model_baseline = BaselineFusionRegressor(backbone_name=backbone)
        params_baseline = count_parameters(model_baseline)
        print(f"Baseline Fusion model: {params_baseline:.2f}M parameters")

        # Test late fusion model
        model_late = LateFusionRegressor(backbone_name=backbone)
        params_late = count_parameters(model_late)
        print(f"Late Fusion model: {params_late:.2f}M parameters")

        # Test feature fusion model
        model_fusion = FeatureFusionRegressor(backbone_name=backbone)
        params_fusion = count_parameters(model_fusion)
        print(f"Feature Fusion model: {params_fusion:.2f}M parameters")

        # Test forward pass
        dummy_input = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out_single = model_single(dummy_input)
            out_baseline = model_baseline(dummy_input, dummy_input)
            out_late = model_late(dummy_input, dummy_input)
            out_fusion = model_fusion(dummy_input, dummy_input)

        print(f"Single-view output shape: {out_single.shape}")
        print(f"Baseline Fusion output shape: {out_baseline.shape}")
        print(f"Late Fusion output shape: {out_late.shape}")
        print(f"Feature Fusion output shape: {out_fusion.shape}")
