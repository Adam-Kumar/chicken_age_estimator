"""Model package for chicken age estimation.

This package contains the core deep learning components:
- model.py: Neural network architectures (Baseline, Late Fusion, Feature Fusion)
- dataset.py: PyTorch Dataset classes and data transformations

All models support multiple backbones:
- CNNs: ResNet-18/50/101, EfficientNet-B0, ConvNeXt-Tiny/Base
- Transformers: ViT-B/16, Swin-Tiny/Base

Fusion strategies:
- Baseline: View-agnostic (single model for both views)
- Late Fusion: View-aware ensemble (separate models, averaged predictions)
- Feature Fusion: Learned fusion (concatenate features before prediction)
"""

from .model import (
    ResNetRegressor,
    LateFusionRegressor,
    FeatureFusionRegressor,
    count_parameters,
)

from .dataset import (
    ChickenAgeDataset,
    ChickenAgePairedDataset,
    get_default_transforms,
)

__all__ = [
    # Models
    'ResNetRegressor',
    'LateFusionRegressor',
    'FeatureFusionRegressor',
    'count_parameters',
    # Datasets
    'ChickenAgeDataset',
    'ChickenAgePairedDataset',
    'get_default_transforms',
]
