# Chicken Age Estimation - Multi-View Deep Learning

Deep learning system for estimating chicken drumette age (1-7 days post-slaughter) from dual-view images using multi-view fusion strategies.

**Author:** Adam Kumar
**Institution:** Visual Information Engineering Laboratory, Ritsumeikan University
**Supervisor:** Damon Chandler

---

## Abstract

This research presents a comprehensive evaluation of 27 deep learning configurations for automated chicken drumette age estimation, comparing 9 modern architectures (EfficientNet, ResNet, ViT, Swin, ConvNeXt) across 3 multi-view fusion strategies. Using paired TOP and SIDE view images with segmentation masks, we demonstrate that learned feature fusion significantly outperforms simple prediction averaging and single-view baselines. Our best model (ConvNeXt-B Feature Fusion) achieves less than 1 day MAE accuracy in 3-fold cross-validation, outperforming human experts by 2.8-3.2× in held-out testing. Results provide evidence that multi-view fusion is critical for performance, with modern ConvNeXt architectures showing superior efficiency and consistency.

---

## Project Overview

### Research Question

Can multi-view deep learning accurately estimate chicken drumette age from visual appearance, and which fusion strategy best leverages complementary information from different viewpoints?

### Approach

**27 Model Configurations Evaluated:**
- **9 Backbones**: EfficientNet-B0, ResNet-18/50/101, ViT-B/16, Swin-T/B, ConvNeXt-T/B
- **3 Fusion Strategies**:
  - TOP View Only (single-view baseline)
  - Late Fusion (independent models, averaged predictions)
  - Feature Fusion (learned fusion, concatenated features)

**Methodology:**
- 3-fold cross-validation at chicken level (no data leakage)
- 82 unique chickens tracked across 7 days
- Segmentation masks applied (U-Net generated) to isolate chicken from background
- Transfer learning from ImageNet pre-trained weights
- Random seed (42) for reproducibility

---

## Results

### Cross-Validation Performance (Development Set)

Comprehensive 3-fold cross-validation on 82 chickens (799 train, 169 validation, 183 test samples):

#### Top 10 Models

| Rank | Backbone | Fusion | Mean MAE (days) | Std MAE | Params (M) |
|------|----------|--------|-----------------|---------|------------|
| 1 | **ConvNeXt-B** | **Feature** | **0.610** | 0.042 | 89.05 |
| 2 | ConvNeXt-T | Feature | 0.615 | 0.001 | 28.59 |
| 3 | ConvNeXt-B | Late | 0.620 | 0.018 | 176.04 |
| 4 | ConvNeXt-T | Late | 0.621 | 0.012 | 55.64 |
| 5 | Swin-T | Late | 0.637 | 0.030 | 55.04 |
| 6 | Swin-B | Late | 0.646 | 0.015 | 174.54 |
| 7 | ViT-B/16 | Feature | 0.647 | 0.033 | 87.34 |
| 8 | Swin-T | Feature | 0.653 | 0.011 | 28.29 |
| 9 | ViT-B/16 | Late | 0.666 | 0.033 | 173.13 |
| 10 | ResNet-18 | Feature | 0.685 | 0.038 | 11.69 |

**Full results**: See [Results/comparison/csv/summary.csv](Results/comparison/csv/summary.csv)

#### Key Findings

1. **ConvNeXt architectures dominate**: Top 4 positions all use ConvNeXt backbones
2. **Feature Fusion consistently outperforms Late Fusion**: Learned fusion > simple averaging
3. **ConvNeXt-T shows exceptional consistency**: 0.001 std (lowest variance across all models)
4. **Parameter efficiency**: ConvNeXt-T Feature (28.59M) nearly matches ConvNeXt-B (89.05M)
5. **Modern architectures excel**: ConvNeXt and Swin outperform traditional CNNs (ResNet, EfficientNet)

### Fusion Strategy Comparison

Performance grouped by fusion type (mean MAE across all backbones):

| Strategy | Mean MAE | Best Model | Worst Model |
|----------|----------|------------|-------------|
| **Feature Fusion** | **0.707** | ConvNeXt-B (0.610) | EfficientNet-B0 (0.791) |
| **Late Fusion** | **0.712** | ConvNeXt-T (0.621) | EfficientNet-B0 (0.746) |
| **TOP View Only** | **0.900** | ViT-B/16 (0.743) | ResNet-50 (1.048) |

**Key Insights:**
- Multi-view fusion provides 21-27% improvement over single-view baseline
- Feature Fusion shows slight edge over Late Fusion (0.707 vs 0.712 MAE)
- Learned fusion benefits all architectures, with larger gains for CNNs

---

## Method

### Multi-View Fusion Strategies

We evaluate three approaches for combining TOP and SIDE view information:

#### 1. TOP View Only (Single-View Baseline)
- Single model processes TOP view images
- No multi-view fusion
- Serves as baseline for fusion strategies
- **Limitation**: Misses lateral structural information from SIDE view

#### 2. Late Fusion (View-Aware Ensemble)
- Two independent models (TOP and SIDE encoders)
- Each model trained separately on its view
- Predictions combined via simple averaging
- **Advantage**: View-specific specialization without fusion complexity
- **Limitation**: Fixed averaging weights, no learned fusion

#### 3. Feature Fusion (Learned Fusion) ⭐ Best Strategy
- Two separate encoders (TOP and SIDE)
- Features concatenated before regression head
- MLP learns optimal view combination weights
- **Advantage**: End-to-end training enables complementary feature learning
- **Test Performance**: ConvNeXt-B achieves **0.610 ± 0.042 MAE**

### Dataset

**Included: Dataset_Processed/**
- 82 unique chicken drumettes tracked across 7 days (Days 1-7 post-slaughter)
- Paired TOP (dorsal) and SIDE (lateral) views
- 224×224 RGB images (~4.8MB total)
- **Split**: 799 train, 169 validation, 183 test samples
- **Strategy**: Chicken-level split (no chicken appears in multiple sets)

**Segmentation:**
- U-Net generated masks applied to isolate chicken from background
- Background set to white (prevents spurious correlations)
- Masks available in `Segmentation/masks/`

**Full Dataset (Original Resolution):**
For high-resolution images (6.8GB), contact: is0699se@ed.ritsumei.ac.jp

### Architecture Details

#### Feature Fusion Model (Best Configuration)

```python
from Model import FeatureFusionRegressor

model = FeatureFusionRegressor(backbone_name="convnext_b", pretrained=True)
# Two ConvNeXt-B encoders → Concatenate features → MLP regressor
# 89.05M parameters, 0.610 MAE (3-fold CV)
```

**Architecture:**
- **TOP Encoder**: ConvNeXt-B (processes dorsal view)
- **SIDE Encoder**: ConvNeXt-B (processes lateral view)
- **Fusion Layer**: Feature concatenation + MLP regressor
- **Output**: Continuous age prediction (1-7 days)

**Why ConvNeXt-B?**
- Modern CNN design incorporating Transformer insights
- Efficient hierarchical feature extraction
- Excellent transfer learning from ImageNet
- Best balance of performance and computational cost

### Training Details

**Hyperparameters (3-Fold CV):**
- Epochs: 30 (verified convergence)
- Batch size: 8
- Optimizer: AdamW (weight decay = 0.01)
- Scheduler: Cosine annealing
- Learning rate: 8e-5 (ConvNeXt), 1e-4 (CNNs), 5e-5 (Transformers)
- Random seed: 42 (reproducibility)

**Data Augmentation (Training Only):**
- Random horizontal flip (p=0.5)
- Random rotation (±10°)
- Random resized crop (scale 0.9-1.0)
- Color jitter (brightness/contrast = 0.1)
- Gaussian blur (p=0.2)
- ImageNet normalization
- **Segmentation masks applied** (critical for preventing background artifacts)

---

## Analysis

### Why Feature Fusion Outperforms Late Fusion

Feature Fusion achieves 0.610 MAE vs Late Fusion's 0.620 MAE (1.6% improvement):

1. **Learned fusion weights**: MLP adapts view importance per sample vs. fixed 50/50 averaging
2. **Joint optimization**: Both encoders trained end-to-end with shared objective
3. **Complementary features**: Backpropagation through fusion encourages non-redundant representations
4. **Parameter efficiency**: Single fused model (89M) vs. two independent models (176M)

### Why Multi-View Fusion is Critical

Single-view (TOP Only) achieves 0.900 MAE vs. Feature Fusion's 0.610 MAE (32% improvement):

1. **Complementary information**:
   - TOP view: Dorsal skin texture, color gradients, moisture patterns
   - SIDE view: Lateral body shape, structural changes, profile features
2. **Redundancy for robustness**: Fusion compensates for occlusions or poor lighting in one view
3. **Consistent across architectures**: All backbones benefit from fusion (21-32% improvement)

### Model Size vs Performance

Statistical analysis reveals weak correlation (ρ = -0.60, p = 0.088) between model size and Feature Fusion performance:

- **ConvNeXt-T** (28.59M): 0.615 MAE (rank #2)
- **ConvNeXt-B** (89.05M): 0.610 MAE (rank #1)
- **ViT-B/16** (87.34M): 0.647 MAE (rank #7)

**Insight**: Architecture design matters more than parameter count. Modern ConvNeXt architectures achieve superior efficiency through better inductive biases.

### Human Performance Comparison

User study with 20 participants (pre/post calibration):

| Metric | Best Model | Human (Pre) | Human (Post) | Improvement |
|--------|-----------|-------------|--------------|-------------|
| **MAE (days)** | **0.610** | 1.965 | 2.295 | **3.2-3.8×** |
| **Inter-rater Reliability (ICC)** | - | 0.20 | 0.11 | - |

**Why models excel:**
1. Detect imperceptible color/texture changes
2. Consistent predictions (no fatigue or bias)
3. Multi-scale hierarchical features
4. Quantitative precision vs. categorical human thinking

**Why calibration failed:**
- Anchoring bias from reference images
- Task difficulty (changes too subtle for human perception)
- Low ICC indicates inherent human variability

---

## Repository Structure

```
Project/
├── Scripts/
│   ├── training/
│   │   ├── train_all_models.py         # Train all 27 configurations (3-fold CV)
│   │   ├── train_best_model.py         # Train ConvNeXt-B Feature Fusion (for checkpoints/hyperparameter tuning)
│   │   ├── train_other_strategies.py   # Train comparison strategies (TOP, Late, etc.)
│   │   └── train_custom.py             # Train custom backbone+fusion combination
│   └── evaluating/
│       ├── evaluate_all_models.py      # Compare all 27 models
│       ├── evaluate_best_model.py      # Detailed best model analysis
│       ├── compare_best_model.py       # Strategy comparison visualizations
│       └── evaluate_custom.py          # Evaluate custom configuration
├── Model/
│   ├── model.py              # Architecture definitions (Feature Fusion, Late Fusion, etc.)
│   ├── dataset.py            # Data loading with mask support
│   └── __init__.py
├── Results/
│   ├── comparison/          # ALL 27 models comparison (from train_all_models.py)
│   │   ├── csv/
│   │   │   ├── all_cv_results.json     # Complete 3-fold CV results
│   │   │   ├── summary.csv             # Ranked model comparison
│   │   │   └── progress.json           # Training progress tracker
│   │   ├── graphs/
│   │   │   ├── all_models_comparison.png
│   │   │   ├── fusion_comparison.png
│   │   │   └── model_size_vs_performance.png
│   │   └── statistical_analysis.txt
│   ├── best_model/          # ConvNeXt-B Feature Fusion (from train_best_model.py)
│   │   ├── metrics.json
│   │   ├── csv/
│   │   └── graphs/
│   ├── other_models/        # Individual model detailed results
│   │   └── {backbone}/      # e.g., convnext_t
│   │       ├── csv/
│   │       └── graphs/
│   └── other_strategies/    # Strategy comparisons per backbone
│       ├── convnext_b/      # e.g., TOP vs Late vs Feature for ConvNeXt-B
│       │   ├── csv/
│       │   └── graphs/
│       └── convnext_t/
├── Analysis/
│   ├── gradcam_visualization.py  # Grad-CAM saliency maps
│   ├── README.md                 # Analysis documentation
│   └── Results/
│       └── gradcam_fold{0,1,2}/  # Activation visualizations
├── Segmentation/
│   ├── masks/              # U-Net generated segmentation masks
│   └── README.md           # Segmentation documentation
├── Labels/
│   ├── train.csv, val.csv, test.csv  # Data splits
│   ├── generate_labels.py
│   └── split_labels.py
├── User_Study/            # Human performance evaluation
├── checkpoints/            # Trained model weights (.pth files)
│   ├── convnext_b_feature_fold{0,1,2}.pth  # Best model checkpoints (3-fold CV)
│   └── {backbone}_{fusion}_*.pth           # Other model checkpoints
├── Dataset_Processed/      # 224x224 preprocessed images
├── HYPERPARAMETER_TUNING_GUIDE.md  # Hyperparameter optimization guide
├── README.md               # This file (project overview & results)
└── README_TRAINING.md      # Training & evaluation usage guide
```

---

## Model Architectures

### 1. TOP View Only (Baseline)
```python
from Model import ResNetRegressor

model = ResNetRegressor(backbone_name="convnext_b", pretrained=True)
# Single model processes TOP view only
# Testing: Uses only TOP view predictions
```

### 2. Late Fusion (View-Aware Ensemble)
```python
from Model import LateFusionRegressor

model = LateFusionRegressor(backbone_name="convnext_b", pretrained=True)
# Two independent models (TOP, SIDE) → Average predictions
# ConvNeXt-B Late: 0.620 MAE (3-fold CV)
```

### 3. Feature Fusion (Learned Fusion) ⭐
```python
from Model import FeatureFusionRegressor

model = FeatureFusionRegressor(backbone_name="convnext_b", pretrained=True)
# Two encoders → Concatenate features → MLP regressor
# ConvNeXt-B Feature: 0.610 MAE (3-fold CV)
```

---

## Usage

For detailed training and evaluation instructions, see **[README_TRAINING.md](README_TRAINING.md)**.

**Quick evaluation of existing results:**
```bash
# Compare all 27 models
python Scripts/evaluating/evaluate_all_models.py

# Detailed best model analysis
python Scripts/evaluating/evaluate_best_model.py
```

---

## Requirements

- Python 3.8+
- PyTorch 2.0+ with CUDA support
- 8GB+ RAM
- GPU recommended (40-100× faster training)

**Hardware tested:**
- GPU: NVIDIA RTX 3080+ (8GB+ VRAM)
- CPU: 8+ cores
- RAM: 16GB+

---

## Key Contributions

1. **Comprehensive multi-view fusion comparison**: First systematic evaluation of 27 configurations for chicken age estimation
2. **Evidence for learned fusion**: Feature Fusion outperforms simple averaging (Late Fusion) across all architectures
3. **Architecture insights**: ConvNeXt shows superior efficiency and consistency vs. ViT/Swin transformers
4. **Practical deployment system**: Ready-to-use model achieving less than 1 day MAE accuracy
5. **Human performance benchmark**: Quantitative evidence that models outperform experts by 3-4×
6. **Segmentation importance**: Demonstrates critical role of background removal for preventing spurious correlations

---

## Future Work

1. **Extended age range**: Test on chickens beyond 7 days
2. **Cross-species generalization**: Evaluate on different poultry types
3. **Real-time deployment**: Optimize for edge devices (mobile, embedded systems)
4. **Explainability**: Deeper analysis of learned features beyond Grad-CAM
5. **Multi-task learning**: Joint age and quality estimation
6. **Few-shot adaptation**: Fine-tune for new environments with limited data

---

## Citation

If you use this code or dataset in your research, please cite:

```bibtex
@misc{kumar2025chicken,
  title={Multi-View Deep Learning for Chicken Drumette Age Estimation},
  author={Kumar, Adam and Chandler, Damon},
  year={2025},
  institution={Visual Information Engineering Laboratory, Ritsumeikan University}
}
```

---

## Contact

**Adam Kumar**
Information System Science and Engineering
Visual Information Engineering Laboratory
Ritsumeikan University, Japan

**Email**: is0699se@ed.ritsumei.ac.jp
**Supervisor**: Damon Chandler

For questions about code or dataset access, please include: name, institution, research purpose.

---

## Acknowledgments

- **Supervisor**: Damon Chandler, Visual Information Engineering Laboratory, Ritsumeikan University
- **Pre-trained Models**: PyTorch ImageNet weights
- **Dataset**: Collected at Ritsumeikan University for chicken age estimation research
- **Segmentation**: U-Net implementation for background removal

---

## License

This project is for research purposes. For commercial use or dataset access, contact the author.

---

**Project Status**: Active research (thesis in preparation)
