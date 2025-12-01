# Chicken Age Estimation - Multi-View Fusion

Deep learning system for estimating chicken age (1-7 days) from dual-view images using various CNN and Transformer backbones with multi-view fusion strategies.

## Project Overview

This project compares **27 model configurations** for automated chicken age estimation:
- **9 Backbones**: EfficientNet-B0, ResNet-18/50/101, ViT-B/16, Swin-T/B, ConvNeXt-T/B
- **3 Fusion Strategies**: Baseline (view-agnostic), Late Fusion (ensemble), Feature Fusion (learned)

**Best Model on Held-Out Test Set**: **ConvNeXt-T Late Fusion**
**Test Set Performance**: **0.181 days MAE** (evaluated on 13 unseen chickens)
**Model Parameters**: 55.64M

## Method

### Multi-View Fusion Strategies

We evaluate three approaches for fusing TOP and SIDE view information:

1. **Baseline (View-Agnostic)**
   - Single model processes both views without distinguishing them
   - Simplest approach, treats all images equally
   - Training: Mixed TOP+SIDE images
   - Testing: Average predictions from both views

2. **Late Fusion (View-Aware Ensemble)** ⭐ **Best on Test Set**
   - Two separate models (one for TOP, one for SIDE)
   - Each model specializes in its view
   - Predictions combined via simple averaging
   - **Test MAE: 0.181 days**

3. **Feature Fusion (Learned Fusion)**
   - Two separate encoders (one for each view)
   - Features concatenated before final prediction
   - MLP learns optimal combination of view features
   - Best in cross-validation but slightly lower generalization

### Dataset Split

- **Training**: 70% (57 chickens) - Model learning
- **Validation**: 15% (12 chickens) - Hyperparameter tuning & early stopping
- **Test**: 15% (13 chickens) - **Held-out for final evaluation only**

The held-out test set represents truly unseen chickens, providing an unbiased estimate of real-world performance.

## Results

### Held-Out Test Set Performance (Final Results)

Performance on 13 completely unseen chickens (held-out test set):

| Rank | Strategy | Test MAE | Test RMSE | Params (M) |
|------|----------|----------|-----------|------------|
| 1 | **Late Fusion** | **0.181** | 0.228 | 55.64 |
| 2 | Feature Fusion | 0.204 | 0.264 | 28.59 |
| 3 | TOP View Only | 0.205 | 0.271 | 27.82 |
| 4 | Baseline | 0.310 | 0.445 | 27.82 |
| 5 | SIDE View Only | 0.496 | 0.687 | 27.82 |

**Key Results:**
- **Late Fusion achieves best generalization** (0.181 days) despite not being best in CV
- **Feature Fusion ranks 2nd** (0.204 days), showing strong but slightly lower generalization
- **TOP View alone nearly matches Feature Fusion** (0.205 days), suggesting TOP view is highly informative
- **Multi-view fusion significantly outperforms Baseline** (0.181 vs 0.310), proving value of view-aware processing
- **SIDE View alone performs worst** (0.496 days), indicating it provides less discriminative information

### Cross-Validation Results (Development)

Performance during 3-fold cross-validation (used for model selection):

| Rank | Backbone | Fusion | Mean MAE | Std MAE | Params (M) |
|------|----------|--------|----------|---------|------------|
| 1 | ConvNeXt-T | Feature | **0.172** | 0.016 | 28.59 |
| 2 | ConvNeXt-T | Late | 0.197 | 0.006 | 55.64 |
| 3 | ConvNeXt-B | Late | 0.226 | 0.008 | 176.04 |
| 4 | ConvNeXt-B | Feature | 0.270 | 0.058 | 89.05 |
| 5 | Swin-T | Late | 0.280 | 0.004 | 55.04 |

**Development Findings:**
- **ConvNeXt-T Feature Fusion** achieved best CV performance (0.172 ± 0.016)
- **ConvNeXt architectures** dominated top positions across fusion types
- **Model size ≠ performance**: ConvNeXt-T (28.59M) outperformed larger models
- All top models used ConvNeXt or Swin architectures (modern convnets & transformers)

### Analysis: Why These Results?

#### Why Late Fusion Generalizes Best

**Late Fusion wins on test set (0.181 vs 0.204) despite Feature Fusion winning CV:**

1. **Simpler fusion is more robust**
   - Late Fusion: Simple averaging of independent predictions
   - Feature Fusion: Learned MLP fusion → risk of overfitting to training data
   - Test set includes unseen chickens with potentially different appearance distributions

2. **View specialization benefits**
   - Each Late Fusion model learns view-specific features without interference
   - TOP model focuses purely on dorsal patterns (skin texture, color)
   - SIDE model focuses on lateral features (body shape, leg positioning)
   - Independent specialization → better generalization

3. **CV-test discrepancy is normal**
   - CV: Multiple train/val splits from same 82 chickens
   - Test: Completely different 13 chickens
   - Feature Fusion's learned fusion may have fit CV-specific patterns

#### Why TOP View is So Informative

**TOP View alone (0.205) nearly matches Feature Fusion (0.204):**

1. **Richer visual information**
   - Larger visible surface area (entire dorsal side)
   - More distinctive aging markers: skin color changes, texture degradation
   - Consistent viewpoint reduces variability

2. **Age-discriminative features**
   - Skin darkening: Progressive color change over 7 days
   - Moisture loss: Surface desiccation visible from top
   - Texture changes: Smoothness → roughness progression

3. **SIDE View limitations**
   - Smaller visible area (lateral profile only)
   - Less distinctive aging patterns
   - More sensitive to positioning variations
   - Explains poor SIDE-only performance (0.496)

#### Why Baseline Performs Poorly

**Baseline (0.310) significantly worse than view-aware methods:**

1. **Loss of view-specific information**
   - Model sees mixed TOP+SIDE images without labels
   - Cannot learn view-specific features (e.g., "if TOP, focus on skin color")
   - Forced to learn view-agnostic features only

2. **Conflicting optimization objectives**
   - TOP images require different feature extractors than SIDE
   - Single shared network must compromise
   - Results in suboptimal features for both views

3. **View-awareness is critical**
   - Late Fusion: +58% improvement over Baseline (0.310 → 0.181)
   - Feature Fusion: +34% improvement over Baseline (0.310 → 0.204)
   - Proves explicit view modeling is essential

#### Model vs Human Performance

| Metric | Best Model | Human (Pre) | Human (Post) | Improvement |
|--------|-----------|-------------|--------------|-------------|
| **MAE** | **0.181 days** | 1.965 days | 2.295 days | **~11-13×** |

**Why models dominate humans:**

1. **Subtle visual patterns**: Models detect imperceptible color/texture changes
2. **Consistency**: No human biases or fatigue
3. **Multi-scale features**: Hierarchical features from local textures to global patterns
4. **Quantitative precision**: Continuous age prediction vs human categorical thinking

**Why human calibration failed:**
- Anchoring bias from reference images
- Task difficulty: Changes too subtle for human perception
- Low inter-rater reliability (ICC: 0.20 → 0.11) indicates inherent human variability

### Implications

**For Deployment:**
- Use **Late Fusion** for production (best generalization, robust)
- **TOP view alone** acceptable if SIDE unavailable (0.205 vs 0.181)
- Avoid Baseline approach (view-awareness critical)

**For Research:**
- CV performance doesn't guarantee test performance (Late: rank 2 → rank 1)
- Simpler fusion can outperform complex fusion on unseen data
- View importance varies: TOP >> SIDE for this task
- Modern architectures (ConvNeXt) outperform older CNNs (ResNet)

## Repository Structure

```
Project/
├── Scripts/
│   ├── training/
│   │   ├── train_best_model.py
│   │   ├── train_other_strategies.py
│   │   └── train_custom.py
│   └── evaluating/
│       ├── evaluate_best_model.py
│       ├── compare_best_model.py
│       ├── evaluate_all_models.py
│       └── evaluate_custom.py
├── Model/
│   ├── model.py              # Architecture definitions
│   ├── dataset.py            # Data loading
│   └── __init__.py
├── Results/
│   ├── convnext_t/          # Best model detailed results
│   │   ├── metrics.json
│   │   ├── other_strategies_metrics.json
│   │   ├── csv/
│   │   └── graphs/
│   ├── comparison/          # All models comparison
│   │   ├── csv/
│   │   └── graphs/
│   └── other_models/        # Custom evaluations
├── Labels/
│   ├── train.csv, val.csv, test.csv
│   ├── generate_labels.py
│   └── split_labels.py
├── checkpoints/            # Saved model weights
├── Dataset_Processed/      # 224x224 RGB images
└── User_Study/            # Human performance evaluation
```

## Quick Start

### 1. Evaluate Best Model (ConvNeXt-T Feature Fusion)

Generate detailed analysis plots:

```bash
cd c:\Users\xadam\OneDrive\Documents\Project
python Scripts/evaluating/evaluate_best_model.py
```

**Output** (`Results/convnext_t/`):
- `scatter_plot.png` - Predicted vs actual with metrics
- `confusion_matrix.png` - Age classification matrix
- `error_distribution.png` - Error histogram & box plot
- `per_day_performance.png` - MAE by day
- `predictions.csv` - All predictions with errors
- `metrics.json` - Performance metrics

### 2. Compare All 5 Strategies

Compare Feature Fusion, Late Fusion, Baseline, TOP-only, and SIDE-only:

```bash
python Scripts/evaluating/compare_best_model.py
```

**Output** (`Results/comparison/convnext_t/`):
- `summary.csv` - Performance table
- `strategy_comparison.png` - Bar chart comparison
- `val_vs_test.png` - Generalization analysis
- `grouped_comparison.png` - Single-view vs multi-view
- `analysis_report.txt` - Statistical analysis

### 3. Evaluate All 27 Models

Compare all backbones and fusion strategies:

```bash
python Scripts/evaluating/evaluate_all_models.py
```

**Output** (`Results/comparison/`):
- `summary.csv` - Complete ranking table
- `all_models_comparison.png` - Performance by fusion type
- `model_size_vs_performance.png` - Efficiency analysis
- `fusion_comparison.png` - Fusion strategy comparison

### 4. Evaluate Custom Model

Evaluate any specific backbone-fusion combination:

```bash
python Scripts/evaluating/evaluate_custom.py --backbone convnext_t --fusion feature
python Scripts/evaluating/evaluate_custom.py --backbone resnet50 --fusion late
```

**Output** (`Results/other_models/{backbone}_{fusion}/`):
- Scatter plots, confusion matrices, error distributions
- Predictions and metrics CSVs

## Dataset

### Included: Dataset_Processed
- **Size**: ~4.8MB (224×224 RGB images)
- **Structure**: TOP VIEW and SIDE VIEW folders by day (1-7)
- **Splits**: 799 train, 169 validation, 183 test samples
- **Content**: 82 unique chicken drumettes tracked across 7 days
- **Split Strategy**: Chicken-level (no chicken appears in multiple splits)

### Data Characteristics
- **Views**: Paired TOP (dorsal) and SIDE (lateral) images
- **Age Range**: Days 1-7 post-slaughter
- **Preprocessing**: Resized to 224×224, normalized for ImageNet models
- **Augmentation** (training only): Random crop, flip, rotation, color jitter

### Full Dataset (Original Resolution)

For **original high-resolution images** (6.8GB):

**Contact**: is0699se@ed.ritsumei.ac.jp
Please include: name, institution, research purpose

## Model Architectures

### Baseline (View-Agnostic)
```python
from Model import ResNetRegressor

model = ResNetRegressor(backbone_name="convnext_t", pretrained=True)
# Training: Mixes TOP+SIDE images (view-agnostic)
# Testing: Averages predictions from both views
```

### Late Fusion (View-Aware Ensemble) ⭐ Best
```python
from Model import LateFusionRegressor

model = LateFusionRegressor(backbone_name="convnext_t", pretrained=True)
# Two separate models (TOP, SIDE) → Average predictions
# ConvNeXt-T Late: 0.181 MAE (test set)
```

### Feature Fusion (Learned Fusion)
```python
from Model import FeatureFusionRegressor

model = FeatureFusionRegressor(backbone_name="convnext_t", pretrained=True)
# Two backbones → Concatenate features → MLP regressor
# ConvNeXt-T Feature: 0.204 MAE (test set)
```

## Training Details

### Hyperparameters
- **Epochs**: 30
- **Batch Size**: 8
- **Optimizer**: AdamW (weight decay 1e-2)
- **Scheduler**: Cosine annealing
- **Learning Rates**:
  - CNNs: 1e-4
  - ConvNeXt: 8e-5
  - Transformers: 5e-5

### Data Augmentation
- Random horizontal flip (p=0.5)
- Random rotation (±10°)
- Color jitter (brightness/contrast=0.2)
- ImageNet normalization

### Training Your Own Models

See **[README_TRAINING.md](README_TRAINING.md)** for detailed training instructions.

Quick training commands:

```bash
# Train best model
python Scripts/training/train_best_model.py

# Train all alternative strategies
python Scripts/training/train_other_strategies.py

# Train specific configuration
python Scripts/training/train_custom.py --backbone convnext_t --fusion feature
```

## User Study: Model vs Human Performance

### Study Design
- **Participants**: 20 individuals
- **Task**: Estimate chicken drumette age (1-7 days)
- **Conditions**: Pre-calibration and post-calibration
- **Images**: 10 test images per condition

### Results

| Metric | Best Model | Human (Pre) | Human (Post) |
|--------|-----------|-------------|--------------|
| **MAE** | **0.181 days** | 1.965 days | 2.295 days |
| **Accuracy** | High | 18.5% | Lower |
| **ICC** | - | 0.20 | 0.11 |

**Key Findings:**
1. **Models outperform humans by 11-13×** in MAE
2. **Calibration paradoxically worsened human performance** (p=0.0108)
   - Anchoring bias from reference images
   - Overthinking effect after seeing examples
3. **Low inter-rater reliability** indicates task difficulty for humans
4. **Models excel at detecting subtle visual changes** imperceptible to humans

### Running the Analysis

```bash
python User_Study/analysis.py
```

**Generates** (`User_Study/Results/`):
- Statistical metrics (console)
- Prediction scatter plots
- Model vs human comparison chart
- Per-participant performance
- Detailed metrics CSV

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA GPU (recommended)
- 8GB+ RAM
- ~5GB disk space

## Hardware Recommendations

**Minimum:**
- CPU: 4 cores
- RAM: 8GB
- GPU: Optional (slow without)

**Recommended:**
- CPU: 8+ cores
- RAM: 16GB+
- GPU: NVIDIA 8GB+ VRAM (RTX 3080+)
- Training time: ~3-5 hours per model

## Documentation

- **[README_TRAINING.md](README_TRAINING.md)** - Comprehensive training guide
- **Code docstrings** - Detailed inline documentation

## Author

**Adam Kumar**
Information System Science and Engineering
Visual Information Engineering Laboratory
Ritsumeikan University

**Supervisor**: Damon Chandler
**Contact**: is0699se@ed.ritsumei.ac.jp

## Acknowledgments

- Supervisor: Damon Chandler, Visual Information Engineering Lab, Ritsumeikan University
- PyTorch pretrained models (ImageNet)
- Dataset collected at Ritsumeikan University for chicken age estimation research

## Contact

**For questions about code or dataset:**
- Email: is0699se@ed.ritsumei.ac.jp
- Include: name, institution, research purpose

**Institution:**
Visual Information Engineering Laboratory
Information System Science and Engineering
Ritsumeikan University, Japan

---

**Project Status**: Active research. Paper in preparation for publication.
