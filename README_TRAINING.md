# Training & Evaluation Guide

Complete guide for training and evaluating all 27 model configurations in the chicken age estimation project.

> **See [README.md](README.md) for project overview, installation, and model architectures.**

---

## Table of Contents
- [Quick Reference](#quick-reference)
- [Training Scripts](#training-scripts)
- [Evaluation Scripts](#evaluation-scripts)
- [Training Details](#training-details)
- [Output Structure](#output-structure)
- [Progress Tracking](#progress-tracking)
- [Model Configurations](#model-configurations)
- [Troubleshooting](#troubleshooting)
- [Tips for Best Results](#tips-for-best-results)

---

## Quick Reference

### Training Scripts (Scripts/training/)

| Script | Purpose | Output | Time Estimate |
|--------|---------|--------|---------------|
| **train_all_models.py** | Train all 27 models (9 baseline + 18 fusion) | Results/baseline_cv_results.json<br>Results/comparison/csv/progress.json | 40-70 hours |
| **train_best_model.py** | Train ConvNeXt-T Feature Fusion only | checkpoints/convnext_t_feature_best.pth<br>Results/convnext_t/graphs/training_curves.png | 3-5 hours |
| **train_custom.py** | Train specific backbone + fusion | checkpoints/{backbone}_{fusion}_best.pth<br>Results/custom_training/graphs/ | 1-3 hours |
| **train_single_view.py** | Train ConvNeXt-T with TOP/SIDE separately | Results/single_view_results.json | 6-10 hours |

### Evaluation Scripts (Scripts/evaluating/)

| Script | Purpose | Output |
|--------|---------|--------|
| **evaluate_all_models.py** | Compare all 27 configurations | Results/comparison/csv/summary.csv<br>Results/comparison/graphs/ |
| **evaluate_best_model.py** | Detailed ConvNeXt-T analysis | Results/convnext_t/csv/predictions.csv<br>Results/convnext_t/graphs/ |
| **evaluate_custom.py** | Evaluate specific model with detailed analysis | Results/other_models/{backbone}_{fusion}/csv/<br>Results/other_models/{backbone}_{fusion}/graphs/ |

---

## Training Scripts

### Option 1: Quick Start (Recommended)

Evaluate existing trained models:

```bash
# Compare all 27 configurations
python Scripts/evaluating/evaluate_all_models.py

# Detailed best model analysis
python Scripts/evaluating/evaluate_best_model.py

# Evaluate specific model with full analysis
python Scripts/evaluating/evaluate_custom.py --backbone convnext_t --fusion feature
python Scripts/evaluating/evaluate_custom.py --backbone resnet50 --fusion late
```

### Option 2: Train Specific Model

Train a specific configuration:

```bash
# Train ConvNeXt-T with Feature Fusion
python Scripts/training/train_custom.py --backbone convnext_t --fusion feature

# Train ResNet-50 with Late Fusion
python Scripts/training/train_custom.py --backbone resnet50 --fusion late --epochs 50

# Train EfficientNet-B0 baseline (single-view)
python Scripts/training/train_custom.py --backbone efficientnet_b0 --fusion baseline
```

**Available options:**
- Backbones: `efficientnet_b0`, `resnet18`, `resnet50`, `resnet101`, `vit_b_16`, `swin_t`, `swin_b`, `convnext_t`, `convnext_b`
- Fusion: `baseline` (view-agnostic), `late` (view-aware ensemble), `feature` (learned fusion)
- Additional args: `--folds`, `--epochs`, `--batch_size`

### Option 3: Train Best Model Only

Train the champion model from scratch:

```bash
python Scripts/training/train_best_model.py
```

This trains ConvNeXt-T Feature Fusion with 3-fold CV and saves:
- Checkpoint: `checkpoints/convnext_t_feature_best.pth`
- Training curves: `Results/convnext_t/graphs/training_curves.png`

### Option 4: Train Everything (Long Running)

Train all 27 models from scratch:

```bash
python Scripts/training/train_all_models.py
```

**Warning:** This takes 40-70 hours with GPU. The script is resumable - if interrupted, it will continue from the last completed fold.

---

## Evaluation Scripts

### Compare All Models

```bash
python Scripts/evaluating/evaluate_all_models.py
```

Generates comprehensive comparison:
- `Results/comparison/csv/summary.csv` - All 27 models ranked
- `Results/comparison/graphs/all_models_comparison.png` - Performance by fusion type
- `Results/comparison/graphs/model_size_vs_performance.png` - Efficiency analysis
- `Results/comparison/graphs/fusion_comparison.png` - Fusion strategy comparison

### Evaluate Best Model

```bash
python Scripts/evaluating/evaluate_best_model.py
```

Detailed ConvNeXt-T Feature Fusion analysis:
- `Results/convnext_t/csv/predictions.csv` - All predictions with errors
- `Results/convnext_t/graphs/scatter_plot.png` - Predicted vs actual with metrics
- `Results/convnext_t/graphs/confusion_matrix.png` - Age classification matrix
- `Results/convnext_t/graphs/error_distribution.png` - Histogram and box plot
- `Results/convnext_t/graphs/per_day_performance.png` - MAE breakdown by day

### Evaluate Specific Model

```bash
python Scripts/evaluating/evaluate_custom.py --backbone convnext_t --fusion feature
python Scripts/evaluating/evaluate_custom.py --backbone resnet50 --fusion late
python Scripts/evaluating/evaluate_custom.py --backbone efficientnet_b0 --fusion baseline
```

**Optional arguments:**
- `--backbone` (required): Model backbone (see list above)
- `--fusion` (required): Fusion type (baseline, late, feature)
- `--checkpoint`: Path to custom checkpoint file

**Output** (in `Results/other_models/{backbone}_{fusion}/`):
- `csv/predictions.csv` - Per-sample predictions with errors
- `csv/metrics.csv` - MAE, RMSE, R², correlation
- `graphs/scatter_plot.png` - Predicted vs actual
- `graphs/confusion_matrix.png` - Age classification
- `graphs/error_distribution.png` - Error histogram and box plot
- `graphs/per_day_performance.png` - MAE by age group

---

## Training Details

### Cross-Validation
- **3-fold CV** at chicken level (no data leakage)
- Each chicken appears in only one fold
- 82 unique chickens split across folds

### Hyperparameters
- **Epochs:** 30 (verified convergence)
- **Batch size:** 8 (memory efficient)
- **Optimizer:** AdamW with weight decay 1e-2
- **Scheduler:** Cosine annealing
- **Learning rates (auto-selected):**
  - CNNs (ResNet, EfficientNet): 1e-4
  - ConvNeXt: 8e-5
  - Transformers (ViT, Swin): 5e-5

### Data Augmentation (Training Only)
- Random horizontal flip (p=0.5)
- Random rotation (±10°)
- Color jitter (brightness=0.2, contrast=0.2)
- Gaussian blur (p=0.2)
- ImageNet normalization

### Understanding Training Output

Example training log:
```
Epoch 18/30 | train loss 0.3245 MAE 0.324 | val loss 0.4302 MAE 0.430
  -> New best MAE. Checkpoint saved to checkpoints/convnext_t_feature_best.pth
```

**What to look for:**

✅ **Good Training:**
- Train and val MAE decrease together
- Val MAE improves periodically
- Small gap between train/val MAE (< 0.2 days)

⚠️ **Overfitting:**
- Train MAE much lower than val MAE (gap > 0.5 days)
- Val MAE stops improving while train continues decreasing

⚠️ **Underfitting:**
- Both train and val MAE remain high (> 1.5 days)
- Little improvement over epochs

---

## Output Structure

### Training Outputs

```
Results/
├── comparison/
│   └── csv/
│       └── progress.json              # Late/Feature fusion CV results
├── baseline_cv_results.json           # Baseline models CV results
├── single_view_results.json           # TOP/SIDE only analysis
└── convnext_t/
    └── graphs/
        └── training_curves.png        # Best model training curves

checkpoints/
├── convnext_t_feature_best.pth        # Best model weights
└── {backbone}_{fusion}_best.pth       # Custom model weights
```

### Evaluation Outputs

```
Results/
├── comparison/
│   ├── csv/
│   │   └── summary.csv                # All 27 models ranked
│   └── graphs/
│       ├── all_models_comparison.png
│       ├── model_size_vs_performance.png
│       └── fusion_comparison.png
├── convnext_t/
│   ├── csv/
│   │   └── predictions.csv            # Test set predictions
│   └── graphs/
│       ├── scatter_plot.png
│       ├── confusion_matrix.png
│       ├── error_distribution.png
│       └── per_day_performance.png
└── other_models/
    └── {backbone}_{fusion}/
        ├── csv/
        │   ├── predictions.csv
        │   └── metrics.csv
        └── graphs/
            ├── scatter_plot.png
            ├── confusion_matrix.png
            ├── error_distribution.png
            └── per_day_performance.png
```

---

## Progress Tracking

All training scripts support **resumable training**:
- Progress saved after each fold
- Skips already completed experiments
- Can safely interrupt and restart

To check progress:
```bash
# Check baseline progress
cat Results/baseline_cv_results.json

# Check fusion progress
cat Results/comparison/csv/progress.json
```

---

## Model Configurations

### 27 Total Configurations

**9 Backbones × 3 Fusion Types = 27 Models**

| Fusion Type | Description | Models |
|-------------|-------------|--------|
| **Baseline** | View-agnostic (TOP+SIDE mixed training) | 9 models |
| **Late Fusion** | View-aware ensemble (separate models) | 9 models |
| **Feature Fusion** | Learned fusion (concatenate features) | 9 models |

### Performance Summary

| Rank | Backbone | Fusion | Mean MAE | Std MAE | Params (M) |
|------|----------|--------|----------|---------|------------|
| 1 | ConvNeXt-T | Feature | **0.172** | 0.016 | 28.59 |
| 2 | ConvNeXt-T | Late | 0.197 | 0.006 | 55.64 |
| 3 | ConvNeXt-B | Late | 0.226 | 0.008 | 176.04 |
| 4 | ConvNeXt-B | Feature | 0.270 | 0.058 | 89.05 |
| 5 | Swin-T | Late | 0.280 | 0.004 | 55.04 |

**Key Findings:**
- ConvNeXt-T Feature Fusion achieves best performance
- Feature Fusion generally outperforms Late Fusion
- ConvNeXt architectures dominate top positions
- Model size doesn't correlate directly with performance

---

## Troubleshooting

### CUDA Out of Memory

Reduce batch size:
```bash
python Scripts/training/train_custom.py --backbone convnext_t --fusion feature --batch_size 4
```

### Import Errors

Always run from project root:
```bash
cd c:\Users\xadam\OneDrive\Documents\Project
python Scripts/training/train_custom.py --backbone convnext_t --fusion feature
```

### Resume Training

Training is automatic - just re-run the same command:
```bash
python Scripts/training/train_all_models.py  # Will skip completed experiments
```

### Missing Checkpoint Error

```
[ERROR] Checkpoint not found: checkpoints/convnext_t_feature_best.pth
```

**Solution:** Train the model first
```bash
python Scripts/training/train_best_model.py
```

### Missing Images Error

```
FileNotFoundError: Dataset_Processed/...
```

**Solutions:**
1. Ensure `Dataset_Processed/` folder exists
2. Check `Labels/*.csv` files point to correct paths
3. Regenerate labels if needed:
```bash
cd Labels
python generate_labels.py
python split_labels.py
```

---

## Tips for Best Results

1. ✅ **Start with evaluation** - Use existing CV results before training
2. ✅ **Train best model only** - Use `train_best_model.py` for quick results
3. ✅ **Monitor training curves** - Detect overfitting early
4. ✅ **Use GPU if available** - 40-100x faster than CPU
5. ✅ **Check progress regularly** - View JSON files to track completed folds
6. ✅ **Save results before retraining** - Copy checkpoints to backup

---

## Complete Workflow

```bash
# 1. Evaluate existing models (< 5 min)
python Scripts/evaluating/evaluate_all_models.py

# 2. Analyze results
# - Check: Results/comparison/csv/summary.csv
# - View: Results/comparison/graphs/

# 3. Train best model if needed (3-5 hours)
python Scripts/training/train_best_model.py

# 4. Detailed evaluation (< 5 min)
python Scripts/evaluating/evaluate_best_model.py

# 5. Compare specific models
python Scripts/evaluating/evaluate_custom.py --backbone resnet50 --fusion late
```

---

## Expected Performance

Based on comprehensive 3-fold CV testing (82 chickens, 1151 total samples):

| Model | Test MAE | Test RMSE | Parameters |
|-------|----------|-----------|------------|
| **ConvNeXt-T Feature** ⭐ | **0.172 ± 0.016** | **0.220** | 28.59M |
| ConvNeXt-T Late | 0.197 ± 0.006 | 0.248 | 55.64M |
| ConvNeXt-B Late | 0.226 ± 0.008 | 0.285 | 176.04M |

**All models significantly outperform humans** (~11-13× lower MAE than human study)

---

## Notes

- All scripts use the same random seed (42) for reproducibility
- GPU is highly recommended (40-100x faster than CPU)
- Results are saved after each fold to prevent data loss
- Training curves help diagnose convergence issues
- Use `evaluate_all_models.py` after training to compare results
- The `checkpoints/` folder stores trained model weights

---

**You're ready to train and evaluate chicken age estimation models!** 🚀

For general info, see [README.md](README.md) | For issues, open a GitHub issue

---

## Contact & Support

**Author:** Adam Kumar (is0699se@ed.ritsumei.ac.jp)
**Institution:** Visual Information Engineering Laboratory, Ritsumeikan University
**Supervisor:** Damon Chandler
