# Training & Evaluation Guide

Complete guide for training and evaluating all 27 model configurations in the chicken age estimation project.

> **See [README.md](README.md) for project overview, results analysis, and research findings.**

---

## Table of Contents
- [Quick Start](#quick-start)
- [Training Scripts](#training-scripts)
- [Evaluation Scripts](#evaluation-scripts)
- [Training Details](#training-details)
- [Output Structure](#output-structure)
- [Progress Tracking](#progress-tracking)
- [Model Configurations](#model-configurations)
- [Grad-CAM Visualization](#grad-cam-visualization)
- [Troubleshooting](#troubleshooting)
- [Tips for Best Results](#tips-for-best-results)

---

## Quick Start

### Evaluate Existing Results (Fastest)

All 27 models have been pre-trained. Start by exploring existing results:

```bash
# Compare all 27 models
python Scripts/evaluating/evaluate_all_models.py

# Detailed best model analysis
python Scripts/evaluating/evaluate_best_model.py

# Compare fusion strategies (TOP, Late, Feature)
python Scripts/evaluating/compare_best_model.py
```

**Generated files:**
- [Results/comparison/csv/summary.csv](Results/comparison/csv/summary.csv) - All 27 models ranked
- [Results/comparison/graphs/](Results/comparison/graphs/) - Comparison visualizations
- [Results/best_model/](Results/best_model/) - ConvNeXt-B Feature Fusion detailed results

### Train Best Model (for Checkpoints/Hyperparameter Tuning)

If you need fresh checkpoints or want to tune hyperparameters:

```bash
python Scripts/training/train_best_model.py
```

**Purpose**: Generates checkpoints for Grad-CAM analysis and provides baseline for hyperparameter experiments

**Output**:
- `checkpoints/convnext_b_feature_fold{0,1,2}.pth` - 3-fold CV checkpoints
- `Results/best_model/metrics.json` - Training history and performance
- Training time: ~3-5 hours with GPU

**Note**: Results may vary slightly (±0.05 MAE) from reported 0.6099 MAE due to random initialization variance. For reproducible comparison, use results from `train_all_models.py` (already run).

### Train Everything from Scratch (Optional)

Re-train all 27 models (only if you need to modify architectures or hyperparameters):

```bash
python Scripts/training/train_all_models.py
```

**Warning**: 40-70 hours with GPU. Resumable if interrupted.

---

## Training Scripts

### Training Script Overview

| Script | Purpose | When to Use | Output | Time |
|--------|---------|-------------|--------|------|
| **train_all_models.py** | Train all 27 models (3-fold CV) | Research: comprehensive comparison | Results/comparison/csv/all_cv_results.json | 40-70 hrs |
| **train_best_model.py** | Train ConvNeXt-B Feature Fusion only | Generate checkpoints for Grad-CAM or hyperparameter tuning | checkpoints/convnext_b_feature_fold*.pth<br>Results/best_model/ | 3-5 hrs |
| **train_other_strategies.py** | Train TOP/SIDE/Late/Feature comparison | Compare strategies within single backbone | Results/other_strategies/convnext_b/ | 30-40 min |
| **train_custom.py** | Train specific backbone+fusion | Experiment with specific configuration | checkpoints/{backbone}_{fusion}_best.pth | 1-3 hrs |

---

## Evaluation Scripts

### Evaluation Script Overview

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| **evaluate_all_models.py** | Compare all 27 models | Results/comparison/csv/all_cv_results.json | Results/comparison/ (graphs + summary) |
| **evaluate_best_model.py** | Detailed ConvNeXt-B analysis | checkpoints/convnext_b_feature_fold*.pth | Results/best_model/ (predictions, graphs) |
| **compare_best_model.py** | Compare fusion strategies | Results/best_model/ + Results/other_strategies/ | Results/other_strategies/convnext_b/graphs/ |
| **evaluate_custom.py** | Evaluate specific model | Custom checkpoint | Results/other_models/{backbone}/ |

### Compare All 27 Models

```bash
python Scripts/evaluating/evaluate_all_models.py
```

**Generates:**
- `Results/comparison/csv/summary.csv` - All models ranked by MAE
- `Results/comparison/graphs/all_models_comparison.png` - Performance by fusion type
- `Results/comparison/graphs/model_size_vs_performance.png` - Efficiency analysis
- `Results/comparison/graphs/fusion_comparison.png` - Fusion strategy comparison

### Evaluate Best Model (ConvNeXt-B Feature Fusion)

```bash
python Scripts/evaluating/evaluate_best_model.py
```

**Generates:**
- `Results/best_model/csv/predictions.csv` - Per-sample predictions with errors
- `Results/best_model/graphs/scatter_plot.png` - Predicted vs actual
- `Results/best_model/graphs/confusion_matrix.png` - Age classification matrix
- `Results/best_model/graphs/error_distribution.png` - Error histogram & box plot
- `Results/best_model/graphs/per_day_performance.png` - MAE breakdown by day
- `Results/best_model/graphs/training_curves.png` - 3-fold CV training progression

### Compare Fusion Strategies

```bash
python Scripts/evaluating/compare_best_model.py
```

Compares TOP View, Late Fusion, and Feature Fusion for ConvNeXt-B.

**Generates:**
- `Results/other_strategies/convnext_b/csv/summary.csv` - Strategy comparison table
- `Results/other_strategies/convnext_b/graphs/strategy_comparison.png` - Bar chart
- `Results/other_strategies/convnext_b/graphs/grouped_comparison.png` - Single vs multi-view
- `Results/other_strategies/convnext_b/analysis_report.txt` - Statistical analysis

### Evaluate Custom Model

```bash
# Evaluate specific backbone+fusion combination
python Scripts/evaluating/evaluate_custom.py --backbone convnext_t --fusion feature
python Scripts/evaluating/evaluate_custom.py --backbone resnet50 --fusion late

# Use custom checkpoint
python Scripts/evaluating/evaluate_custom.py --backbone convnext_b --fusion feature --checkpoint checkpoints/my_model.pth
```

**Available options:**
- `--backbone`: efficientnet_b0, resnet18, resnet50, resnet101, vit_b_16, swin_t, swin_b, convnext_t, convnext_b
- `--fusion`: baseline (TOP view only), late, feature
- `--checkpoint`: Path to .pth file (optional)

**Output**: `Results/other_models/{backbone}/`
- CSV predictions, metrics
- Scatter plots, confusion matrices, error distributions

---

## Training Details

### Option 1: Train All 27 Models (Comprehensive)

```bash
python Scripts/training/train_all_models.py
```

**What it does:**
- Trains all 9 backbones × 3 strategies = 27 models
- 3-fold cross-validation at chicken level
- Saves results to `Results/comparison/csv/all_cv_results.json`
- **Does NOT save model checkpoints** (only metrics)
- Resumable: Skip completed experiments if interrupted

**Time**: 40-70 hours with GPU (GTX 3080+)

**Output:**
- `Results/comparison/csv/all_cv_results.json` - Complete results
- `Results/comparison/csv/top_view_cv_results.json` - TOP view baseline results
- `Results/comparison/csv/progress.json` - Training progress tracker

**When to use**: Research comparison of all architectures and fusion strategies

### Option 2: Train Best Model (for Checkpoints)

```bash
python Scripts/training/train_best_model.py
```

**What it does:**
- Trains ConvNeXt-B Feature Fusion with 3-fold CV
- Saves checkpoints for Grad-CAM analysis
- Generates training curves
- Provides baseline for hyperparameter tuning experiments

**Time**: 3-5 hours with GPU

**Output:**
- `checkpoints/convnext_b_feature_fold0.pth` - Fold 0 checkpoint (best fold)
- `checkpoints/convnext_b_feature_fold1.pth` - Fold 1 checkpoint
- `checkpoints/convnext_b_feature_fold2.pth` - Fold 2 checkpoint
- `Results/best_model/metrics.json` - Training history with per-epoch MAE

**When to use**:
- Need checkpoints for Grad-CAM visualization
- Hyperparameter tuning experiments (see [HYPERPARAMETER_TUNING_GUIDE.md](HYPERPARAMETER_TUNING_GUIDE.md))
- Fresh training run for verification

**Note**: Performance may differ slightly from `train_all_models.py` (0.6099 MAE) due to random variance. For reported results, use comparison metrics.

### Option 3: Train Comparison Strategies

```bash
python Scripts/training/train_other_strategies.py
```

**What it does:**
- Trains 4 strategies using ConvNeXt-B: TOP View, SIDE View, Late Fusion, Feature Fusion (holdout split)
- Enables detailed strategy comparison for a single backbone

**Time**: 30-40 minutes with GPU

**Output:**
- `checkpoints/convnext_b_{top,side,late,feature}_holdout.pth` - 4 checkpoints
- `Results/other_strategies/convnext_b/other_strategies_metrics.json` - Comparison metrics

**When to use**: Detailed fusion strategy analysis for ConvNeXt-B

### Option 4: Train Custom Configuration

```bash
# Train specific backbone + fusion
python Scripts/training/train_custom.py --backbone convnext_t --fusion feature

# With custom hyperparameters
python Scripts/training/train_custom.py --backbone resnet50 --fusion late --epochs 50 --batch_size 16

# Different number of folds
python Scripts/training/train_custom.py --backbone swin_t --fusion feature --folds 5
```

**Available options:**
- `--backbone` (required): Model architecture
- `--fusion` (required): Fusion strategy
- `--epochs`: Training epochs (default: 30)
- `--batch_size`: Batch size (default: 8)
- `--folds`: Number of CV folds (default: 3)

**Output:**
- `checkpoints/{backbone}_{fusion}_best.pth` - Best model checkpoint
- `Results/custom_training/graphs/training_curves.png` - Training progression

**When to use**: Experimentation with specific configurations

---

## Cross-Validation Setup

### Data Split Strategy

**3-Fold Cross-Validation at Chicken Level:**
- 82 unique chickens split into 3 folds
- Each chicken appears in only one fold (prevents data leakage)
- Each fold: ~57 train, ~12 val, ~13 test chickens
- Ensures generalization to unseen chickens

**sklearn.model_selection.KFold:**
```python
kfold = KFold(n_splits=3, shuffle=True, random_state=42)
```

### Why Chicken-Level Splitting?

**Problem**: Image-level splitting would allow same chicken in train and test
**Solution**: Split by unique chicken IDs before creating folds
**Benefit**: Tests true generalization to unseen individuals

---

## Hyperparameters

### Best Model Hyperparameters (train_best_model.py)

```python
# ConvNeXt-B Feature Fusion
BACKBONE = "convnext_b"
EPOCHS = 50
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 1e-2
WARMUP_EPOCHS = 3
EARLY_STOP_PATIENCE = 10
LR_SCHEDULER = "plateau"  # ReduceLROnPlateau
RANDOM_SEED = 42
```

**Typical convergence**: Best epoch 15-25, early stops around epoch 30-35

### All Models Hyperparameters (train_all_models.py)

```python
# Used for comprehensive 27-model comparison
EPOCHS = 30
BATCH_SIZE = 8
WEIGHT_DECAY = 1e-2
SCHEDULER = "cosine"  # CosineAnnealingLR
RANDOM_SEED = 42

# Learning rates (architecture-dependent):
LR_CONVNEXT = 8e-5
LR_CNN = 1e-4  # ResNet, EfficientNet
LR_TRANSFORMER = 5e-5  # ViT, Swin
```

### Data Augmentation (Training Only)

```python
# Applied to training set only
transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.9, 1.0), ratio=(0.95, 1.05)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.RandomRotation(10)], p=0.3),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
    transforms.RandomApply([transforms.GaussianBlur(5, sigma=(0.1, 0.5))], p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

**Critical**: Segmentation masks applied BEFORE augmentation (background → white)

**Why simple augmentation?**
- Dataset is small (~250 chickens)
- Aggressive augmentation causes overfitting
- Simple augmentation (scale 0.9-1.0, ±10° rotation) works best

---

## Output Structure

### Training Outputs

```
Results/
├── comparison/                 # From train_all_models.py (ALL 27 models)
│   ├── csv/
│   │   ├── all_cv_results.json       # Complete 3-fold CV results for all models
│   │   ├── top_view_cv_results.json  # TOP view baseline results
│   │   ├── summary.csv               # Ranked comparison table
│   │   └── progress.json             # Training progress tracker
│   ├── graphs/
│   │   ├── all_models_comparison.png
│   │   ├── fusion_comparison.png
│   │   └── model_size_vs_performance.png
│   └── statistical_analysis.txt      # Model size vs performance analysis
│
├── best_model/                # From train_best_model.py (ConvNeXt-B Feature Fusion)
│   ├── metrics.json                  # 3-fold CV results + training history
│   ├── csv/
│   │   └── predictions.csv           # Test set predictions
│   └── graphs/
│       ├── scatter_plot.png
│       ├── confusion_matrix.png
│       ├── error_distribution.png
│       ├── per_day_performance.png
│       └── training_curves.png       # 3-fold CV training progression
│
├── other_models/              # From evaluate_custom.py (individual models)
│   └── {backbone}/                   # e.g., convnext_t
│       ├── csv/
│       │   ├── predictions.csv
│       │   └── metrics.json
│       └── graphs/
│           ├── scatter_plot.png
│           ├── confusion_matrix.png
│           ├── error_distribution.png
│           └── per_day_performance.png
│
└── other_strategies/          # From train_other_strategies.py & compare_best_model.py
    ├── convnext_b/                   # Strategy comparison for ConvNeXt-B
    │   ├── csv/
    │   │   └── summary.csv
    │   ├── graphs/
    │   │   ├── strategy_comparison.png
    │   │   └── grouped_comparison.png
    │   └── analysis_report.txt
    └── convnext_t/                   # Strategy comparison for ConvNeXt-T (if run)
```

### Checkpoint Files

```
checkpoints/
├── convnext_b_feature_fold0.pth      # Best model, Fold 0 (best performing)
├── convnext_b_feature_fold1.pth      # Best model, Fold 1
├── convnext_b_feature_fold2.pth      # Best model, Fold 2
├── convnext_b_{top,side,late}_holdout.pth  # Comparison strategies (if trained)
└── {backbone}_{fusion}_*.pth         # Custom model checkpoints

temp/                                  # Temporary files (gitignored)
├── all_models/                        # Progress from train_all_models.py
└── other_strategies/                  # Progress from train_other_strategies.py
```

---

## Progress Tracking

All training scripts support **resumable training**:
- Progress saved after each fold to JSON files
- Skips already completed experiments on restart
- Safe to interrupt (Ctrl+C) and resume later

### Check Training Progress

```bash
# Check all models progress
cat Results/comparison/csv/progress.json

# Check completed models
python -m json.tool Results/comparison/csv/all_cv_results.json | grep -A 5 "convnext_b_feature"

# Check best model progress
cat Results/best_model/metrics.json
```

### Resume Training

Simply re-run the same command:
```bash
# Will skip completed experiments
python Scripts/training/train_all_models.py
```

---

## Model Configurations

### 27 Total Configurations

**9 Backbones × 3 Fusion Types = 27 Models**

| Fusion Type | Description | Count |
|-------------|-------------|-------|
| **TOP View Only** | Single-view baseline | 9 |
| **Late Fusion** | Independent models, averaged predictions | 9 |
| **Feature Fusion** | Learned fusion, concatenated features | 9 |

### Expected Performance (3-Fold CV)

Based on comprehensive evaluation (`train_all_models.py` results):

#### Top 10 Models

| Rank | Backbone | Fusion | Mean MAE | Std MAE | Params (M) |
|------|----------|--------|----------|---------|------------|
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

- **ConvNeXt dominates**: Top 4 models all use ConvNeXt
- **Feature Fusion > Late Fusion**: Learned fusion consistently outperforms averaging
- **ConvNeXt-T exceptional consistency**: std = 0.001 (lowest variance)
- **Parameter efficiency**: ConvNeXt-T (28M) nearly matches ConvNeXt-B (89M)

---

## Grad-CAM Visualization

Generate saliency maps showing which image regions drive predictions:

```bash
# Visualize 20 random test samples (fold 0 checkpoint)
python Analysis/gradcam_visualization.py --num_samples 20

# Use different fold checkpoint
python Analysis/gradcam_visualization.py --fold 1 --num_samples 20

# Visualize specific day
python Analysis/gradcam_visualization.py --day 5 --num_samples 10

# Visualize all test samples
python Analysis/gradcam_visualization.py --num_samples 200
```

**Output**: `Analysis/Results/gradcam_fold{N}/`
- Individual visualizations: `gradcam_sample_XXX_dayX_errorX.XX.png`
- Summary statistics: `gradcam_summary.csv`
- Summary plots: `gradcam_summary.png`

**Model**: Uses ConvNeXt-B Feature Fusion checkpoints from `train_best_model.py`

**Documentation**: See [Analysis/README.md](Analysis/README.md)

---

## Troubleshooting

### CUDA Out of Memory

**Symptom**: `RuntimeError: CUDA out of memory`

**Solutions:**
```bash
# Option 1: Reduce batch size
python Scripts/training/train_custom.py --backbone convnext_b --fusion feature --batch_size 4

# Option 2: Use smaller model
python Scripts/training/train_custom.py --backbone convnext_t --fusion feature --batch_size 8

# Option 3: Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"
```

### Import Errors

**Symptom**: `ModuleNotFoundError: No module named 'Model'`

**Solution**: Always run from project root
```bash
cd c:\Users\xadam\OneDrive\Documents\Project
python Scripts/training/train_best_model.py
```

### Missing Checkpoint Error

**Symptom**:
```
[ERROR] Checkpoint not found: checkpoints/convnext_b_feature_fold0.pth
```

**Solution**: Train the model first
```bash
python Scripts/training/train_best_model.py
```

### Missing Dataset Error

**Symptom**:
```
FileNotFoundError: Dataset_Processed/TOP VIEW/Day 1/...
```

**Solutions:**
1. Ensure `Dataset_Processed/` exists with TOP VIEW and SIDE VIEW folders
2. Check `Labels/*.csv` files have correct paths
3. Regenerate labels if needed:
```bash
cd Labels
python generate_labels.py
python split_labels.py
```

### Training Not Converging

**Symptom**: Val MAE stays high (> 1.5 days) after many epochs

**Diagnosis:**
```bash
# Check training curves in Results/best_model/graphs/training_curves.png
# Look for:
# - Overfitting: train MAE << val MAE (gap > 0.5)
# - Underfitting: both high (> 1.5)
# - Learning: both decreasing together
```

**Solutions:**
- **Overfitting**: Reduce learning rate, increase weight decay
- **Underfitting**: Increase learning rate, reduce weight decay
- **Not learning**: Check data loading, verify masks applied

---

## Tips for Best Results

1. ✅ **Start with evaluation** - Explore existing results before training
2. ✅ **Use GPU** - 40-100× faster than CPU
3. ✅ **Train best model for checkpoints** - Use `train_best_model.py` for Grad-CAM
4. ✅ **Monitor training curves** - Detect overfitting early
5. ✅ **Check progress regularly** - View JSON files to track folds
6. ✅ **Save before retraining** - Backup checkpoints before overwriting
7. ✅ **Use random seed** - All scripts use seed=42 for reproducibility
8. ✅ **Verify segmentation masks** - Check `Segmentation/masks/` exists

---

## Understanding Training Output

Example training log:
```
================================================================================
TRAINING: ConvNeXt-B Feature Fusion (Fold 1/3)
================================================================================

Epoch 18/50
--------------------------------------------------------------------------------
  Train | loss: 0.3245  MAE: 0.324 days
  Val   | loss: 0.4302  MAE: 0.430 days

  -> New best MAE (0.430 < 0.456). Checkpoint saved.
  -> Learning rate: 2.00e-05
--------------------------------------------------------------------------------
```

**What to look for:**

✅ **Good Training:**
- Train and val MAE decrease together
- Val MAE improves periodically
- Small gap between train/val MAE (< 0.2 days)
- Checkpoints saved every few epochs

⚠️ **Overfitting:**
- Train MAE much lower than val MAE (gap > 0.5 days)
- Val MAE stops improving while train continues decreasing
- Early stopping triggers before max epochs

⚠️ **Underfitting:**
- Both train and val MAE remain high (> 1.5 days)
- Little improvement over epochs
- May need more epochs or higher learning rate

---

## Complete Workflow Example

```bash
# 1. Evaluate existing models (< 5 min)
python Scripts/evaluating/evaluate_all_models.py

# 2. Analyze comparison results
# Check: Results/comparison/csv/summary.csv
# View: Results/comparison/graphs/

# 3. Detailed best model evaluation (< 5 min)
python Scripts/evaluating/evaluate_best_model.py

# 4. Compare fusion strategies (< 5 min)
python Scripts/evaluating/compare_best_model.py

# 5. Generate Grad-CAM visualizations (2-5 min)
python Analysis/gradcam_visualization.py --num_samples 20

# 6. (Optional) Train fresh checkpoints for experiments (3-5 hours)
python Scripts/training/train_best_model.py

# 7. (Optional) Hyperparameter tuning
# See HYPERPARAMETER_TUNING_GUIDE.md for detailed instructions
```

---

## Hyperparameter Tuning

For detailed hyperparameter optimization guide, see **[HYPERPARAMETER_TUNING_GUIDE.md](HYPERPARAMETER_TUNING_GUIDE.md)**.

**Quick start tuning:**
1. Reduce learning rate: `LR = 1e-5` (instead of 2e-5)
2. Add dropout: `DROPOUT = 0.2` in fusion head
3. Use cosine scheduler: `SCHEDULER = "cosine"`

---

## Notes

- All scripts use random seed 42 for reproducibility
- GPU highly recommended (40-100× faster than CPU)
- Training is resumable - safe to interrupt and restart
- Results saved after each fold to prevent data loss
- `train_all_models.py` results (0.6099 MAE) are used for reported performance
- `train_best_model.py` is for generating checkpoints and hyperparameter tuning only
- Checkpoints stored in `checkpoints/` folder
- Use `evaluate_all_models.py` after training to compare results

---

**You're ready to train and evaluate chicken age estimation models!** 🚀

For project overview and research findings, see [README.md](README.md)

For issues or questions, contact: is0699se@ed.ritsumei.ac.jp

---

## Contact & Support

**Author:** Adam Kumar (is0699se@ed.ritsumei.ac.jp)
**Institution:** Visual Information Engineering Laboratory, Ritsumeikan University
**Supervisor:** Damon Chandler
