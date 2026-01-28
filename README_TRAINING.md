# Training & Evaluation Guide

Complete guide for training and evaluating all 27 model configurations in the chicken age estimation project.

> **See [README.md](README.md) for project overview, results analysis, and research findings.**

---

## Table of Contents
- [Quick Start](#quick-start)
- [Scripts Overview](#scripts-overview)
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
# Generate all visualizations and analysis
python Scripts/generate_final_analysis.py
```

**Generated files:**
- [Results/CSV_and_Analysis/summary.csv](Results/CSV_and_Analysis/summary.csv) - All 27 models ranked
- [Results/Graphs/](Results/Graphs/) - All visualization files (9 PNG files)
- [Results/CSV_and_Analysis/](Results/CSV_and_Analysis/) - Statistical analysis and tables

### Train All Models from Scratch

Re-train all 27 models (only if you need to modify architectures or hyperparameters):

```bash
python Scripts/train_all_models_full.py
```

**What it does:**
- Trains all 9 backbones × 3 strategies = 27 models
- 3-fold cross-validation at chicken level
- Saves checkpoints, training histories, and predictions
- Resumable: Skip completed experiments if interrupted

**Time**: 40-70 hours with GPU (GTX 3080+)

**Output:**
- `Results/Data/checkpoints/` - All model checkpoints (81 .pth files)
- `Results/Data/training_history/` - Per-epoch metrics (81 .json files)
- `Results/Data/predictions/` - Validation predictions (81 .csv files)
- `Results/CSV_and_Analysis/full_progress.json` - Complete training results

---

## Scripts Overview

The project uses 2 main scripts for training and evaluation:

| Script | Purpose | Time | Output |
|--------|---------|------|--------|
| **train_all_models_full.py** | Train all 27 models (3-fold CV) | 40-70 hrs | Results/Data/ + full_progress.json |
| **generate_final_analysis.py** | Generate all graphs and analysis | < 5 min | Results/Graphs/ + CSV_and_Analysis/ |

### Training: train_all_models_full.py

```bash
python Scripts/train_all_models_full.py
```

**What it does:**
- Trains all 9 backbones × 3 strategies = 27 models
- 3-fold cross-validation at chicken level
- Saves checkpoints, training histories, and predictions for all models
- Resumable: Skip completed experiments if interrupted

**Output:**
- `Results/Data/checkpoints/` - Model weights for all 81 experiments (27 models × 3 folds)
- `Results/Data/training_history/` - Per-epoch metrics (train/val MAE, loss)
- `Results/Data/predictions/` - Validation predictions for all folds
- `Results/CSV_and_Analysis/full_progress.json` - Summary of all training results

### Evaluation: generate_final_analysis.py

```bash
python Scripts/generate_final_analysis.py
```

**What it does:**
- Loads results from `Results/CSV_and_Analysis/full_progress.json`
- Generates all visualizations and statistical analysis
- Creates comparison tables and rankings

**Generates:**

**Graphs (9 PNG files):**
- `all_models_comparison_3panel.png` - 3-panel comparison by fusion type
- `all_models_comparison_single.png` - Single-panel comparison
- `fusion_comparison.png` - Fusion strategy effectiveness
- `model_size_vs_performance.png` - Parameter efficiency analysis
- `training_curves_[best_model].png` - 3-fold CV training progression
- `predictions_vs_actual_[best_model].png` - Predicted vs actual ages
- `confusion_matrix_[best_model].png` - Age classification matrix (normalized)
- `error_distribution_[best_model].png` - Error histogram & Q-Q plot
- `architecture_comparison.png` - CNN vs Transformer vs Hybrid comparison

**CSV and Analysis (6 files):**
- `summary.csv` - All 27 models ranked by MAE
- `top_10_models.csv` - Top performers with statistics
- `fusion_comparison.csv` - Strategy comparison table
- `architecture_comparison.csv` - Backbone comparison
- `statistical_tests.txt` - Significance tests and effect sizes
- `full_progress.json` - Complete training results

---

## Training Details

### Training All 27 Models

```bash
python Scripts/train_all_models_full.py
```

**What it does:**
- Trains all 9 backbones × 3 strategies = 27 models
- 3-fold cross-validation at chicken level
- Saves checkpoints, training histories, and predictions for all experiments
- Resumable: Skip completed experiments if interrupted

**Time**: 40-70 hours with GPU (GTX 3080+)

**Output:**
- `Results/Data/checkpoints/` - All model weights (81 .pth files: 27 models × 3 folds)
- `Results/Data/training_history/` - Per-epoch metrics (81 .json files)
- `Results/Data/predictions/` - Validation predictions (81 .csv files)
- `Results/CSV_and_Analysis/full_progress.json` - Summary of all results

**When to use**:
- Initial training run
- Re-training after architecture or hyperparameter changes
- Generating fresh checkpoints for all models

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

### Training Hyperparameters (train_all_models_full.py)

```python
# Uniform across all 27 models
EPOCHS = 50
BATCH_SIZE = 16
LEARNING_RATE = 2e-5  # Uniform for all architectures
WEIGHT_DECAY = 1e-2
WARMUP_EPOCHS = 3
EARLY_STOP_PATIENCE = 10
LR_SCHEDULER = "plateau"  # ReduceLROnPlateau
RANDOM_SEED = 42
```

**Typical convergence**: Best epoch 15-25, early stops around epoch 30-35

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

### Results Folder Structure

```
Results/
├── Data/                      # All training data (243 files)
│   ├── checkpoints/           # Model weights (81 .pth files)
│   │   └── {backbone}_{fusion}_fold{0,1,2}.pth
│   ├── training_history/      # Per-epoch metrics (81 .json files)
│   │   └── {backbone}_{fusion}_fold{0,1,2}_history.json
│   └── predictions/           # Validation predictions (81 .csv files)
│       └── {backbone}_{fusion}_fold{0,1,2}_predictions.csv
│
├── Graphs/                    # All visualizations (9 PNG files)
│   ├── all_models_comparison_3panel.png
│   ├── all_models_comparison_single.png
│   ├── fusion_comparison.png
│   ├── model_size_vs_performance.png
│   ├── training_curves_vit_b_16_feature.png
│   ├── predictions_vs_actual_vit_b_16_feature.png
│   ├── confusion_matrix_vit_b_16_feature.png
│   ├── error_distribution_vit_b_16_feature.png
│   └── architecture_comparison.png
│
└── CSV_and_Analysis/          # Tables and statistical analysis (6 files)
    ├── full_progress.json     # Complete training results
    ├── summary.csv            # All 27 models ranked
    ├── top_10_models.csv      # Top performers with statistics
    ├── fusion_comparison.csv  # Strategy comparison table
    ├── architecture_comparison.csv  # Backbone comparison
    └── statistical_tests.txt  # Significance tests and effect sizes
```

### Best Model Checkpoints

The best performing model (ViT-B/16 Feature Fusion) checkpoints:
```
Results/Data/checkpoints/
├── vit_b_16_feature_fold0.pth
├── vit_b_16_feature_fold1.pth
└── vit_b_16_feature_fold2.pth
```

For Grad-CAM visualization, use fold 0 checkpoint (best performing fold).

---

## Progress Tracking

Training supports **resumable training**:
- Progress saved after each fold to `full_progress.json`
- Skips already completed experiments on restart
- Safe to interrupt (Ctrl+C) and resume later

### Check Training Progress

```bash
# Check all models progress
cat Results/CSV_and_Analysis/full_progress.json

# Check completed models (Windows PowerShell)
Get-Content Results\CSV_and_Analysis\full_progress.json | ConvertFrom-Json |
  Select-Object -ExpandProperty results | Format-Table -AutoSize

# Check specific model
python -c "import json; data = json.load(open('Results/CSV_and_Analysis/full_progress.json')); print(data['results']['vit_b_16_feature'])"
```

### Resume Training

Simply re-run the same command:
```bash
# Will skip completed experiments
python Scripts/train_all_models_full.py
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

Based on comprehensive evaluation (`train_all_models_full.py` results):

#### Top 10 Models

| Rank | Backbone | Fusion | Mean MAE | Std MAE | Params (M) |
|------|----------|--------|----------|---------|------------|
| 1 | **ViT-B/16** | **Feature** | **0.660** | 0.027 | 87.34 |
| 2 | ViT-B/16 | Late | 0.698 | 0.025 | 173.13 |
| 3 | ConvNeXt-B | Feature | 0.711 | 0.039 | 89.05 |
| 4 | Swin-T | Late | 0.724 | 0.020 | 55.04 |
| 5 | Swin-T | Feature | 0.730 | 0.021 | 28.29 |
| 6 | ConvNeXt-T | Feature | 0.737 | 0.016 | 28.59 |
| 7 | Swin-B | Late | 0.740 | 0.027 | 174.54 |
| 8 | ResNet-101 | Feature | 0.749 | 0.042 | 44.54 |
| 9 | ConvNeXt-B | Late | 0.765 | 0.047 | 176.04 |
| 10 | Swin-B | Feature | 0.767 | 0.033 | 88.02 |

**Full results**: See [Results/CSV_and_Analysis/summary.csv](Results/CSV_and_Analysis/summary.csv)

#### Key Findings

- **Vision Transformers excel**: ViT-B/16 achieves best performance (0.660 MAE)
- **Feature Fusion > Late Fusion**: Learned fusion consistently outperforms averaging
- **Low variance across folds**: Best model std = 0.027 days
- **Parameter efficiency**: ViT and ConvNeXt models offer best balance of performance and size

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

**Model**: Uses best model checkpoints from `Results/Data/checkpoints/vit_b_16_feature_fold{0,1,2}.pth`

**Documentation**: See [Analysis/README.md](Analysis/README.md)

---

## Troubleshooting

### CUDA Out of Memory

**Symptom**: `RuntimeError: CUDA out of memory`

**Solutions:**
```bash
# Option 1: Edit batch size in train_all_models_full.py
# Change: BATCH_SIZE = 16 -> BATCH_SIZE = 8

# Option 2: Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"

# Option 3: Use smaller models (modify MODEL_CONFIGS in train_all_models_full.py)
```

### Import Errors

**Symptom**: `ModuleNotFoundError: No module named 'Model'`

**Solution**: Always run from project root
```bash
cd c:\Users\xadam\OneDrive\Documents\Project
python Scripts/train_all_models_full.py
```

### Missing Checkpoint Error

**Symptom**:
```
[ERROR] Checkpoint not found: Results/Data/checkpoints/vit_b_16_feature_fold0.pth
```

**Solution**: Train the models first
```bash
python Scripts/train_all_models_full.py
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
# Check training curves in Results/Graphs/best_model_training_curves.png
# Look for:
# - Overfitting: train MAE << val MAE (gap > 0.5)
# - Underfitting: both high (> 1.5)
# - Learning: both decreasing together

# Or check training history directly
python -c "import json; h = json.load(open('Results/Data/training_history/vit_b_16_feature_fold0_history.json')); print([e['val_mae'] for e in h])"
```

**Solutions:**
- **Overfitting**: Reduce learning rate, increase weight decay
- **Underfitting**: Increase learning rate, reduce weight decay
- **Not learning**: Check data loading, verify masks applied

---

## Tips for Best Results

1. ✅ **Start with evaluation** - Run `generate_final_analysis.py` to explore existing results
2. ✅ **Use GPU** - 40-100× faster than CPU
3. ✅ **Monitor training curves** - Check `Results/Graphs/best_model_training_curves.png`
4. ✅ **Check progress regularly** - View `Results/CSV_and_Analysis/full_progress.json`
5. ✅ **Save before retraining** - Backup `Results/Data/` before overwriting
6. ✅ **Use random seed** - All scripts use seed=42 for reproducibility
7. ✅ **Verify segmentation masks** - Check `Segmentation/masks/` exists
8. ✅ **Training is resumable** - Safe to interrupt with Ctrl+C and restart

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
python Scripts/generate_final_analysis.py

# 2. Analyze results
# Check: Results/CSV_and_Analysis/summary.csv
# View: Results/Graphs/

# 3. Generate Grad-CAM visualizations (2-5 min)
python Analysis/gradcam_visualization.py --num_samples 20

# 4. (Optional) Train all models from scratch (40-70 hours)
python Scripts/train_all_models_full.py

# 5. (Optional) Re-run analysis after training
python Scripts/generate_final_analysis.py
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
- Best model: ViT-B/16 Feature Fusion (0.660 ± 0.027 MAE)
- All checkpoints stored in `Results/Data/checkpoints/`
- All visualizations in `Results/Graphs/`
- Use `generate_final_analysis.py` after training to create all graphs and analysis

---

**You're ready to train and evaluate chicken age estimation models!** 🚀

For project overview and research findings, see [README.md](README.md)

For issues or questions, contact: is0699se@ed.ritsumei.ac.jp

---

## Contact & Support

**Author:** Adam Kumar (is0699se@ed.ritsumei.ac.jp)
**Institution:** Visual Information Engineering Laboratory, Ritsumeikan University
**Supervisor:** Damon Chandler
