# Training & Evaluation Guide

Complete guide for training and evaluating chicken age regression models.

> **See [README.md](README.md) for project overview, installation, and model architectures.**

---

## Table of Contents
- [Quick Start](#quick-start)
- [Training Scripts](#training-scripts)
- [Evaluation Scripts](#evaluation-scripts)
- [Training Configuration](#training-configuration)
- [Understanding Training Output](#understanding-training-output)
- [Output Files](#output-files)
- [Troubleshooting](#troubleshooting)
- [Advanced Usage](#advanced-usage)

---

## Quick Start

### Train All Models
```bash
python Model/Training/train_all_models.py
```
**Time:** 1.5-3 hours (GPU) | **Outputs:** Model checkpoints + training curves

### Evaluate All Models
```bash
python Model/Evaluating/evaluate_all_models.py
```
**Outputs:** Comparison table, scatter plots, predictions

---

## Training Scripts

### Train All Models (Recommended)

```bash
python Model/Training/train_all_models.py
```

Trains baseline, late fusion, and feature fusion sequentially.

**Outputs:**
- `Model/checkpoints/best_{model_type}.pth` - Model weights
- `Results/training_curves/train_val_mae_{model_type}.png` - Training progress

### Train Individual Models

```bash
# Baseline (~20-40 min)
python Model/Training/train_baseline.py

# Late Fusion (~30-60 min)
python Model/Training/train_latefusion.py

# Feature Fusion (~30-60 min)
python Model/Training/train_featurefusion.py
```

---

## Evaluation Scripts

### Compare All Models

```bash
python Model/Evaluating/evaluate_all_models.py
```

Generates comprehensive comparison:
- `Results/comparison/metrics_comparison.csv` - Performance table
- `Results/comparison/metrics_comparison.png` - Bar charts (MAE/RMSE)
- `Results/comparison/scatter_comparison.png` - Scatter plots (all models)
- `Results/comparison/predictions_{model}.csv` - Individual predictions

### Evaluate Individual Models

```bash
# Baseline
python Model/Evaluating/evaluate_baseline.py

# Late Fusion
python Model/Evaluating/evaluate_latefusion.py

# Feature Fusion
python Model/Evaluating/evaluate_featurefusion.py
```

**Options:**
```bash
# Evaluate on validation set
python Model/Evaluating/evaluate_baseline.py --split val

# Evaluate on training set
python Model/Evaluating/evaluate_baseline.py --split train

# Custom batch size
python Model/Evaluating/evaluate_baseline.py --batch_size 32
```

---

## Training Configuration

### Default Parameters

Optimized for this dataset (799 train, 169 val, 183 test samples):

```python
epochs = 30              # Training epochs
batch_size = 32          # Images per batch
lr = 1e-4                # Learning rate
weight_decay = 1e-2      # L2 regularization
seed = 42                # Random seed
```

### Optimizer & Scheduler

- **Optimizer:** AdamW (Adam with weight decay)
- **Scheduler:** CosineAnnealingLR (learning rate decay)
- **Loss:** L1Loss (Mean Absolute Error)

### Data Augmentation

**Training:**
- Random resized crop (90-100%)
- Random horizontal flip
- Random rotation (±10°, 30% probability)
- Color jitter (brightness, contrast, saturation, hue)
- Gaussian blur (20% probability)

**Validation/Test:**
- Resize to 224x224
- Center crop
- Normalization (ImageNet mean/std)

### Customizing Parameters

**Method 1: Edit train_all_models.py**

Edit the `CONFIG` dictionary in `Model/Training/train_all_models.py`:

```python
CONFIG = {
    "epochs": 50,        # Train longer
    "batch_size": 16,    # Reduce for GPU memory
    "lr": 5e-5,          # Lower learning rate
    "weight_decay": 1e-2,
    "seed": 42,
}
```

**Method 2: Command-line arguments**

```bash
python Model/Training/train_baseline.py --epochs 50 --batch_size 16 --lr 5e-5
```

**Available arguments:**
- `--epochs` - Number of training epochs
- `--batch_size` - Images per batch
- `--lr` - Learning rate
- `--weight_decay` - L2 regularization strength
- `--seed` - Random seed for reproducibility
- `--freeze_backbone` - Only train final layer (transfer learning)

---

## Understanding Training Output

### Example Training Log

```
Epoch 18/30 | train loss 0.3245 MAE 0.324 | val loss 0.4302 MAE 0.430
  -> New best MAE. Checkpoint saved to Model/checkpoints/best_late_fusion.pth
```

### What to Look For

**✅ Good Training:**
- Train and val MAE decrease together
- Val MAE improves periodically
- Small gap between train/val MAE (< 0.2 days)

**⚠️ Overfitting:**
- Train MAE much lower than val MAE (gap > 0.5 days)
- Val MAE stops improving while train continues decreasing
- Model saves checkpoint early and never improves

**⚠️ Underfitting:**
- Both train and val MAE remain high (> 1.5 days)
- Little improvement over epochs
- Model might need more capacity or longer training

### Interpreting MAE Values

| MAE | Interpretation |
|-----|----------------|
| < 0.5 days | Excellent - predictions very accurate |
| 0.5-1.0 days | Good - acceptable for most applications |
| 1.0-1.5 days | Fair - room for improvement |
| > 1.5 days | Poor - model needs debugging |

---

## Output Files

### Training Outputs

**Checkpoints** (`Model/checkpoints/`):
```
best_baseline.pth          # Baseline model weights
best_late_fusion.pth       # Late fusion weights
best_feature_fusion.pth    # Feature fusion weights
```

Each checkpoint contains:
- `model_state` - Model weights
- `model_type` - Architecture name
- `epoch` - Best epoch number
- `val_mae` - Best validation MAE

**Training Curves** (`Results/training_curves/`):
```
train_val_mae_baseline.png
train_val_mae_late_fusion.png
train_val_mae_feature_fusion.png
```

Shows MAE progression over epochs for train and validation sets.

### Evaluation Outputs

**Comparison Results** (`Results/comparison/`):
```
metrics_comparison.csv     # Performance table
metrics_comparison.png     # Bar charts (MAE/RMSE)
scatter_comparison.png     # Scatter plots (all models)
predictions_*.csv          # Predictions for each model
```

**Individual Plots** (`Results/plots/`):
```
eval_baseline_test_scatter.png
eval_latefusion_test_scatter.png
eval_featurefusion_test_scatter.png
```

---

## Troubleshooting

### Missing Checkpoint Error

```
[ERROR] Checkpoint not found: Model/checkpoints/best_baseline.pth
```

**Solution:** Train the model first
```bash
python Model/Training/train_baseline.py
```

### CUDA Out of Memory

```
RuntimeError: CUDA out of memory
```

**Solutions:**

1. Reduce batch size:
```bash
python Model/Training/train_baseline.py --batch_size 16
```

2. Use CPU (much slower):
```bash
# PyTorch will automatically use CPU if CUDA unavailable
python Model/Training/train_baseline.py
```

### Import Errors

```
ModuleNotFoundError: No module named 'Model'
```

**Solution:** Always run from project root
```bash
cd /path/to/Project
python Model/Training/train_baseline.py
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

## Advanced Usage

### Freeze Backbone (Transfer Learning)

Only train the final layer (faster, prevents overfitting):

```bash
python Model/Training/train_baseline.py --freeze_backbone
```

### Evaluate on Different Splits

```bash
# Validation set
python Model/Evaluating/evaluate_baseline.py --split val

# Training set (check for overfitting)
python Model/Evaluating/evaluate_baseline.py --split train
```

### Re-train a Model

Simply run the training script again:

```bash
python Model/Training/train_baseline.py --epochs 50
```

---

## Expected Performance

Based on our dataset (799 train, 169 val, 183 test):

| Model | Test MAE | Test RMSE | Best Epoch | Time |
|-------|----------|-----------|------------|------|
| **Late Fusion** ⭐ | **0.430** | **0.539** | 18 | ~40-60 min |
| Feature Fusion | 0.465 | 0.569 | 17 | ~40-60 min |
| Baseline | 0.490 | 0.645 | 24 | ~20-40 min |

**Why Late Fusion Wins:**
- Small dataset (< 1000 samples)
- Acts like ensemble (2 models)
- Simple averaging = better generalization

---

## Tips for Best Results

1. ✅ **Train all 3 models** - Compare on your data
2. ✅ **Monitor training curves** - Detect overfitting early
3. ✅ **Use late fusion for small datasets** - Better generalization
4. ✅ **Check train/val gap** - If > 0.5 days, increase regularization
5. ✅ **Save results before retraining** - Copy checkpoints
6. ✅ **Use GPU if available** - 10-20x faster than CPU

---

## Complete Workflow

```bash
# 1. Train all models (1.5-3 hours)
python Model/Training/train_all_models.py

# 2. Evaluate and compare (< 5 min)
python Model/Evaluating/evaluate_all_models.py

# 3. Analyze results
# - Check: Results/comparison/metrics_comparison.csv
# - View: Results/comparison/scatter_comparison.png

# 4. Select best model (likely late fusion)
```

---

**You're ready to train and evaluate chicken age regression models!** 🚀

For general info, see [README.md](README.md) | For issues, open a GitHub issue

---

## Contact & Support

**Author:** Adam Kumar (is0699se@ed.ritsumei.ac.jp)
**Institution:** Visual Information Engineering Laboratory, Ritsumeikan University
**Supervisor:** Damon Chandler

For general info, see [README.md](README.md) | For issues, open a GitHub issue
