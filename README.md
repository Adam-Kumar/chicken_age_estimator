# Chicken Drumette Age Regression

Deep learning models for predicting chicken drumette age (1-7 days) from multi-view images using PyTorch.

## Overview

This project implements and compares three deep learning architectures for chicken age regression:

1. **Baseline (ResNetRegressor)** - Single-view ResNet50 model
2. **Late Fusion** - Averages predictions from TOP and SIDE views ⭐ **Best Performance**
3. **Feature Fusion** - Concatenates features from both views before prediction

### Key Results

| Model | Test MAE | Test RMSE | Parameters |
|-------|----------|-----------|------------|
| **Late Fusion** ⭐ | **0.430 days** | **0.539 days** | 47.0M |
| Feature Fusion | 0.465 days | 0.569 days | 49.1M |
| Baseline | 0.490 days | 0.645 days | 23.5M |

**Finding:** Late fusion outperforms feature fusion on this small dataset (< 1000 samples) due to its simpler ensemble-like averaging strategy.

## Repository Structure

```
.
├── Dataset_Processed/          # Preprocessed images (224x224)
├── Labels/                     # Train/val/test CSV files
├── Model/                      # Model architectures and training code
│   ├── Training/               # Training scripts
│   ├── Evaluating/             # Evaluation scripts
│   └── checkpoints/            # Trained model checkpoints (not included)
├── Results/                    # Example outputs and visualizations
├── README.md                   # This file
├── README_TRAINING.md          # Detailed training guide
└── requirements.txt            # Python dependencies
```

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/Adam-Kumar/chicken_age_estimator.git
cd chicken_age_estimator

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Train Models

Train all three models:
```bash
python Model/Training/train_all_models.py
```

Or train individual models:
```bash
python Model/Training/train_baseline.py
python Model/Training/train_latefusion.py
python Model/Training/train_featurefusion.py
```

### 3. Evaluate and Compare

```bash
python Model/Evaluating/evaluate_all_models.py
```

Results will be saved to `Results/comparison/`.

## Dataset

### Included: Dataset_Processed
- **Size:** ~4.8MB
- **Format:** 224x224 RGB images
- **Structure:** TOP VIEW and SIDE VIEW folders organized by day (1-7)
- **Splits:** 799 train, 169 validation, 183 test samples
- **Content:** 82 unique chicken pieces, tracked across 7 days

### Full Dataset (Dataset_Original)

The preprocessed dataset included in this repository is sufficient for training and reproducing the results. However, if you need access to the **original high-resolution images** (6.8GB), please contact me:

**Email:** is0699se@ed.ritsumei.ac.jp

Please include your name, institution, and intended research purpose.

### Dataset Details
- **Views:** Paired TOP and SIDE view images
- **Days:** 1-7 (age progression)
- **Preprocessing:** Resized, normalized for ResNet50
- **Augmentation:** Random crop, flip, rotation, color jitter (training only)

## Model Architectures

### Baseline (Single-View)
```
Input (224x224x3) → ResNet50 → FC → Age prediction
```
- Fastest training (~20-40 min)
- Uses individual images independently

### Late Fusion ⭐ Best Model
```
TOP image → ResNet50 → pred_top ──┐
                                   ├─→ average → final prediction
SIDE image → ResNet50 → pred_side ─┘
```
- Best for small datasets
- Simple ensemble strategy
- Test MAE: 0.430 days

### Feature Fusion
```
TOP image → ResNet50 → features_top ──┐
                                       ├─→ concatenate → FC(512) → prediction
SIDE image → ResNet50 → features_side ─┘
```
- Learns cross-view interactions
- Better for large datasets (>5000 samples)

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (recommended)
- 8GB+ RAM
- ~5GB disk space (for dataset + checkpoints)

## Training Configuration

Default parameters (optimized for this dataset):
- **Epochs:** 30
- **Batch size:** 32
- **Learning rate:** 1e-4
- **Weight decay:** 1e-2 (L2 regularization)
- **Optimizer:** AdamW
- **Scheduler:** CosineAnnealingLR

## Results

Example outputs in `Results/`:
- **Training curves:** Loss/MAE progression over epochs
- **Comparison plots:** Side-by-side scatter plots
- **Metrics table:** MAE, RMSE, parameters comparison
- **Predictions:** CSV files with individual predictions

## Documentation

- **[README_TRAINING.md](README_TRAINING.md)** - Comprehensive training and evaluation guide
- **Code comments** - Detailed docstrings in all modules

## Project Structure Details

### Model/
- `model.py` - Architecture definitions
- `dataset.py` - PyTorch Dataset and DataLoader
- `train_core.py` - Core training loop
- `Training/` - Training scripts for each model
- `Evaluating/` - Evaluation scripts for each model

### Labels/
- `train.csv`, `val.csv`, `test.csv` - Data splits
- `generate_labels.py` - Script to create labels from images
- `split_labels.py` - Script to split dataset

### Results/
- `training_curves/` - Training progress plots
- `comparison/` - Model comparison visualizations
- `plots/` - Individual evaluation plots

## Hardware Recommendations

**Minimum:**
- CPU: 4 cores
- RAM: 8GB
- GPU: Not required but very slow without

**Recommended:**
- CPU: 8+ cores
- RAM: 16GB+
- GPU: NVIDIA GPU with 6GB+ VRAM (e.g., RTX 3060)
- Training time: ~1.5-3 hours for all models

## Troubleshooting

**CUDA Out of Memory:**
```bash
python Model/Training/train_baseline.py --batch_size 16
```

**Import Errors:**
```bash
cd /path/to/project/root
python Model/Training/train_baseline.py
```

**Missing Dependencies:**
```bash
pip install -r requirements.txt
```

See [README_TRAINING.md](README_TRAINING.md) for detailed troubleshooting.

## Author

**Adam Kumar**
Information System Science and Engineering
Visual Information Engineering Laboratory
Ritsumeikan University

**Supervisor:** Damon Chandler
**Contact:** is0699se@ed.ritsumei.ac.jp

## Acknowledgments

- **Supervisor:** Damon Chandler, Visual Information Engineering Laboratory, Ritsumeikan University
- ResNet50 backbone pretrained on ImageNet (PyTorch)
- Dataset collected at Ritsumeikan University for chicken age estimation research

## Contact

**For questions about the code or dataset:**
- Open an issue on GitHub
- Email: is0699se@ed.ritsumei.ac.jp

**Institution:**
Visual Information Engineering Laboratory
Information System Science and Engineering
Ritsumeikan University
Japan

---

**Project Status:** Active research project. Paper in preparation for publication.
