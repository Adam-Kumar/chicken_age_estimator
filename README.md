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
| Human (Pre-calibration) | 1.965 days | 2.487 days | - |
| Human (Post-calibration) | 2.295 days | 2.796 days | - |

**Findings:**
- Late fusion outperforms feature fusion on this small dataset (< 1000 samples) due to its simpler ensemble-like averaging strategy
- **All models significantly outperform humans (4-5× lower MAE)**, demonstrating that deep learning captures subtle visual patterns humans cannot
- Human performance **worsened after calibration** (p=0.0108), suggesting the task is difficult for humans to learn even with reference images

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
├── User_Study/                 # Human performance evaluation study
│   ├── analysis.py             # Statistical analysis script
│   ├── Chicken Decay Estimation Study.csv  # User study responses
│   ├── pre_survey_images.txt   # Ground truth for pre-calibration
│   ├── post_survey_images.txt  # Ground truth for post-calibration
│   └── Results/                # Analysis outputs
│       ├── 1_predictions_vs_groundtruth.png
│       ├── 2_model_vs_human_comparison.png
│       ├── 3_individual_participant_performance.png
│       └── participant_detailed_results.csv
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
- **Comparison plots:**
  - Side-by-side scatter plots for all 3 models
  - Bar charts with unified 0-1 day scale for easy comparison
  - Combined confusion matrices
- **Metrics table:** MAE, RMSE, parameters comparison
- **Predictions:** CSV files with individual predictions

## User Study: Model vs Human Performance

### Overview

To validate the model's performance, we conducted a user study comparing human estimation accuracy against the ResNet-50 model. The study involved 20 participants estimating chicken drumette age before and after calibration.

### Study Design

- **Participants:** 20 individuals
- **Task:** Estimate the age (1-7 days) of chicken drumettes from images
- **Structure:**
  - **Pre-calibration:** 10 images without any reference
  - **Calibration phase:** Participants shown reference images for each day
  - **Post-calibration:** 10 different images to test improvement
- **Views:** Both TOP and SIDE view images provided (matching model inputs)

### Key Findings

| Metric | Baseline | Feature Fusion | Late Fusion | Human (Pre-calib) | Human (Post-calib) |
|--------|----------|----------------|-------------|-------------------|-------------------|
| **MAE** | 0.490 days | 0.465 days | **0.430 days** | 1.965 days | 2.295 days |
| **RMSE** | 0.645 days | 0.569 days | **0.539 days** | 2.487 days | 2.796 days |
| **Exact Accuracy** | - | - | - | 18.5% | 12.0% |
| **Within ±1 day** | - | - | - | 46.0% | 41.0% |

**Key Findings:**

1. **All models significantly outperform humans:** Even the baseline single-view model (MAE: 0.490) performs 4× better than humans
2. **Late Fusion is best:** The multi-view Late Fusion model (MAE: 0.430) achieves the lowest error
3. **Calibration paradoxically hurt performance:** Human accuracy decreased after calibration (MAE: 1.965 → 2.295, p=0.0108)
   - This suggests reference images may have introduced anchoring bias or confusion
   - Humans may "overthink" after seeing examples rather than relying on intuition
   - The task is fundamentally difficult for humans to learn even with training
4. **Low inter-rater reliability:** ICC scores (0.20 pre, 0.11 post) indicate high disagreement between participants

### Inter-Rater Reliability

- **Pre-calibration ICC:** 0.20 (poor agreement)
- **Post-calibration ICC:** 0.11 (poor agreement)

Low ICC values indicate high variability between human raters, suggesting the task is challenging even with calibration.

### Running the Analysis

To reproduce the user study analysis:

```bash
python User_Study/analysis.py
```

This generates (saved to `User_Study/Results/`):
- **Statistical metrics:** MAE, RMSE, accuracy, inter-rater reliability printed to console
- **3 visualization plots:**
  - Predictions vs ground truth (pre & post calibration)
  - Model vs human comparison (all 3 architectures + human performance)
  - Individual participant performance
- **Detailed CSV:** Per-participant metrics and improvement scores

### Study Files

**Input files:**
- `Chicken Decay Estimation Study.csv` - Raw survey responses from 20 participants
- `pre_survey_images.txt` - Ground truth labels for pre-calibration images
- `post_survey_images.txt` - Ground truth labels for post-calibration images
- `analysis.py` - Statistical analysis script with full documentation

**Output files (Results/ folder):**
- `1_predictions_vs_groundtruth.png` - Scatter plots showing human prediction accuracy
- `2_model_vs_human_comparison.png` - Bar chart comparing all 3 models vs humans
- `3_individual_participant_performance.png` - Per-participant pre/post comparison
- `participant_detailed_results.csv` - Complete metrics for each participant

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
- `comparison/` - Model comparison visualizations:
  - `metrics_comparison.csv` - Performance metrics table
  - `metrics_comparison.png` - Bar charts (MAE & RMSE with unified 0-1 day scale)
  - `scatter_comparison.png` - Side-by-side scatter plots for all 3 models
  - `confusion_matrix_comparison.png` - Combined confusion matrices
- `predictions/` - Individual model prediction CSV files
- `plots/` - Individual evaluation plots

### User_Study/
- `analysis.py` - Statistical analysis script with comprehensive metrics
- `Chicken Decay Estimation Study.csv` - Survey responses from 20 participants
- `pre_survey_images.txt` - Ground truth for pre-calibration test images (10 images)
- `post_survey_images.txt` - Ground truth for post-calibration test images (10 images)
- `Results/` - Analysis outputs:
  - `1_predictions_vs_groundtruth.png` - Human prediction scatter plots
  - `2_model_vs_human_comparison.png` - Bar chart with all 3 models vs humans
  - `3_individual_participant_performance.png` - Per-participant pre/post comparison
  - `participant_detailed_results.csv` - Detailed metrics for all 20 participants

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
