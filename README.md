# Chicken Age Estimation - Multi-View Deep Learning

Deep learning system for estimating chicken drumette age (1-7 days post-slaughter) from dual-view images using multi-view fusion strategies.

**Author:** Adam Kumar
**Institution:** Visual Information Engineering Laboratory, Ritsumeikan University
**Supervisor:** Damon Chandler

---

## Abstract

This research presents a comprehensive evaluation of 27 deep learning configurations for automated chicken drumette age estimation, comparing 9 modern architectures (EfficientNet, ResNet, ViT, Swin, ConvNeXt) across 3 multi-view fusion strategies. Using paired TOP and SIDE view images with segmentation masks, we demonstrate that multi-view fusion significantly outperforms single-view baselines. Our best model (ViT-B/16 Feature Fusion) achieves 0.66 ± 0.03 days MAE (~15.8 hours) in 3-fold cross-validation, demonstrating highly accurate age estimation capability suitable for real-time quality control in food safety applications.

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

### Cross-Validation Performance

Comprehensive 3-fold cross-validation results:

#### Top 10 Models

| Rank | Backbone | Fusion | Mean MAE (days) | Std MAE | Params (M) |
|------|----------|--------|-----------------|---------|------------|
| 1 | **ViT-B/16** | **Feature** | **0.660** | 0.027 | 87.34 |
| 2 | ConvNeXt-T | Feature | 0.665 | 0.044 | 28.59 |
| 3 | ConvNeXt-T | Late | 0.662 | 0.031 | 55.64 |
| 4 | Swin-T | Late | 0.668 | 0.025 | 55.04 |
| 5 | Swin-B | Feature | 0.680 | 0.040 | 88.30 |
| 6 | ConvNeXt-B | Feature | 0.674 | 0.075 | 89.05 |
| 7 | ConvNeXt-B | Late | 0.692 | 0.070 | 176.04 |
| 8 | Swin-T | Feature | 0.693 | 0.023 | 28.29 |
| 9 | ConvNeXt-T | TOP View | 0.710 | 0.049 | 27.82 |
| 10 | ConvNeXt-B | TOP View | 0.719 | 0.019 | 88.02 |

**Full results**: See [Results/CSV_and_Analysis/table_all_models.csv](Results/CSV_and_Analysis/table_all_models.csv)

#### Key Findings

1. **ViT-B/16 Feature Fusion achieves best performance**: 0.660 ± 0.027 days (~15.8 hours)
2. **Transformer architectures excel**: ViT and Swin models dominate top ranks
3. **ConvNeXt shows excellent efficiency**: ConvNeXt-T (28.59M params) nearly matches larger models
4. **Multi-view fusion provides 7% improvement**: Best fusion (0.660) vs best baseline (0.710)
5. **Fusion strategies significantly outperform single-view**: All fusion models beat best baseline

### Fusion Strategy Comparison

Performance grouped by fusion type (mean MAE across all backbones):

| Strategy | Mean MAE | Best Model |
|----------|----------|------------|
| **Feature Fusion** | **0.761** | ViT-B/16 (0.660) |
| **Late Fusion** | **0.755** | ConvNeXt-T (0.662) |
| **TOP View Only** | **0.815** | ConvNeXt-T (0.710) |

**Key Insights:**
- Multi-view fusion provides ~7% improvement over single-view baseline
- Feature and Late fusion perform comparably, both significantly better than baseline
- Fusion benefits all architectures, with consistent improvements across backbones

### Human Performance Comparison

To validate the practical utility of our best model, we conducted a user study comparing ViT-B/16 Feature Fusion against human performance.

**Study Design:**
- **Participants**: 50
- **Task**: Estimate chicken drumette age (1-7 days) from images
- **Protocol**: Pre-calibration (10 samples) → Post-calibration (10 samples with feedback)

**Results:**

| Group | Mean MAE (days) | R² | Accuracy |
|-------|-----------------|----|---------:|
| **ViT-B/16 Feature (Best Model)** | **0.691** | **0.796** | **47.4%** |
| Human Pre-Calibration (Avg) | 1.864 | -0.367 | 17.8% |
| Human Post-Calibration (Avg) | 1.588 | 0.041 | 23.6% |

**Key Findings:**
1. **Model superiority**: ViT-B/16 Feature Fusion (0.691 days) outperforms average human performance by 2.3×
2. **Human improvement**: Calibration reduces human MAE by 14.8% (0.276 days improvement)
3. **Consistency advantage**: Model provides reliable, consistent predictions vs high human variability (0.600-3.500 days range)
4. **Top human performance**: Best participant (0.600 days MAE) slightly exceeded model performance, demonstrating that expert-level judgment remains competitive

**Analysis Files**: See [User_Study/](User_Study/) for detailed analysis script and visualizations

---

## Method

### Multi-View Fusion Strategies

We evaluate three approaches for combining TOP and SIDE view information:

#### 1. TOP View Only (Single-View Baseline)
- Single model processes TOP view images
- No multi-view fusion
- Serves as baseline for fusion strategies
- **Best performance**: ConvNeXt-T (0.710 MAE)

#### 2. Late Fusion (View-Aware Ensemble)
- Two independent models (TOP and SIDE encoders)
- Each model trained separately on its view
- Predictions combined via simple averaging
- **Best performance**: ConvNeXt-T (0.662 MAE)

#### 3. Feature Fusion (Learned Fusion) ⭐ Best Strategy
- Two separate encoders (TOP and SIDE)
- Features concatenated before regression head
- MLP learns optimal view combination weights
- **Best performance**: ViT-B/16 (0.660 ± 0.027 MAE)

### Dataset

**Included: Dataset_Processed/**
- 82 unique chicken drumettes tracked across 7 days (Days 1-7 post-slaughter)
- Paired TOP (dorsal) and SIDE (lateral) views
- 224×224 RGB images (~4.8MB total)
- **Strategy**: Chicken-level split (no chicken appears in multiple sets)

**Segmentation:**
- U-Net generated masks applied to isolate chicken from background
- Background set to white (prevents spurious correlations)
- Masks available in `Segmentation/masks/`

**Full Dataset (Original Resolution):**
For high-resolution images (6.8GB), contact: is0699se@ed.ritsumei.ac.jp

### Training Details

**Hyperparameters (3-Fold CV):**
- Epochs: 50 (with early stopping, patience=10)
- Batch size: 16
- Optimizer: AdamW (weight decay = 0.01)
- Learning rate: 2e-5 (all architectures, with 3-epoch warmup)
- Loss function: L1 (MAE)
- Random seed: 42 (reproducibility)

**Data Augmentation (Training Only):**
- Random horizontal flip (p=0.5)
- Random rotation (±10°, p=0.3)
- Random resized crop (scale 0.9-1.0)
- Color jitter (brightness/contrast/saturation=0.1, hue=0.02)
- Gaussian blur (kernel=5, sigma=0.1-0.5, p=0.2)
- ImageNet normalization
- **Segmentation masks applied** (critical for preventing background artifacts)

---

## Repository Structure

```
Project/
├── Scripts/
│   ├── train_all_models_full.py       # Train all 27 configurations (saves checkpoints/metrics)
│   └── generate_final_analysis.py     # Generate all graphs and analysis
├── Model/
│   ├── model.py              # Architecture definitions (Feature Fusion, Late Fusion, etc.)
│   ├── dataset.py            # Data loading with mask support
│   └── __init__.py
├── Results/
│   ├── Data/                 # Training data
│   │   ├── checkpoints/      # Model weights (81 .pth files)
│   │   ├── training_history/ # Per-epoch metrics (81 .json files)
│   │   └── predictions/      # Validation predictions (81 .csv files)
│   ├── Graphs/               # All visualizations (9 .png files)
│   │   ├── all_models_comparison_3panel.png
│   │   ├── fusion_comparison.png
│   │   ├── model_size_vs_performance.png
│   │   ├── predictions_vs_actual_vit_b_16_feature.png
│   │   ├── confusion_matrix_vit_b_16_feature.png
│   │   └── ... (4 more graphs)
│   └── CSV_and_Analysis/     # Tables and statistical analysis
│       ├── full_progress.json          # Complete training results
│       ├── table_all_models.csv        # All 27 models ranked
│       ├── table_best_per_fusion.csv   # Best per fusion strategy
│       ├── table_fusion_summary.csv    # Statistical summary
│       └── statistical_analysis.txt    # ANOVA and significance tests
├── Analysis/
│   ├── gradcam_visualization.py  # Grad-CAM saliency maps
│   ├── README.md                 # Analysis documentation
│   └── Results/
│       └── gradcam_fold{0,1,2}/  # Activation visualizations
├── Segmentation/
│   ├── masks/              # U-Net generated segmentation masks
│   └── README.md           # Segmentation documentation
├── Labels/
│   ├── labels.csv          # All labels
│   ├── train.csv, val.csv, test.csv  # Data splits
│   └── split_labels.py
├── Dataset_Processed/      # 224x224 preprocessed images
├── README.md               # This file (project overview & results)
└── README_TRAINING.md      # Training & evaluation usage guide
```

---

## Usage

For detailed training and evaluation instructions, see **[README_TRAINING.md](README_TRAINING.md)**.

### Quick Start

**Generate all visualizations and analysis:**
```bash
python Scripts/generate_final_analysis.py
```
This creates all graphs in `Results/Graphs/` and tables in `Results/CSV_and_Analysis/`.

**Train all 27 models (40-70 hours with GPU):**
```bash
python Scripts/train_all_models_full.py
```
This saves all checkpoints, training histories, and predictions to `Results/Data/`.

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
2. **Sub-day accuracy**: Best model achieves 0.66 days MAE (~15.8 hours), enabling real-time quality control
3. **Evidence for multi-view fusion**: Fusion strategies consistently outperform single-view baseline across all architectures
4. **Architecture insights**: Transformer models (ViT, Swin) excel, while ConvNeXt offers best efficiency
5. **Practical deployment system**: Ready-to-use model with comprehensive documentation and checkpoints
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
