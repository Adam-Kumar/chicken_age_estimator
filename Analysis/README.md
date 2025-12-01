# Feature Importance Analysis

**Location**: `Project/Analysis/`

This directory contains scripts for analyzing what visual features drive the ConvNeXt-T Feature Fusion model's age predictions for chicken drumettes.

## Overview

Understanding which features (color, texture, shape, spatial patterns) are most important for age estimation helps:
1. **Model interpretability**: Validate that the model learns meaningful aging indicators
2. **Trust and debugging**: Ensure predictions are based on relevant features, not artifacts
3. **Domain insights**: Identify which aging markers are most discriminative
4. **Model improvement**: Guide architecture and data collection decisions

## Analysis Approaches

### 1. Grad-CAM Visualization ✅ (Implemented)

**Script**: `gradcam_visualization.py`

**What it does**:
- Generates class activation maps showing which spatial regions drive predictions
- Visualizes both TOP and SIDE view encoder activations separately
- Creates heatmap overlays on original images
- Provides summary statistics across samples

**Limitations**:
- Shows **WHERE** the model looks, not **WHAT** features it uses
- Cannot distinguish between color, texture, and shape
- Highlights entire chicken region without feature-level granularity

**Usage**:
```bash
# Visualize 20 random test samples
python Analysis/gradcam_visualization.py --num_samples 20

# Visualize only day 5 samples
python Analysis/gradcam_visualization.py --day 5 --num_samples 10

# Visualize all test samples
python Analysis/gradcam_visualization.py --num_samples 200
```

**Output**: `Analysis/Results/gradcam/`
- Individual visualizations: `gradcam_sample_XXX_dayX_errorX.XX.png`
- Summary statistics: `gradcam_summary.csv`
- Summary plots: `gradcam_summary.png`

---

### 2. Input Ablation Studies (Planned)

**Approach**: Systematically remove or modify different feature types and measure prediction degradation.

#### a. Color Ablation
- **Grayscale conversion**: Remove all color information
- **Color normalization**: Equalize histograms to reduce color variation
- **Channel ablation**: Remove R/G/B channels individually
- **Expected**: If color is important, grayscale → large MAE increase

#### b. Texture Ablation
- **Gaussian blur**: Remove fine texture, preserve color
- **Edge-only**: Keep only edges, remove smooth regions
- **High-pass filter**: Keep only high-frequency texture
- **Expected**: If texture is important, blur → large MAE increase

#### c. Shape Ablation
- **Contour-only**: Keep silhouette, fill interior with mean color
- **Random crop**: Test spatial invariance
- **Distortion**: Test shape sensitivity
- **Expected**: If shape is important, contour-only maintains performance

**Implementation plan**:
```python
# Analysis/input_ablation.py
- evaluate_grayscale()     # Color importance
- evaluate_blurred()       # Texture importance
- evaluate_edges_only()    # Shape importance
- evaluate_combinations()  # Interaction effects
```

---

### 3. Color Statistics Analysis (Planned)

**Approach**: Analyze color distributions and their correlation with age predictions.

#### Analyses:
- **Mean RGB per day**: Plot average color progression across ages
- **Color histograms**: Compare distributions across age groups
- **HSV analysis**: Separate hue, saturation, brightness
- **Skin darkness metric**: Track progressive darkening with age
- **Correlation with predictions**: Compute color-MAE relationships

**Implementation plan**:
```python
# Analysis/color_statistics.py
- plot_rgb_progression()
- compute_darkness_metric()
- correlate_color_with_error()
```

---

### 4. Attention Rollout (Planned - if Transformer backbones)

**Approach**: For ViT or Swin Transformer models, trace attention flow through layers.

**Note**: ConvNeXt is CNN-based, so this is only applicable for ViT/Swin comparisons.

---

### 5. Feature Importance via Model Probing (Future)

**Approach**: Train linear probes on intermediate features to decode specific properties:
- Color probe: Can intermediate features predict mean RGB?
- Texture probe: Can features predict Gabor filter responses?
- Age probe: Which layer first encodes age information?

---

## Current Status

✅ **Completed**:
- Grad-CAM visualization for Feature Fusion model
- Infrastructure for spatial attention analysis
- Organized output structure

🔄 **In Progress**:
- None

📋 **Planned**:
- Input ablation studies (color, texture, shape)
- Color statistics analysis
- Quantitative feature importance metrics

---

## Directory Structure

```
Analysis/
├── gradcam_visualization.py    # Grad-CAM implementation
├── input_ablation.py           # Ablation studies (placeholder)
├── color_statistics.py         # Color analysis (placeholder)
├── README.md                   # This file
└── Results/                    # All analysis outputs
    ├── gradcam/                # Grad-CAM visualizations
    │   ├── gradcam_sample_XXX.png  # Individual heatmaps
    │   ├── gradcam_summary.csv     # Activation statistics
    │   └── gradcam_summary.png     # Summary plots
    ├── ablation/               # Ablation study results (future)
    │   ├── grayscale/
    │   ├── blurred/
    │   └── edges_only/
    └── color_analysis/         # Color statistics (future)
        ├── rgb_progression.png
        └── color_metrics.csv
```

---

## Model Architecture Notes

**Feature Fusion Model**:
- **Backbone**: ConvNeXt-Tiny (two independent encoders)
- **TOP encoder**: Processes dorsal view images
- **SIDE encoder**: Processes lateral view images
- **Fusion**: Concatenate features → MLP regressor
- **Target layer for Grad-CAM**: `encoder.stages[-1]` (last ConvNeXt stage)

**Why Feature Fusion?**:
- Best model in cross-validation (0.172 MAE)
- 2nd best on held-out test set (0.204 MAE)
- Provides insights into both view-specific feature importance

---

## Key Questions to Answer

1. **Spatial patterns**: Which regions (Grad-CAM) are most important?
   - Is the entire chicken important or specific areas (skin, bone, joints)?

2. **Color importance**: How much does color contribute?
   - Test: Grayscale vs RGB performance difference

3. **Texture importance**: Are fine-grained textures critical?
   - Test: Blurred vs original performance difference

4. **Shape importance**: Does body shape matter?
   - Test: Contour-only vs full image performance

5. **View comparison**: TOP vs SIDE feature differences?
   - Analyze Grad-CAM activation patterns per view
   - Compare ablation effects per view

6. **Age-specific patterns**: Do features change importance across ages?
   - Early days (1-3): What features distinguish?
   - Late days (5-7): What features distinguish?

---

## Usage Tips

1. **Start with Grad-CAM**: Quick visual understanding of spatial attention
2. **Follow with ablation**: Quantify feature type importance
3. **Dive into color stats**: If color is important, understand how it changes
4. **Cross-reference findings**: Combine qualitative (Grad-CAM) + quantitative (ablation)

---

## Dependencies

All scripts use the same dependencies as main project:
- PyTorch 2.0+
- OpenCV (for heatmap visualization)
- Matplotlib, Seaborn (for plotting)
- NumPy, Pandas (for analysis)
- PIL (for image loading)

---

## Author

Adam Kumar
Visual Information Engineering Laboratory
Ritsumeikan University

---

## Notes

- All analysis uses the trained ConvNeXt-T Feature Fusion model (`checkpoints/convnext_t_feature_holdout.pth`)
- Test set is held-out (13 chickens, 91 paired samples)
- Ensure model is trained before running analysis scripts
