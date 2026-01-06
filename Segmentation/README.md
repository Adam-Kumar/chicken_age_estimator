# Chicken Segmentation - U-Net Training Workflow

**Goal**: Create binary masks to isolate chicken from plate/background using a trained U-Net model.

This addresses the spurious correlation issue discovered via Grad-CAM analysis.

**Dataset**: Uses 224x224 processed images (Dataset_Processed) for fast annotation and training.

---

## Workflow

### Step 1: Annotate Training Data (10-20 minutes)

```bash
python Segmentation/annotate_masks.py --num_images 30
```

**What it does:**
- Interactive tool to create ground truth masks using GrabCut
- Click foreground points (chicken) and background points (plate)
- Adjust erosion to fine-tune mask boundaries
- Manual painting mode for refinement if needed
- Fast processing on 224x224 images

**Controls:**
- **'a'**: AUTO mode - generate mask with default rectangle (NO POINTS NEEDED)
- **Left-click**: Foreground point (chicken) - GREEN
- **Right-click**: Background point (plate) - RED
- **'s'**: Preview mask with GrabCut (manual mode)
- **'e'**: Manual painting mode
- **'+/-'**: Adjust erosion
- **'r'**: Reset points
- **'q'**: Quit

**Tips:**
- **Fastest**: Press 'a' for automatic mask generation (try this first!)
- **Manual**: Add 2-3 foreground points on chicken + 1-2 background points on plate, then press 's'
- Use '+/-' to adjust erosion (removes plate edges)
- Use 'e' for manual touch-ups if needed

**Recommended**: 30+ images for good performance, minimum 20

---

### Step 2: Train U-Net Model (5-15 minutes)

```bash
python Segmentation/train_unet.py
```

**What it does:**
- Trains lightweight U-Net on 224x224 annotated masks
- Uses data augmentation (rotation, flip, brightness)
- 80/20 train/validation split
- Saves best model to `checkpoints/unet_best.pth`
- Generates training curves plot

**Requirements:**
- At least 20 annotated masks (30 recommended)
- GPU recommended but not required
- PyTorch installed

**Time**:
- GPU: ~5-10 minutes
- CPU: ~10-15 minutes

---

### Step 3: Generate All Masks (2-10 minutes)

```bash
python Segmentation/generate_masks_unet.py
```

**What it does:**
- Loads trained U-Net model
- Segments all 1148 images (224x224)
- Saves masks to `Segmentation/masks/`
- Preserves directory structure

**Time**:
- GPU: ~2-3 minutes
- CPU: ~5-10 minutes

---

## Output

Masks saved to: `Segmentation/masks/`

**Directory structure:**
```
Segmentation/masks/
├── TOP VIEW/
│   ├── Day 1/*.jpg
│   ├── Day 2/*.jpg
│   └── ...
└── SIDE VIEW/
    ├── Day 1/*.jpg
    └── ...
```

**Mask properties:**
- Same filename as original images
- Resolution: 224×224 (matches Dataset_Processed)
- Binary: 255 = chicken, 0 = background
- JPEG format

---

## Next Steps After Segmentation

1. **Update dataset loaders** (`Model/dataset.py`):
   - Load mask alongside image
   - Apply mask during data loading
   - Option 1: Multiply RGB by mask
   - Option 2: Set background to white

2. **Retrain models** with masked data:
   ```bash
   python Scripts/train/train_convnext_t_feature_fusion.py
   ```

3. **Evaluate and compare**:
   ```bash
   python Scripts/evaluate/evaluate_convnext_t_feature_fusion.py
   ```

4. **Expected improvement**:
   - Model focuses on chicken features (not plate/background)
   - Better generalization
   - Lower MAE on test set
   - Grad-CAM should show focus on chicken

---

## File Overview

### Scripts
- **annotate_masks.py** - Interactive annotation tool
- **train_unet.py** - Train U-Net model
- **generate_masks_unet.py** - Batch generate all masks

### Outputs
- **training_masks/** - Manually annotated ground truth (30 images)
- **checkpoints/** - Trained U-Net model
  - `unet_best.pth` - Best model (lowest validation loss)
  - `unet_final.pth` - Final model after all epochs
  - `training_curves.png` - Loss curves
- **masks/** - Final masks for all 1148 images

---

## Troubleshooting

**Issue**: "No training masks found!"
**Fix**: Run annotation step first: `python Segmentation/annotate_masks.py --num_images 30`

**Issue**: "Only X annotated masks found! Recommended: 30+"
**Fix**: Annotate more images for better performance

**Issue**: Annotation tool window not responding
**Fix**: Check OpenCV installation: `pip install opencv-python`

**Issue**: U-Net training fails (CUDA out of memory)
**Fix**: Reduce batch size in train_unet.py (change `batch_size=4` to `batch_size=2`)

**Issue**: Generated masks have poor quality
**Fix**:
- Annotate more training images (50+)
- Ensure training annotations are high quality
- Retrain with more epochs

**Issue**: PyTorch not installed
**Fix**: `pip install torch torchvision`

---

## U-Net Architecture

**Model**: Lightweight U-Net with skip connections
**Parameters**: ~31 million
**Input**: 224×224 RGB image
**Output**: 224×224 binary mask

**Architecture:**
- Encoder: 4 levels (64→128→256→512 channels)
- Bottleneck: 1024 channels
- Decoder: 4 levels with skip connections
- Total layers: 23 convolutional layers

**Loss function**: BCE + Dice Loss (combined)
**Optimizer**: Adam (lr=1e-4)
**Training**: 50 epochs with early stopping

---

## Tips for Good Annotations

1. **Be consistent**: Annotate chicken boundaries similarly across all images
2. **Click strategically**:
   - Foreground points: Center of chicken body
   - Background points: Far from chicken, on plate/lightbox
3. **Don't over-annotate**: 2-4 points per type is usually enough
4. **Review before saving**: Press 's' to preview, 'r' if not satisfied
5. **Diverse samples**: Annotate different days, views, chicken sizes

---

## Author

Adam Kumar
Visual Information Engineering Laboratory
Ritsumeikan University
