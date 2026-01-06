"""gradcam_visualization.py

Grad-CAM visualization for ConvNeXt-B Feature Fusion model (Best Model).
Generates class activation maps to understand which image regions
contribute most to age predictions.

For Feature Fusion models, visualizes both:
- TOP view encoder activation maps
- SIDE view encoder activation maps

This helps identify spatial features (color patterns, texture regions,
shapes) that drive predictions, though it cannot distinguish between
color vs texture vs shape without additional ablation studies.

Input: Trained ConvNeXt-B Feature Fusion model (0.67 MAE, 3-fold CV)
Output:
  - Individual heatmaps for TOP and SIDE views
  - Overlay visualizations
  - Side-by-side comparisons
  - Analysis by age group

Usage:
  python Analysis/gradcam_visualization.py --num_samples 20
  python Analysis/gradcam_visualization.py --day 5 --num_samples 10
  python Analysis/gradcam_visualization.py --fold 0  # Use specific fold checkpoint
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import argparse

from Model.model import FeatureFusionRegressor
from Model.dataset import get_default_transforms
import torchvision.transforms as transforms


class GradCAM:
    """Grad-CAM implementation for ConvNeXt architectures."""

    def __init__(self, model, target_layer):
        """
        Args:
            model: The neural network model
            target_layer: The convolutional layer to visualize (typically last conv layer)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        """Hook to capture forward pass activations."""
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        """Hook to capture backward pass gradients."""
        self.gradients = grad_output[0].detach()

    def generate_cam(self, input_tensor, target_output=None):
        """
        Generate class activation map.

        Args:
            input_tensor: Input image tensor [1, C, H, W]
            target_output: If None, uses model's prediction

        Returns:
            cam: Class activation map [H, W]
        """
        # Forward pass
        self.model.eval()
        output = self.forward_pass(input_tensor)

        # Backward pass
        self.model.zero_grad()
        if target_output is None:
            target_output = output
        target_output.backward()

        # Generate CAM
        gradients = self.gradients  # [1, C, H, W]
        activations = self.activations  # [1, C, H, W]

        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # [1, C, 1, 1]

        # Weighted combination of activation maps
        cam = torch.sum(weights * activations, dim=1, keepdim=True)  # [1, 1, H, W]
        cam = F.relu(cam)  # ReLU to keep only positive influences

        # Normalize
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam

    def forward_pass(self, input_tensor):
        """Override in subclass for specific model architecture."""
        raise NotImplementedError


class GradCAMFeatureFusion(GradCAM):
    """Grad-CAM for Feature Fusion models with separate TOP and SIDE encoders."""

    def __init__(self, model, view='top'):
        """
        Args:
            model: FeatureFusionRegressor model
            view: 'top' or 'side' - which encoder to visualize
        """
        self.view = view.lower()

        # Get the appropriate encoder
        if self.view == 'top':
            encoder = model.backbone_top
        elif self.view == 'side':
            encoder = model.backbone_side
        else:
            raise ValueError("view must be 'top' or 'side'")

        # For ConvNeXt wrapped in ConvNeXtFeatureExtractor:
        # encoder.features contains the Sequential of ConvNeXt stages
        # Hook into the last layer that outputs spatial feature maps (before avgpool)
        target_layer = encoder.features

        super().__init__(model, target_layer)
        self.encoder = encoder

    def generate_cam(self, input_top, input_side, target_output=None):
        """
        Generate class activation map for Feature Fusion model.

        Args:
            input_top: TOP view image [1, 3, H, W]
            input_side: SIDE view image [1, 3, H, W]
            target_output: If None, uses model's prediction

        Returns:
            cam: Class activation map [H, W]
        """
        # Forward pass
        self.model.eval()
        output = self.forward_pass(input_top, input_side)

        # Backward pass
        self.model.zero_grad()
        if target_output is None:
            target_output = output
        target_output.backward()

        # Generate CAM
        gradients = self.gradients  # [1, C, H, W]
        activations = self.activations  # [1, C, H, W]

        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # [1, C, 1, 1]

        # Weighted combination of activation maps
        cam = torch.sum(weights * activations, dim=1, keepdim=True)  # [1, 1, H, W]
        cam = F.relu(cam)  # ReLU to keep only positive influences

        # Normalize
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam

    def forward_pass(self, input_top, input_side):
        """
        Forward pass through Feature Fusion model.

        Args:
            input_top: TOP view image [1, 3, H, W]
            input_side: SIDE view image [1, 3, H, W]

        Returns:
            output: Model prediction
        """
        return self.model(input_top, input_side)


def load_model(device, fold=0):
    """
    Load the trained ConvNeXt-B Feature Fusion model (Best Model).

    Args:
        device: torch device
        fold: Which fold checkpoint to load (0, 1, or 2). Default: 0 (best fold)

    Returns:
        model: Loaded FeatureFusionRegressor model
    """
    model = FeatureFusionRegressor(backbone_name="convnext_b", pretrained=True).to(device)

    checkpoint_path = project_root / "checkpoints" / f"convnext_b_feature_fold{fold}.pth"

    if checkpoint_path.exists():
        print(f"Loading checkpoint: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Model loaded successfully (Fold {fold}, 3-fold CV)\n")
    else:
        print(f"WARNING: Checkpoint not found at {checkpoint_path}")
        print("Available checkpoints: convnext_b_feature_fold{0,1,2}.pth")
        print("Using pretrained ImageNet weights only (not fine-tuned)")
        print("Results will not be meaningful. Train the model first.\n")

    model.eval()
    return model


def load_image_pair(row, root_dir, transform):
    """
    Load TOP and SIDE view images for a given sample.

    Args:
        row: DataFrame row with 'relative_path' and 'day'
        root_dir: Root directory for Dataset_Processed
        transform: Image transformation pipeline

    Returns:
        img_top_tensor: Transformed TOP view [1, 3, 224, 224]
        img_side_tensor: Transformed SIDE view [1, 3, 224, 224]
        img_top_orig: Original TOP view (PIL Image)
        img_side_orig: Original SIDE view (PIL Image)
    """
    # Get TOP view path
    top_path = root_dir / row['relative_path']

    # Construct SIDE view path
    side_path_str = str(row['relative_path']).replace('TOP VIEW', 'SIDE VIEW')
    side_path = root_dir / side_path_str

    # Load images
    img_top_orig = Image.open(top_path).convert('RGB')
    img_side_orig = Image.open(side_path).convert('RGB')

    # Transform for model input
    img_top_tensor = transform(img_top_orig).unsqueeze(0)
    img_side_tensor = transform(img_side_orig).unsqueeze(0)

    return img_top_tensor, img_side_tensor, img_top_orig, img_side_orig


def apply_colormap_overlay(image, cam, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """
    Apply CAM heatmap overlay on original image.

    Args:
        image: PIL Image or numpy array (RGB)
        cam: Class activation map [H, W], values in [0, 1]
        alpha: Overlay transparency (0=only image, 1=only heatmap)
        colormap: OpenCV colormap

    Returns:
        overlay: RGB image with heatmap overlay
    """
    # Convert PIL to numpy if needed
    if isinstance(image, Image.Image):
        image = np.array(image)

    # Resize CAM to match image size
    h, w = image.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))

    # Apply colormap
    cam_colored = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), colormap)
    cam_colored = cv2.cvtColor(cam_colored, cv2.COLOR_BGR2RGB)

    # Blend
    overlay = (alpha * cam_colored + (1 - alpha) * image).astype(np.uint8)

    return overlay


def visualize_sample(model, row, root_dir, transform, device, output_dir, sample_idx):
    """
    Generate and save Grad-CAM visualizations for a single sample.

    Args:
        model: Trained FeatureFusionRegressor
        row: DataFrame row
        root_dir: Dataset root directory
        transform: Image transformation
        device: torch device
        output_dir: Directory to save results
        sample_idx: Sample index for filename
    """
    # Load images
    img_top_tensor, img_side_tensor, img_top_orig, img_side_orig = load_image_pair(
        row, root_dir, transform
    )

    img_top_tensor = img_top_tensor.to(device)
    img_side_tensor = img_side_tensor.to(device)

    # Get prediction
    with torch.no_grad():
        pred_age = model(img_top_tensor, img_side_tensor).item()

    actual_age = row['day']
    error = abs(pred_age - actual_age)

    # Generate Grad-CAM for TOP view
    gradcam_top = GradCAMFeatureFusion(model, view='top')
    cam_top = gradcam_top.generate_cam(img_top_tensor, img_side_tensor)

    # Generate Grad-CAM for SIDE view
    gradcam_side = GradCAMFeatureFusion(model, view='side')
    cam_side = gradcam_side.generate_cam(img_top_tensor, img_side_tensor)

    # Create overlays
    overlay_top = apply_colormap_overlay(img_top_orig, cam_top, alpha=0.4)
    overlay_side = apply_colormap_overlay(img_side_orig, cam_side, alpha=0.4)

    # Create visualization
    fig = plt.figure(figsize=(16, 8))

    # TOP view - original
    ax1 = plt.subplot(2, 4, 1)
    ax1.imshow(img_top_orig)
    ax1.set_title('TOP View - Original', fontsize=12, fontweight='bold')
    ax1.axis('off')

    # TOP view - heatmap only
    ax2 = plt.subplot(2, 4, 2)
    im = ax2.imshow(cam_top, cmap='jet', vmin=0, vmax=1)
    ax2.set_title('TOP View - Activation Map', fontsize=12, fontweight='bold')
    ax2.axis('off')
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

    # TOP view - overlay
    ax3 = plt.subplot(2, 4, 3)
    ax3.imshow(overlay_top)
    ax3.set_title('TOP View - Overlay', fontsize=12, fontweight='bold')
    ax3.axis('off')

    # TOP view - high activation regions (threshold)
    ax4 = plt.subplot(2, 4, 4)
    cam_top_binary = (cam_top > 0.7).astype(float)  # Threshold at 70%
    ax4.imshow(img_top_orig)
    ax4.imshow(cam_top_binary, cmap='Reds', alpha=0.4, vmin=0, vmax=1)
    ax4.set_title('TOP View - High Activation (>70%)', fontsize=12, fontweight='bold')
    ax4.axis('off')

    # SIDE view - original
    ax5 = plt.subplot(2, 4, 5)
    ax5.imshow(img_side_orig)
    ax5.set_title('SIDE View - Original', fontsize=12, fontweight='bold')
    ax5.axis('off')

    # SIDE view - heatmap only
    ax6 = plt.subplot(2, 4, 6)
    im = ax6.imshow(cam_side, cmap='jet', vmin=0, vmax=1)
    ax6.set_title('SIDE View - Activation Map', fontsize=12, fontweight='bold')
    ax6.axis('off')
    plt.colorbar(im, ax=ax6, fraction=0.046, pad=0.04)

    # SIDE view - overlay
    ax7 = plt.subplot(2, 4, 7)
    ax7.imshow(overlay_side)
    ax7.set_title('SIDE View - Overlay', fontsize=12, fontweight='bold')
    ax7.axis('off')

    # SIDE view - high activation regions
    ax8 = plt.subplot(2, 4, 8)
    cam_side_binary = (cam_side > 0.7).astype(float)
    ax8.imshow(img_side_orig)
    ax8.imshow(cam_side_binary, cmap='Reds', alpha=0.4, vmin=0, vmax=1)
    ax8.set_title('SIDE View - High Activation (>70%)', fontsize=12, fontweight='bold')
    ax8.axis('off')

    # Add prediction info
    info_text = f'Actual Age: {actual_age:.1f} days | Predicted Age: {pred_age:.3f} days | Error: {error:.3f} days'
    fig.suptitle(info_text, fontsize=14, fontweight='bold', y=0.98)

    # Save
    filename = f"gradcam_sample_{sample_idx:03d}_day{int(actual_age)}_error{error:.2f}.png"
    output_path = output_dir / filename
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return {
        'sample_idx': sample_idx,
        'actual_age': actual_age,
        'predicted_age': pred_age,
        'error': error,
        'filename': filename,
        'top_activation_mean': cam_top.mean(),
        'side_activation_mean': cam_side.mean(),
        'top_activation_max': cam_top.max(),
        'side_activation_max': cam_side.max()
    }


def generate_summary_statistics(results_list, output_dir):
    """Generate summary statistics and visualizations across all samples."""
    df = pd.DataFrame(results_list)

    # Save summary CSV
    df.to_csv(output_dir / "gradcam_summary.csv", index=False)

    # Create summary plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Mean activation by view
    ax1 = axes[0, 0]
    x = np.arange(len(df))
    width = 0.35
    ax1.bar(x - width/2, df['top_activation_mean'], width, label='TOP View', alpha=0.7)
    ax1.bar(x + width/2, df['side_activation_mean'], width, label='SIDE View', alpha=0.7)
    ax1.set_xlabel('Sample Index', fontsize=10, fontweight='bold')
    ax1.set_ylabel('Mean Activation', fontsize=10, fontweight='bold')
    ax1.set_title('Mean Activation by View', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # 2. Activation vs prediction error
    ax2 = axes[0, 1]
    ax2.scatter(df['top_activation_mean'], df['error'], alpha=0.6, label='TOP View', s=50)
    ax2.scatter(df['side_activation_mean'], df['error'], alpha=0.6, label='SIDE View', s=50)
    ax2.set_xlabel('Mean Activation', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Prediction Error (days)', fontsize=10, fontweight='bold')
    ax2.set_title('Activation vs Prediction Error', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Activation by actual age
    ax3 = axes[1, 0]
    age_groups = df.groupby('actual_age').agg({
        'top_activation_mean': 'mean',
        'side_activation_mean': 'mean'
    }).reset_index()
    ax3.plot(age_groups['actual_age'], age_groups['top_activation_mean'],
            'o-', linewidth=2, markersize=8, label='TOP View')
    ax3.plot(age_groups['actual_age'], age_groups['side_activation_mean'],
            's-', linewidth=2, markersize=8, label='SIDE View')
    ax3.set_xlabel('Actual Age (days)', fontsize=10, fontweight='bold')
    ax3.set_ylabel('Mean Activation', fontsize=10, fontweight='bold')
    ax3.set_title('Mean Activation by Age', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Statistics summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    stats_text = f"""Summary Statistics (n={len(df)}):

TOP View Activation:
  Mean: {df['top_activation_mean'].mean():.4f}
  Std:  {df['top_activation_mean'].std():.4f}
  Max:  {df['top_activation_max'].mean():.4f}

SIDE View Activation:
  Mean: {df['side_activation_mean'].mean():.4f}
  Std:  {df['side_activation_mean'].std():.4f}
  Max:  {df['side_activation_max'].mean():.4f}

Prediction Performance:
  Mean Error: {df['error'].mean():.4f} days
  Std Error:  {df['error'].std():.4f} days
"""
    ax4.text(0.1, 0.5, stats_text, transform=ax4.transAxes,
            fontsize=11, verticalalignment='center', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_dir / "gradcam_summary.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nSummary statistics saved to: {output_dir / 'gradcam_summary.csv'}")
    print(f"Summary visualization saved to: {output_dir / 'gradcam_summary.png'}")


def main():
    parser = argparse.ArgumentParser(description='Generate Grad-CAM visualizations for ConvNeXt-B Feature Fusion (Best Model)')
    parser.add_argument('--num_samples', type=int, default=20,
                       help='Number of test samples to visualize (default: 20)')
    parser.add_argument('--day', type=int, default=None, choices=[1,2,3,4,5,6,7],
                       help='Only visualize samples from specific day (default: all days)')
    parser.add_argument('--fold', type=int, default=0, choices=[0,1,2],
                       help='Which fold checkpoint to use (default: 0 - best fold)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for sample selection (default: 42)')

    args = parser.parse_args()

    print("="*80)
    print("GRAD-CAM VISUALIZATION: ConvNeXt-B Feature Fusion (Best Model)")
    print("="*80)
    print()

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Load model
    model = load_model(device, fold=args.fold)

    # Load test data
    test_csv = project_root / "Labels" / "test.csv"
    root_dir = project_root / "Dataset_Processed"

    df = pd.read_csv(test_csv)
    df = df[df['view'].str.upper() == 'TOP VIEW']  # Only need TOP view rows (pairs are implicit)

    # Filter by day if specified
    if args.day is not None:
        df = df[df['day'] == args.day]
        print(f"Filtering to day {args.day} only")

    # Sample random subset
    np.random.seed(args.seed)
    if len(df) > args.num_samples:
        df = df.sample(n=args.num_samples, random_state=args.seed)

    print(f"Visualizing {len(df)} samples\n")

    # Create output directory
    output_dir = project_root / "Analysis" / "Results" / f"gradcam_fold{args.fold}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get transforms (evaluation mode, no augmentation)
    transform = get_default_transforms(train=False)

    # Process samples
    print("="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80 + "\n")

    results_list = []
    for idx, (_, row) in enumerate(df.iterrows(), 1):
        print(f"Processing sample {idx}/{len(df)}: Day {int(row['day'])}, {row['relative_path']}")

        try:
            result = visualize_sample(
                model, row, root_dir, transform, device, output_dir, idx
            )
            results_list.append(result)
            print(f"  Predicted: {result['predicted_age']:.3f}, Error: {result['error']:.3f}")
            print(f"  TOP activation: {result['top_activation_mean']:.4f}, SIDE activation: {result['side_activation_mean']:.4f}")
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    # Generate summary
    if results_list:
        print("\n" + "="*80)
        print("GENERATING SUMMARY STATISTICS")
        print("="*80 + "\n")
        generate_summary_statistics(results_list, output_dir)

    # Final summary
    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE!")
    print("="*80)
    print(f"\nModel: ConvNeXt-B Feature Fusion (Fold {args.fold})")
    print(f"Generated {len(results_list)} visualizations")
    print(f"Results saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - gradcam_sample_XXX_dayX_errorX.XX.png (individual visualizations)")
    print("  - gradcam_summary.csv (activation statistics)")
    print("  - gradcam_summary.png (summary plots)")
    print("\nInterpretation:")
    print("  - Heatmap colors: Red = high activation, Blue = low activation")
    print("  - High activation regions indicate areas most important for prediction")
    print("  - Compare TOP vs SIDE to see which view contributes more")
    print("  - Note: Grad-CAM shows WHERE the model looks, not WHAT features (color/texture/shape)")
    print("\nTo use different fold: python Analysis/gradcam_visualization.py --fold 1")


if __name__ == "__main__":
    main()
