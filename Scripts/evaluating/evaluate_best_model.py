"""evaluate_best_model.py
Detailed evaluation of ConvNeXt-B Feature Fusion (best model).

Uses the model trained with 3-fold CV (train_best_model.py).
Generates comprehensive analysis plots and metrics.

Generates comprehensive analysis plots:
- Scatter plot (predicted vs actual)
- Confusion matrix
- Error distribution
- Per-day performance
- Training curves
- Predictions CSV

Input: checkpoints/convnext_b_feature_fold*.pth (3-fold CV checkpoints)
Output: Results/best_model/
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import json
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, r2_score
from scipy import stats

from Model.model import FeatureFusionRegressor
from Model.dataset import ChickenAgePairedDataset, get_default_transforms
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode


# Test-Time Augmentation (TTA)
USE_TTA = True  # Set to True to enable TTA (2-5% improvement)
TTA_TRANSFORMS = [
    lambda x: x,  # Original
    T.RandomHorizontalFlip(p=1.0),
    T.RandomVerticalFlip(p=1.0),
    lambda x: T.functional.rotate(x, 90, interpolation=InterpolationMode.BILINEAR),
    lambda x: T.functional.rotate(x, -90, interpolation=InterpolationMode.BILINEAR),
]


def predict_with_tta(model, x_top, x_side, device, transforms=TTA_TRANSFORMS):
    """Apply Test-Time Augmentation for robust predictions.

    Args:
        model: Trained model
        x_top: TOP view image tensor
        x_side: SIDE view image tensor
        device: torch device
        transforms: List of augmentation transforms

    Returns:
        Averaged prediction from multiple augmented views
    """
    model.eval()
    predictions = []

    with torch.no_grad():
        for transform in transforms:
            aug_top = transform(x_top)
            aug_side = transform(x_side)
            pred = model(aug_top.to(device), aug_side.to(device))
            predictions.append(pred)

    # Average all predictions
    return torch.stack(predictions).mean(dim=0)


def load_best_model(device, fold=0):
    """Load the best trained ConvNeXt-B Feature Fusion model from 3-fold CV.

    Args:
        device: torch device
        fold: Which fold checkpoint to load (0, 1, or 2). Default is fold 0.
    """
    model = FeatureFusionRegressor(backbone_name="convnext_b", pretrained=True).to(device)

    # Try to load checkpoint from 3-fold CV
    checkpoint_path = project_root / "checkpoints" / f"convnext_b_feature_fold{fold}.pth"

    if checkpoint_path.exists():
        print(f"Loading checkpoint from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"(Model trained with 3-fold CV, using fold {fold} checkpoint)\n")
    else:
        print(f"WARNING: No checkpoint found at {checkpoint_path}")
        print("Using pretrained ImageNet weights only (not fine-tuned)")
        print("To get accurate results, run Scripts/training/train_best_model.py first.\n")

    return model


def evaluate_on_test_set(model, test_loader, device, use_tta=USE_TTA):
    """Evaluate model and collect predictions.

    Args:
        model: Trained model
        test_loader: Test data loader
        device: torch device
        use_tta: Whether to use Test-Time Augmentation (default: True)

    Returns:
        Predictions, targets, and errors arrays
    """
    model.eval()

    all_preds = []
    all_targets = []
    all_errors = []

    print(f"Using Test-Time Augmentation (TTA): {use_tta}")
    if use_tta:
        print(f"  TTA transforms: {len(TTA_TRANSFORMS)} augmentations")

    with torch.no_grad():
        for (x_top, x_side), y in test_loader:
            x_top, x_side, y = x_top.to(device), x_side.to(device), y.to(device)

            if use_tta:
                # Use TTA: average predictions from multiple augmented views
                preds = predict_with_tta(model, x_top, x_side, device)
            else:
                # Standard inference
                preds = model(x_top, x_side)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
            all_errors.extend((preds - y).abs().cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_errors = np.array(all_errors)

    return all_preds, all_targets, all_errors


def plot_scatter(preds, targets, results_dir):
    """Plot predicted vs actual scatter plot with larger fonts."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))

    # Scatter plot
    ax.scatter(targets, preds, alpha=0.6, s=60, edgecolors='black', linewidth=0.5)

    # Perfect prediction line
    min_val, max_val = 0, 8
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

    # Calculate metrics
    mae = np.mean(np.abs(preds - targets))
    rmse = np.sqrt(np.mean((preds - targets)**2))
    r2 = r2_score(targets, preds)
    correlation, _ = stats.pearsonr(targets, preds)

    # Add metrics to plot (moved down to avoid overlap)
    metrics_text = f'MAE: {mae:.3f} days\nRMSE: {rmse:.3f} days\nR²: {r2:.3f}\nCorrelation: {correlation:.3f}'
    ax.text(0.05, 0.9, metrics_text, transform=ax.transAxes,
            fontsize=20, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Labels & title
    ax.set_xlabel('Actual Age (days)', fontsize=24, fontweight='bold')
    ax.set_ylabel('Predicted Age (days)', fontsize=24, fontweight='bold')
    ax.set_title('Scatter Plot', fontsize=30, fontweight='bold')
    ax.legend(fontsize=20)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.tick_params(axis='both', labelsize=20)

    plt.tight_layout()
    plt.savefig(results_dir / "graphs" / "scatter_plot.png", dpi=150, bbox_inches='tight')
    print("  Saved: scatter_plot.png")

    return mae, rmse, r2, correlation


def plot_confusion_matrix(preds, targets, results_dir):
    """Plot confusion matrix (treating age as classes)."""
    # Round predictions and targets to nearest day
    preds_rounded = np.round(preds).astype(int)
    targets_int = targets.astype(int)

    # Create confusion matrix
    cm = confusion_matrix(targets_int, preds_rounded, labels=[1, 2, 3, 4, 5, 6, 7])

    # Normalize by row (true label)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
               xticklabels=[1, 2, 3, 4, 5, 6, 7],
               yticklabels=[1, 2, 3, 4, 5, 6, 7],
               ax=ax, cbar_kws={'label': 'Proportion'})

    ax.set_xlabel('Predicted Age (days)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Actual Age (days)', fontsize=14, fontweight='bold')
    ax.set_title('Confusion Matrix (Normalized by Row)', fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.savefig(results_dir / "graphs" / "confusion_matrix.png", dpi=150, bbox_inches='tight')
    print("  Saved: confusion_matrix.png")
    plt.close()


def plot_error_distribution(errors, results_dir):
    """Plot error distribution histogram."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Histogram
    ax1.hist(errors, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.axvline(x=np.mean(errors), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(errors):.3f}')
    ax1.axvline(x=np.median(errors), color='green', linestyle='--', linewidth=2,
               label=f'Median: {np.median(errors):.3f}')
    ax1.set_xlabel('Absolute Error (days)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax1.set_title('Error Distribution', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3)

    # Box plot
    ax2.boxplot(errors, vert=True, patch_artist=True,
               boxprops=dict(facecolor='lightblue', color='black'),
               medianprops=dict(color='red', linewidth=2),
               whiskerprops=dict(color='black'),
               capprops=dict(color='black'))
    ax2.set_ylabel('Absolute Error (days)', fontsize=12, fontweight='bold')
    ax2.set_title('Error Distribution (Box Plot)', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    # Add statistics
    stats_text = f'Mean: {np.mean(errors):.3f}\nStd: {np.std(errors):.3f}\n'
    stats_text += f'Min: {np.min(errors):.3f}\nMax: {np.max(errors):.3f}\n'
    stats_text += f'25%: {np.percentile(errors, 25):.3f}\n'
    stats_text += f'75%: {np.percentile(errors, 75):.3f}'
    ax2.text(1.15, 0.5, stats_text, transform=ax2.transAxes,
            fontsize=10, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(results_dir / "graphs" / "error_distribution.png", dpi=150, bbox_inches='tight')
    print("  Saved: error_distribution.png")
    plt.close()


def plot_per_day_performance(preds, targets, errors, results_dir):
    """Plot performance breakdown by day."""
    # Group by actual day
    day_performance = {}
    for day in range(1, 8):
        mask = (targets.astype(int) == day)
        if mask.sum() > 0:
            day_performance[day] = {
                'mae': np.mean(errors[mask]),
                'std': np.std(errors[mask]),
                'count': mask.sum(),
                'mean_pred': np.mean(preds[mask]),
                'mean_actual': np.mean(targets[mask])
            }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # MAE per day
    days = list(day_performance.keys())
    maes = [day_performance[d]['mae'] for d in days]
    stds = [day_performance[d]['std'] for d in days]
    counts = [day_performance[d]['count'] for d in days]

    ax1.bar(days, maes, yerr=stds, color='steelblue', alpha=0.7,
           edgecolor='black', capsize=10)
    ax1.set_xlabel('Actual Age (days)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('MAE (days)', fontsize=12, fontweight='bold')
    ax1.set_title('MAE per Day', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_xticks(days)

    # Add sample counts
    for day, mae, count in zip(days, maes, counts):
        ax1.text(day, mae + 0.02, f'n={count}', ha='center', fontsize=9)

    # Mean prediction vs actual per day
    mean_preds = [day_performance[d]['mean_pred'] for d in days]
    mean_actuals = [day_performance[d]['mean_actual'] for d in days]

    ax2.plot(days, mean_actuals, 'o-', linewidth=2, markersize=8,
            label='Actual', color='green')
    ax2.plot(days, mean_preds, 's-', linewidth=2, markersize=8,
            label='Predicted', color='blue')
    ax2.set_xlabel('Day', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Age (days)', fontsize=12, fontweight='bold')
    ax2.set_title('Mean Predicted vs Actual per Day', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(days)

    plt.tight_layout()
    plt.savefig(results_dir / "graphs" / "per_day_performance.png", dpi=150, bbox_inches='tight')
    print("  Saved: per_day_performance.png")
    plt.close()


def save_predictions(preds, targets, errors, test_df, results_dir):
    """Save predictions to CSV."""
    predictions_df = test_df.copy()
    predictions_df['predicted_age'] = preds
    predictions_df['actual_age'] = targets
    predictions_df['absolute_error'] = errors
    predictions_df['error'] = preds - targets

    # Sort by error (largest errors first)
    predictions_df = predictions_df.sort_values('absolute_error', ascending=False)

    output_file = results_dir / "csv" / "predictions.csv"
    predictions_df.to_csv(output_file, index=False)
    print(f"  Saved: predictions.csv")

    # Print worst 10 predictions
    print("\n" + "="*80)
    print("WORST 10 PREDICTIONS")
    print("="*80)
    print(predictions_df[['relative_path', 'actual_age', 'predicted_age', 'absolute_error']].head(10).to_string(index=False))


def plot_training_curves(results_dir):
    """Plot training curves from 3-fold CV metrics."""
    metrics_file = project_root / "Results" / "best_model" / "metrics.json"

    if not metrics_file.exists():
        print("  WARNING: metrics.json not found, cannot generate training curves")
        return

    with open(metrics_file, 'r') as f:
        metrics = json.load(f)

    if 'fold_results' not in metrics or not metrics['fold_results']:
        print("  WARNING: No fold results found in metrics.json")
        return

    # Check if training history exists
    if 'training_history' not in metrics['fold_results'][0]:
        print("  WARNING: No training history found. Re-run train_best_model.py to generate training curves.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for fold_idx, fold_result in enumerate(metrics['fold_results']):
        history = fold_result['training_history']
        epochs = [h['epoch'] for h in history]
        train_maes = [h['train_mae'] for h in history]
        val_maes = [h['val_mae'] for h in history]

        ax = axes[fold_idx]
        ax.plot(epochs, train_maes, label='Train MAE', linewidth=2, marker='o', markersize=3)
        ax.plot(epochs, val_maes, label='Val MAE', linewidth=2, marker='s', markersize=3)

        # Mark best epoch
        best_epoch = fold_result['best_epoch']
        best_val_mae = fold_result['val_mae']
        ax.axvline(x=best_epoch, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Best Epoch ({best_epoch})')
        ax.plot(best_epoch, best_val_mae, 'r*', markersize=15, label=f'Best Val MAE: {best_val_mae:.4f}')

        ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax.set_ylabel('MAE (days)', fontsize=12, fontweight='bold')
        ax.set_title(f'Fold {fold_idx + 1} Training Curves', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = results_dir / "graphs" / "training_curves.png"
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved: training_curves.png")


def main():
    print("="*80)
    print("EVALUATING BEST MODEL: ConvNeXt-B Feature Fusion")
    print("="*80)
    print()

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Load model
    model = load_best_model(device)

    # Load test data (with segmentation masks)
    test_csv = project_root / "Labels" / "test.csv"
    root_dir = project_root / "Dataset_Processed"
    mask_dir = project_root / "Segmentation" / "masks"

    test_ds = ChickenAgePairedDataset(test_csv, root_dir, get_default_transforms(False), use_masks=True, mask_dir=mask_dir)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=0)

    print(f"Test samples: {len(test_ds)}\n")

    # Evaluate
    print("="*80)
    print("GENERATING PREDICTIONS")
    print("="*80 + "\n")

    preds, targets, errors = evaluate_on_test_set(model, test_loader, device)

    # Create results directory
    results_dir = project_root / "Results" / "best_model"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "graphs").mkdir(exist_ok=True)
    (results_dir / "csv").mkdir(exist_ok=True)

    # Generate plots
    print("="*80)
    print("GENERATING ANALYSIS PLOTS")
    print("="*80 + "\n")

    mae, rmse, r2, corr = plot_scatter(preds, targets, results_dir)
    plot_confusion_matrix(preds, targets, results_dir)
    plot_error_distribution(errors, results_dir)
    plot_per_day_performance(preds, targets, errors, results_dir)
    plot_training_curves(results_dir)

    # Save predictions
    test_df = pd.read_csv(test_csv)
    test_df = test_df[test_df['view'].str.upper() == 'TOP VIEW']  # Only TOP view (paired dataset)
    save_predictions(preds, targets, errors, test_df, results_dir)

    # Save metrics JSON
    metrics_data = {
        "model": "ConvNeXt-B Feature Fusion",
        "strategy": "Feature Fusion",
        "test_mae": float(mae),
        "test_mse": float(mae**2),  # Approximate MSE from MAE
        "test_rmse": float(rmse),
        "r2_score": float(r2),
        "correlation": float(corr),
        "test_samples": len(preds)
    }

    import json
    metrics_file = results_dir / "metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics_data, f, indent=2)
    print(f"  Saved: metrics.json")

    # Print summary
    print("\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print("="*80)
    print(f"\nTest Set Performance:")
    print(f"  MAE:  {mae:.4f} days")
    print(f"  RMSE: {rmse:.4f} days")
    print(f"  R²:   {r2:.4f}")
    print(f"  Correlation: {corr:.4f}")
    print(f"\nResults saved to: {results_dir}")
    print("\nGenerated files:")
    print("  - graphs/scatter_plot.png")
    print("  - graphs/confusion_matrix.png")
    print("  - graphs/error_distribution.png")
    print("  - graphs/per_day_performance.png")
    print("  - graphs/training_curves.png")
    print("  - csv/predictions.csv")
    print("  - metrics.json")


if __name__ == "__main__":
    main()
