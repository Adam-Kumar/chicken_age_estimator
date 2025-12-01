"""evaluate_custom.py
Evaluate a specific model configuration with detailed analysis.

Usage:
    python Scripts/evaluate_custom.py --backbone convnext_t --fusion feature
    python Scripts/evaluate_custom.py --backbone resnet50 --fusion late
    python Scripts/evaluate_custom.py --backbone efficientnet_b0 --fusion baseline

Arguments:
    --backbone: One of [efficientnet_b0, resnet18, resnet50, resnet101,
                        vit_b_16, swin_t, swin_b, convnext_t, convnext_b]
    --fusion: One of [baseline, late, feature]
    --checkpoint: Path to checkpoint (optional, uses pretrained if not provided)

Output: Results/other_models/{backbone}_{fusion}/
    - graphs/scatter_plot.png
    - graphs/confusion_matrix.png
    - graphs/error_distribution.png
    - graphs/per_day_performance.png
    - csv/predictions.csv
    - csv/metrics.csv
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, r2_score
from scipy import stats

from Model.model import ResNetRegressor, LateFusionRegressor, FeatureFusionRegressor
from Model.dataset import ChickenAgeDataset, ChickenAgePairedDataset, get_default_transforms
from torch.utils.data import DataLoader


# Available options
BACKBONES = [
    "efficientnet_b0",
    "resnet18",
    "resnet50",
    "resnet101",
    "vit_b_16",
    "swin_t",
    "swin_b",
    "convnext_t",
    "convnext_b",
]

FUSION_TYPES = ["baseline", "late", "feature"]


def load_model(backbone, fusion, checkpoint_path, device):
    """Load model with optional checkpoint."""
    print(f"\nLoading model: {backbone.upper()} + {fusion.upper()}")

    if fusion == "baseline":
        model = ResNetRegressor(backbone_name=backbone, pretrained=True).to(device)
    elif fusion == "late":
        model = LateFusionRegressor(backbone_name=backbone, pretrained=True).to(device)
    else:  # feature
        model = FeatureFusionRegressor(backbone_name=backbone, pretrained=True).to(device)

    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"Loading checkpoint from: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        if checkpoint_path:
            print(f"WARNING: Checkpoint not found at {checkpoint_path}")
        print("Using pretrained ImageNet weights only (not fine-tuned)")
        print("Results will reflect pretrained performance, not task-specific training\n")

    return model


def evaluate_on_test_set(model, test_loader, device, is_fusion):
    """Evaluate model and collect predictions."""
    model.eval()

    all_preds = []
    all_targets = []
    all_errors = []

    with torch.no_grad():
        for batch in test_loader:
            if is_fusion:
                (x_top, x_side), y = batch
                x_top, x_side, y = x_top.to(device), x_side.to(device), y.to(device)
                preds = model(x_top, x_side)
            else:
                x, y = batch
                x, y = x.to(device), y.to(device)
                preds = model(x)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
            all_errors.extend((preds - y).abs().cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_errors = np.array(all_errors)

    return all_preds, all_targets, all_errors


def plot_scatter(preds, targets, results_dir, backbone, fusion):
    """Plot predicted vs actual scatter plot."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # Scatter plot
    ax.scatter(targets, preds, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)

    # Perfect prediction line
    min_val, max_val = 0, 8
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

    # Calculate metrics
    mae = np.mean(np.abs(preds - targets))
    rmse = np.sqrt(np.mean((preds - targets)**2))
    r2 = r2_score(targets, preds)
    correlation, _ = stats.pearsonr(targets, preds)

    # Add metrics to plot
    metrics_text = f'MAE: {mae:.3f} days\nRMSE: {rmse:.3f} days\nR²: {r2:.3f}\nCorrelation: {correlation:.3f}'
    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
           fontsize=12, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel('Actual Age (days)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Predicted Age (days)', fontsize=14, fontweight='bold')
    ax.set_title(f'{backbone.upper()} {fusion.capitalize()}: Predicted vs Actual Age',
                fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)

    plt.tight_layout()
    plt.savefig(results_dir / "graphs" / "scatter_plot.png", dpi=150, bbox_inches='tight')
    print("  Saved: graphs/scatter_plot.png")
    plt.close()

    return mae, rmse, r2, correlation


def plot_confusion_matrix(preds, targets, results_dir, backbone, fusion):
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
    ax.set_title(f'{backbone.upper()} {fusion.capitalize()}: Confusion Matrix (Normalized)',
                fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.savefig(results_dir / "graphs" / "confusion_matrix.png", dpi=150, bbox_inches='tight')
    print("  Saved: graphs/confusion_matrix.png")
    plt.close()


def plot_error_distribution(errors, results_dir, backbone, fusion):
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

    plt.suptitle(f'{backbone.upper()} {fusion.capitalize()}: Error Analysis',
                fontweight='bold', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(results_dir / "graphs" / "error_distribution.png", dpi=150, bbox_inches='tight')
    print("  Saved: graphs/error_distribution.png")
    plt.close()


def plot_per_day_performance(preds, targets, errors, results_dir, backbone, fusion):
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

    plt.suptitle(f'{backbone.upper()} {fusion.capitalize()}: Per-Day Performance',
                fontweight='bold', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(results_dir / "graphs" / "per_day_performance.png", dpi=150, bbox_inches='tight')
    print("  Saved: graphs/per_day_performance.png")
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
    print("  Saved: csv/predictions.csv")

    # Print worst 10 predictions
    print("\n" + "="*80)
    print("WORST 10 PREDICTIONS")
    print("="*80)
    print(predictions_df[['relative_path', 'actual_age', 'predicted_age', 'absolute_error']].head(10).to_string(index=False))


def save_metrics(mae, rmse, r2, corr, results_dir, backbone, fusion):
    """Save metrics to CSV."""
    metrics_df = pd.DataFrame({
        'Backbone': [backbone],
        'Fusion': [fusion],
        'MAE': [mae],
        'RMSE': [rmse],
        'R2': [r2],
        'Correlation': [corr]
    })

    output_file = results_dir / "csv" / "metrics.csv"
    metrics_df.to_csv(output_file, index=False)
    print("  Saved: csv/metrics.csv")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a specific model configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python Scripts/evaluate_custom.py --backbone convnext_t --fusion feature
  python Scripts/evaluate_custom.py --backbone resnet50 --fusion late
  python Scripts/evaluate_custom.py --backbone efficientnet_b0 --fusion baseline --checkpoint checkpoints/my_model.pth
        """
    )

    parser.add_argument("--backbone", type=str, required=True, choices=BACKBONES,
                       help="Backbone architecture")
    parser.add_argument("--fusion", type=str, required=True, choices=FUSION_TYPES,
                       help="Fusion strategy")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to model checkpoint (optional)")

    args = parser.parse_args()

    print("="*80)
    print(f"EVALUATING: {args.backbone.upper()} + {args.fusion.upper()}")
    print("="*80)

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    model = load_model(args.backbone, args.fusion, args.checkpoint, device)

    # Load test data
    test_csv = project_root / "Labels" / "test.csv"
    root_dir = project_root / "Dataset_Processed"

    if args.fusion == "baseline":
        # Single-view dataset (TOP view only)
        test_df = pd.read_csv(test_csv)
        test_df = test_df[test_df['view'].str.upper() == 'TOP VIEW'].copy()
        test_ds = ChickenAgeDataset(test_csv, root_dir, get_default_transforms(False))
        is_fusion = False
    else:
        # Paired dataset (TOP+SIDE views)
        test_df = pd.read_csv(test_csv)
        test_df = test_df[test_df['view'].str.upper() == 'TOP VIEW'].copy()
        test_ds = ChickenAgePairedDataset(test_csv, root_dir, get_default_transforms(False))
        is_fusion = True

    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=0)

    print(f"Test samples: {len(test_ds)}\n")

    # Evaluate
    print("="*80)
    print("GENERATING PREDICTIONS")
    print("="*80 + "\n")

    preds, targets, errors = evaluate_on_test_set(model, test_loader, device, is_fusion)

    # Create results directory
    results_dir = project_root / "Results" / "other_models" / f"{args.backbone}_{args.fusion}"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "graphs").mkdir(exist_ok=True)
    (results_dir / "csv").mkdir(exist_ok=True)

    # Generate plots
    print("="*80)
    print("GENERATING ANALYSIS PLOTS")
    print("="*80 + "\n")

    mae, rmse, r2, corr = plot_scatter(preds, targets, results_dir, args.backbone, args.fusion)
    plot_confusion_matrix(preds, targets, results_dir, args.backbone, args.fusion)
    plot_error_distribution(errors, results_dir, args.backbone, args.fusion)
    plot_per_day_performance(preds, targets, errors, results_dir, args.backbone, args.fusion)

    # Save results
    save_predictions(preds, targets, errors, test_df, results_dir)
    save_metrics(mae, rmse, r2, corr, results_dir, args.backbone, args.fusion)

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
    print("  - csv/predictions.csv")
    print("  - csv/metrics.csv")


if __name__ == "__main__":
    main()
