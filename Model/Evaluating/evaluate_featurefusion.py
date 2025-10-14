"""evaluate_featurefusion.py
Evaluate the trained Feature Fusion model on the test set.

This script evaluates the feature fusion model (concatenating features from TOP and SIDE views) and generates:
- Test set MAE and RMSE metrics
- Scatter plot of predictions vs ground truth
- Confusion matrix showing prediction distribution
- Individual predictions saved to CSV

Usage:
    Evaluate with default parameters (test set):
        python Model/Evaluating/evaluate_featurefusion.py

    Evaluate on validation set:
        python Model/Evaluating/evaluate_featurefusion.py --split val

    Custom parameters:
        python Model/Evaluating/evaluate_featurefusion.py --split test --batch_size 32

Outputs:
    - Results/plots/eval_featurefusion_{split}_scatter.png - Scatter plot
    - Results/plots/eval_featurefusion_{split}_confusion_matrix.png - Confusion matrix
    - Results/predictions/predictions_feature_fusion_{split}.csv - All predictions
    - Console output with MAE and RMSE metrics

Requirements:
    - Model must be trained first: python Model/Training/train_featurefusion.py
    - Checkpoint must exist at: Model/checkpoints/best_feature_fusion.pth
    - Requires paired TOP and SIDE view images
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path as _Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Ensure project root on path
_project_root = _Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from Model.dataset import ChickenAgePairedDataset, get_default_transforms  # noqa: E402
from Model.model import FeatureFusionRegressor  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def mae(pred: np.ndarray, target: np.ndarray) -> float:
    """Calculate Mean Absolute Error."""
    return float(np.mean(np.abs(pred - target)))


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    """Calculate Root Mean Squared Error."""
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Feature Fusion model")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"],
                       help="Which split to evaluate (default: test)")
    parser.add_argument("--batch_size", type=int, default=64)
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 80)
    print("EVALUATING FEATURE FUSION MODEL (Feature Concatenation)")
    print("=" * 80)
    print(f"\nEvaluating on {args.split} set...")

    # Paths
    csv_path = _project_root / "Labels" / f"{args.split}.csv"
    checkpoint_path = _project_root / "Model" / "checkpoints" / "best_feature_fusion.pth"
    root_dir = _project_root / "Dataset_Processed"

    # Check checkpoint exists
    if not checkpoint_path.exists():
        print(f"\n[ERROR] Checkpoint not found: {checkpoint_path}")
        print("\nTrain the model first:")
        print("  python Model/Training/train_featurefusion.py")
        return 1

    # Load data
    dataset = ChickenAgePairedDataset(csv_path, root_dir, transforms=get_default_transforms(train=False))
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Load model
    model = FeatureFusionRegressor(pretrained=False)
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE)
    model.eval()

    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', 'N/A')}")
    print(f"Dataset size: {len(dataset)} samples (paired TOP+SIDE views)\n")

    # Evaluate
    preds = []
    gts = []

    with torch.no_grad():
        for (img_top, img_side), labels in loader:
            img_top, img_side = img_top.to(DEVICE), img_side.to(DEVICE)
            outputs = model(img_top, img_side)
            preds.extend(outputs.cpu().numpy().tolist())
            gts.extend(labels.numpy().tolist())

    preds_arr = np.array(preds)
    gts_arr = np.array(gts)

    # Calculate metrics
    mae_val = mae(preds_arr, gts_arr)
    rmse_val = rmse(preds_arr, gts_arr)

    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"MAE:  {mae_val:.4f} days")
    print(f"RMSE: {rmse_val:.4f} days")
    print(f"Best epoch: {ckpt.get('epoch', 'N/A')}")

    # Save predictions
    output_dir = _project_root / "Results" / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_output = output_dir / f"predictions_feature_fusion_{args.split}.csv"

    with open(csv_output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "ground_truth", "prediction", "error"])
        for i, (gt, pred) in enumerate(zip(gts, preds)):
            error = abs(pred - gt)
            writer.writerow([i, gt, pred, error])

    print(f"\nPredictions saved to: {csv_output}")

    # Generate scatter plot
    plot_dir = _project_root / "Results" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    scatter_path = plot_dir / f"eval_featurefusion_{args.split}_scatter.png"

    plt.figure(figsize=(7, 7))
    plt.scatter(gts_arr, preds_arr, alpha=0.6, s=30, color='#2ca02c')
    plt.plot([1, 7], [1, 7], "r--", linewidth=2, label="Perfect prediction")
    plt.xlabel("Ground Truth (day)", fontsize=12)
    plt.ylabel("Prediction (day)", fontsize=12)
    plt.title(f"Feature Fusion Model - {args.split.capitalize()} Set\nMAE: {mae_val:.3f}, RMSE: {rmse_val:.3f}", fontsize=13)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(0.5, 7.5)
    plt.ylim(0.5, 7.5)
    plt.tight_layout()
    plt.savefig(scatter_path, dpi=150)
    print(f"Scatter plot saved to: {scatter_path}")

    # Generate confusion matrix
    # Round predictions to nearest integer day (1-7)
    preds_rounded = np.clip(np.round(preds_arr), 1, 7).astype(int)
    gts_rounded = np.round(gts_arr).astype(int)

    cm = confusion_matrix(gts_rounded, preds_rounded, labels=[1, 2, 3, 4, 5, 6, 7])

    plt.figure(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=[1, 2, 3, 4, 5, 6, 7],
                yticklabels=[1, 2, 3, 4, 5, 6, 7],
                cbar_kws={'label': 'Count'})
    plt.xlabel("Predicted Day", fontsize=12)
    plt.ylabel("Ground Truth Day", fontsize=12)
    plt.title(f"Feature Fusion Model - Confusion Matrix ({args.split.capitalize()} Set)\nMAE: {mae_val:.3f}", fontsize=13)
    plt.tight_layout()

    cm_path = plot_dir / f"eval_featurefusion_{args.split}_confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    print(f"Confusion matrix saved to: {cm_path}")

    print("\n" + "=" * 80)
    print("[SUCCESS] Evaluation complete!")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
