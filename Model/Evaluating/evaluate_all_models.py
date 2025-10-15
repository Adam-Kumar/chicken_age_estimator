"""evaluate_all_models.py
Comprehensive evaluation and comparison of all 3 trained models.

This script:
1. Loads all 3 trained models (baseline, late_fusion, feature_fusion)
2. Evaluates each on the test set
3. Generates comparison visualizations:
   - Metrics comparison table (MAE, RMSE, params)
   - Side-by-side scatter plots
   - Combined confusion matrices (3 models side-by-side)
   - Bar chart comparing metrics
4. Saves results to Results/comparison/ and Results/predictions/

Usage:
    python Model/Evaluating/evaluate_all_models.py

Outputs:
    Results/comparison/:
        - metrics_comparison.csv - Performance metrics table
        - scatter_comparison.png - Side-by-side scatter plots
        - confusion_matrix_comparison.png - Combined confusion matrices
        - metrics_comparison.png - Bar charts (MAE, RMSE)
    Results/predictions/:
        - predictions_baseline_test.csv
        - predictions_late_fusion_test.csv
        - predictions_feature_fusion_test.csv
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path as _Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Ensure project root on path
_project_root = _Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from Model.dataset import ChickenAgeDataset, get_default_transforms, create_dataloaders_fusion  # noqa: E402
from Model.model import build_resnet_regressor, LateFusionRegressor, FeatureFusionRegressor  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = _project_root / "Model" / "checkpoints"
OUTPUT_DIR = _project_root / "Results" / "comparison"
TEST_CSV = _project_root / "Labels" / "test.csv"
ROOT_DIR = _project_root / "Dataset_Processed"

MODEL_CONFIGS = {
    "baseline": {
        "name": "Baseline (ResNet50)",
        "checkpoint": "best_baseline.pth",
        "color": "#1f77b4",
        "needs_fusion_data": False,
    },
    "late_fusion": {
        "name": "Late Fusion",
        "checkpoint": "best_late_fusion.pth",
        "color": "#ff7f0e",
        "needs_fusion_data": True,
    },
    "feature_fusion": {
        "name": "Feature Fusion",
        "checkpoint": "best_feature_fusion.pth",
        "color": "#2ca02c",
        "needs_fusion_data": True,
    },
}


def mae(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - target)))


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_model(model_type: str, checkpoint_path: _Path) -> nn.Module:
    """Load a trained model from checkpoint."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if model_type == "baseline":
        model = build_resnet_regressor(pretrained=False)
    elif model_type == "late_fusion":
        model = LateFusionRegressor(pretrained=False)
    elif model_type == "feature_fusion":
        model = FeatureFusionRegressor(pretrained=False)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE)
    model.eval()

    return model


def evaluate_model(model: nn.Module, loader, is_fusion: bool) -> Tuple[List[float], List[float]]:
    """Evaluate model and return predictions and ground truths."""
    preds: List[float] = []
    gts: List[float] = []

    with torch.no_grad():
        for batch in loader:
            if is_fusion:
                (img_top, img_side), labels = batch
                img_top, img_side = img_top.to(DEVICE), img_side.to(DEVICE)
                outputs = model(img_top, img_side)
            else:
                imgs, labels = batch
                imgs = imgs.to(DEVICE)
                outputs = model(imgs)

            preds.extend(outputs.cpu().numpy().tolist())
            gts.extend(labels.numpy().tolist())

    return preds, gts


def create_comparison_table(results: Dict) -> pd.DataFrame:
    """Create comparison table with metrics."""
    data = []
    for model_type, res in results.items():
        data.append({
            "Model": MODEL_CONFIGS[model_type]["name"],
            "MAE": res["mae"],
            "RMSE": res["rmse"],
            "Parameters (M)": res["params"] / 1e6,
            "Best Epoch": res["best_epoch"],
        })

    df = pd.DataFrame(data)
    return df


def plot_scatter_comparison(results: Dict, output_path: _Path):
    """Create side-by-side scatter plots for all models."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, (model_type, res) in enumerate(results.items()):
        ax = axes[idx]
        gts = np.array(res["ground_truth"])
        preds = np.array(res["predictions"])

        ax.scatter(gts, preds, alpha=0.6, color=MODEL_CONFIGS[model_type]["color"], s=20)
        ax.plot([1, 7], [1, 7], "r--", linewidth=2, label="Perfect prediction")
        ax.set_xlabel("Ground Truth (day)", fontsize=11)
        ax.set_ylabel("Prediction (day)", fontsize=11)
        ax.set_title(f"{MODEL_CONFIGS[model_type]['name']}\nMAE: {res['mae']:.3f}, RMSE: {res['rmse']:.3f}", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xlim(0.5, 7.5)
        ax.set_ylim(0.5, 7.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Scatter comparison saved to {output_path}")
    plt.close()


def plot_metrics_comparison(results: Dict, output_path: _Path):
    """Create bar chart comparing MAE and RMSE."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    models = [MODEL_CONFIGS[mt]["name"] for mt in results.keys()]
    maes = [results[mt]["mae"] for mt in results.keys()]
    rmses = [results[mt]["rmse"] for mt in results.keys()]
    colors = [MODEL_CONFIGS[mt]["color"] for mt in results.keys()]

    # MAE comparison
    ax1.bar(models, maes, color=colors, alpha=0.7)
    ax1.set_ylabel("MAE (days)", fontsize=12)
    ax1.set_title("Mean Absolute Error Comparison", fontsize=13, fontweight="bold")
    ax1.grid(True, axis="y", alpha=0.3)
    for i, v in enumerate(maes):
        ax1.text(i, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontweight="bold")

    # RMSE comparison
    ax2.bar(models, rmses, color=colors, alpha=0.7)
    ax2.set_ylabel("RMSE (days)", fontsize=12)
    ax2.set_title("Root Mean Squared Error Comparison", fontsize=13, fontweight="bold")
    ax2.grid(True, axis="y", alpha=0.3)
    for i, v in enumerate(rmses):
        ax2.text(i, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Metrics comparison saved to {output_path}")
    plt.close()


def plot_confusion_matrices(results: Dict, output_path: _Path):
    """Create combined confusion matrix visualization for all models."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    cmaps = {"baseline": "Blues", "late_fusion": "Oranges", "feature_fusion": "Greens"}

    for idx, (model_type, res) in enumerate(results.items()):
        ax = axes[idx]

        # Round predictions to nearest integer day (1-7)
        preds_arr = np.array(res["predictions"])
        gts_arr = np.array(res["ground_truth"])
        preds_rounded = np.clip(np.round(preds_arr), 1, 7).astype(int)
        gts_rounded = np.round(gts_arr).astype(int)

        # Create confusion matrix
        cm = confusion_matrix(gts_rounded, preds_rounded, labels=[1, 2, 3, 4, 5, 6, 7])

        # Plot heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap=cmaps[model_type],
                    xticklabels=[1, 2, 3, 4, 5, 6, 7],
                    yticklabels=[1, 2, 3, 4, 5, 6, 7],
                    cbar_kws={'label': 'Count'},
                    ax=ax)
        ax.set_xlabel("Predicted Day", fontsize=11)
        ax.set_ylabel("Ground Truth Day", fontsize=11)
        ax.set_title(f"{MODEL_CONFIGS[model_type]['name']}\nMAE: {res['mae']:.3f}", fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Combined confusion matrix saved to {output_path}")
    plt.close()


def save_predictions(results: Dict, output_dir: _Path):
    """Save predictions from all models to CSV."""
    pred_dir = _project_root / "Results" / "predictions"
    pred_dir.mkdir(exist_ok=True, parents=True)

    for model_type, res in results.items():
        csv_path = pred_dir / f"predictions_{model_type}_test.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["index", "ground_truth", "prediction", "error"])
            for i, (gt, pred) in enumerate(zip(res["ground_truth"], res["predictions"])):
                error = abs(pred - gt)
                writer.writerow([i, gt, pred, error])
        print(f"Predictions saved to {csv_path}")


def main():
    print("=" * 80)
    print("EVALUATING ALL MODELS")
    print("=" * 80)

    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    # Load data
    print("\nLoading test datasets...")
    test_dataset_single = ChickenAgeDataset(TEST_CSV, ROOT_DIR, transforms=get_default_transforms(train=False))
    test_loader_single = torch.utils.data.DataLoader(test_dataset_single, batch_size=64, shuffle=False, num_workers=2)

    from Model.dataset import ChickenAgePairedDataset  # noqa: E402
    test_dataset_fusion = ChickenAgePairedDataset(TEST_CSV, ROOT_DIR, transforms=get_default_transforms(train=False))
    test_loader_fusion = torch.utils.data.DataLoader(test_dataset_fusion, batch_size=64, shuffle=False, num_workers=2)

    results = {}

    # Evaluate each model
    for model_type, config in MODEL_CONFIGS.items():
        print(f"\n{'='*80}")
        print(f"Evaluating: {config['name']}")
        print(f"{'='*80}")

        checkpoint_path = CHECKPOINT_DIR / config["checkpoint"]

        if not checkpoint_path.exists():
            print(f"[WARNING] Checkpoint not found: {checkpoint_path}")
            print(f"  Skipping {model_type}. Train it first with:")
            print(f"  python Model/train.py --model_type {model_type}")
            continue

        # Load model
        model = load_model(model_type, checkpoint_path)
        param_count = count_parameters(model)

        # Choose appropriate dataloader
        loader = test_loader_fusion if config["needs_fusion_data"] else test_loader_single

        # Evaluate
        preds, gts = evaluate_model(model, loader, config["needs_fusion_data"])

        # Calculate metrics
        preds_arr = np.array(preds)
        gts_arr = np.array(gts)
        mae_val = mae(preds_arr, gts_arr)
        rmse_val = rmse(preds_arr, gts_arr)

        # Load checkpoint info
        ckpt = torch.load(checkpoint_path, map_location="cpu")

        results[model_type] = {
            "predictions": preds,
            "ground_truth": gts,
            "mae": mae_val,
            "rmse": rmse_val,
            "params": param_count,
            "best_epoch": ckpt.get("epoch", "N/A"),
        }

        print(f"MAE:  {mae_val:.4f}")
        print(f"RMSE: {rmse_val:.4f}")
        print(f"Parameters: {param_count:,} ({param_count/1e6:.2f}M)")
        print(f"Best epoch: {ckpt.get('epoch', 'N/A')}")

    if not results:
        print("\n[FAILED] No models were evaluated. Please train models first.")
        print("\nRun: python Model/train_all_models.py")
        return 1

    # Create comparison table
    print(f"\n{'='*80}")
    print("COMPARISON TABLE")
    print(f"{'='*80}")
    df = create_comparison_table(results)
    print(df.to_string(index=False))

    # Save table to CSV
    table_path = OUTPUT_DIR / "metrics_comparison.csv"
    df.to_csv(table_path, index=False)
    print(f"\nTable saved to {table_path}")

    # Generate visualizations
    print(f"\n{'='*80}")
    print("GENERATING COMPARISON VISUALIZATIONS")
    print(f"{'='*80}\n")

    plot_scatter_comparison(results, OUTPUT_DIR / "scatter_comparison.png")
    plot_metrics_comparison(results, OUTPUT_DIR / "metrics_comparison.png")
    plot_confusion_matrices(results, OUTPUT_DIR / "confusion_matrix_comparison.png")

    # Save predictions
    save_predictions(results, OUTPUT_DIR)

    print(f"\n{'='*80}")
    print("EVALUATION COMPLETE")
    print(f"{'='*80}")
    print(f"\nAll results saved to: {OUTPUT_DIR}")
    print("\nGenerated files in Results/comparison/:")
    print("  - metrics_comparison.csv")
    print("  - metrics_comparison.png")
    print("  - scatter_comparison.png")
    print("  - confusion_matrix_comparison.png")
    print("\nGenerated files in Results/predictions/:")
    print("  - predictions_baseline_test.csv")
    print("  - predictions_late_fusion_test.csv")
    print("  - predictions_feature_fusion_test.csv")

    return 0


if __name__ == "__main__":
    sys.exit(main())
