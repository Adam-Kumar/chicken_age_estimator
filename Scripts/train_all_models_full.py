"""train_all_models_full.py
Master script to train ALL 27 model configurations with 3-fold cross-validation.

ENHANCED VERSION - Saves:
- Model checkpoints for each fold
- Per-epoch training history
- Detailed predictions (for graphs/analysis)
- All metrics (MAE, RMSE, R2, correlation)

Training plan:
- 9 TOP View models (single-view)
- 9 Late Fusion models (two-view)
- 9 Feature Fusion models (two-view)

Total: 27 models × 3 folds = 81 experiments
Estimated time: 40-70 hours with GPU

Features:
- Resumable: Skips completed experiments
- Progress tracking: Saves after each fold
- Comprehensive metrics and checkpoints for analysis
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import json
import time
import random
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr

from Model.model import ResNetRegressor, LateFusionRegressor, FeatureFusionRegressor
from Model.dataset import ChickenAgeDataset, ChickenAgePairedDataset, get_default_transforms
from torch.utils.data import DataLoader


# Configuration - OPTIMIZED hyperparameters
RANDOM_SEED = 42

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

FUSION_TYPES = ["top_view", "late", "feature"]
N_FOLDS = 3
EPOCHS = 50
BATCH_SIZE = 16
WEIGHT_DECAY = 1e-2
WARMUP_EPOCHS = 3
EARLY_STOP_PATIENCE = 10
EARLY_STOP_DELTA = 0.001
GRADIENT_CLIP_NORM = 1.0


def set_random_seeds(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seeds set to {seed} for reproducibility\n")


def get_lr_for_backbone(backbone):
    """Get appropriate learning rate for backbone."""
    if "vit" in backbone or "swin" in backbone:
        return 2e-5
    elif "convnext" in backbone:
        return 2e-5
    else:
        return 2e-5


def compute_detailed_metrics(predictions, targets):
    """Compute all metrics: MAE, RMSE, R2, Pearson correlation."""
    mae = mean_absolute_error(targets, predictions)
    rmse = np.sqrt(mean_squared_error(targets, predictions))
    r2 = r2_score(targets, predictions)
    corr, _ = pearsonr(predictions, targets)

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "correlation": float(corr)
    }


def get_predictions_single(model, loader, device):
    """Get predictions and targets for single-view model."""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            all_preds.extend(preds.cpu().numpy().flatten())
            all_targets.extend(y.cpu().numpy().flatten())

    return np.array(all_preds), np.array(all_targets)


def get_predictions_fusion(model, loader, device):
    """Get predictions and targets for fusion model."""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for (x_top, x_side), y in loader:
            x_top, x_side, y = x_top.to(device), x_side.to(device), y.to(device)
            preds = model(x_top, x_side)
            all_preds.extend(preds.cpu().numpy().flatten())
            all_targets.extend(y.cpu().numpy().flatten())

    return np.array(all_preds), np.array(all_targets)


def train_one_epoch_single(model, loader, criterion, optimizer, device):
    """Train single-view model for one epoch."""
    model.train()
    total_mae = 0.0
    num_batches = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        preds = model(x)
        loss = criterion(preds, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP_NORM)
        optimizer.step()

        total_mae += (preds - y).abs().mean().item()
        num_batches += 1

    return total_mae / num_batches


def train_one_epoch_fusion(model, loader, criterion, optimizer, device):
    """Train fusion model for one epoch."""
    model.train()
    total_mae = 0.0
    num_batches = 0

    for (x_top, x_side), y in loader:
        x_top, x_side, y = x_top.to(device), x_side.to(device), y.to(device)
        optimizer.zero_grad()
        preds = model(x_top, x_side)
        loss = criterion(preds, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP_NORM)
        optimizer.step()

        total_mae += (preds - y).abs().mean().item()
        num_batches += 1

    return total_mae / num_batches


def validate_single(model, loader, device):
    """Validate single-view model."""
    model.eval()
    total_mae = 0.0
    num_batches = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            total_mae += (preds - y).abs().mean().item()
            num_batches += 1

    return total_mae / num_batches


def validate_fusion(model, loader, device):
    """Validate fusion model."""
    model.eval()
    total_mae = 0.0
    num_batches = 0

    with torch.no_grad():
        for (x_top, x_side), y in loader:
            x_top, x_side, y = x_top.to(device), x_side.to(device), y.to(device)
            preds = model(x_top, x_side)
            total_mae += (preds - y).abs().mean().item()
            num_batches += 1

    return total_mae / num_batches


def create_cv_splits_single(labels_csv, n_splits=3, random_state=42):
    """Create K-fold splits for single-view."""
    df = pd.read_csv(labels_csv)
    df = df[df['view'].str.upper() == 'TOP VIEW'].copy()

    unique_chickens = df['piece_id'].unique()
    print(f"  Found {len(unique_chickens)} unique chickens for CV")

    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits = []

    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(unique_chickens)):
        train_chickens = unique_chickens[train_idx]
        val_chickens = unique_chickens[val_idx]

        train_df = df[df['piece_id'].isin(train_chickens)]
        val_df = df[df['piece_id'].isin(val_chickens)]

        splits.append((train_df, val_df))

    return splits


def create_cv_splits_paired(labels_csv, n_splits=3, random_state=42):
    """Create K-fold splits for paired view."""
    df = pd.read_csv(labels_csv)
    df_top = df[df['view'].str.upper() == 'TOP VIEW'].copy()

    unique_chickens = df_top['piece_id'].unique()
    print(f"  Found {len(unique_chickens)} unique chickens for CV")

    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits = []

    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(unique_chickens)):
        train_chickens = unique_chickens[train_idx]
        val_chickens = unique_chickens[val_idx]

        train_df = df_top[df_top['piece_id'].isin(train_chickens)]
        val_df = df_top[df_top['piece_id'].isin(val_chickens)]

        splits.append((train_df, val_df))

    return splits


def train_top_view_experiment(backbone, fold_idx, train_df, val_df, device):
    """Train TOP view experiment with full metric tracking."""
    print(f"\n    Training {backbone} - TOP View - Fold {fold_idx+1}/{N_FOLDS}")

    model = ResNetRegressor(backbone_name=backbone, pretrained=True).to(device)

    # Save temp CSVs
    temp_dir = project_root / "temp" / "all_models"
    temp_dir.mkdir(parents=True, exist_ok=True)

    train_csv = temp_dir / f"train_top_view_{backbone}_f{fold_idx}.csv"
    val_csv = temp_dir / f"val_top_view_{backbone}_f{fold_idx}.csv"
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)

    # Data loaders
    root_dir = project_root / "Dataset_Processed"
    mask_dir = project_root / "Segmentation" / "masks"
    train_ds = ChickenAgeDataset(train_csv, root_dir, get_default_transforms(True), use_masks=True, mask_dir=mask_dir)
    val_ds = ChickenAgeDataset(val_csv, root_dir, get_default_transforms(False), use_masks=True, mask_dir=mask_dir)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Training setup
    criterion = nn.MSELoss()
    lr = get_lr_for_backbone(backbone)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

    def warmup_lambda(epoch):
        if epoch < WARMUP_EPOCHS:
            return (epoch + 1) / WARMUP_EPOCHS
        return 1.0

    warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)
    plateau_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=False)

    best_val_mae = float("inf")
    epochs_without_improvement = 0
    training_history = []
    best_model_state = None

    for epoch in range(1, EPOCHS + 1):
        train_mae = train_one_epoch_single(model, train_loader, criterion, optimizer, device)
        val_mae = validate_single(model, val_loader, device)

        # Save epoch history
        training_history.append({
            "epoch": epoch,
            "train_mae": float(train_mae),
            "val_mae": float(val_mae)
        })

        # Learning rate scheduling
        if epoch <= WARMUP_EPOCHS:
            warmup_scheduler.step()
        else:
            plateau_scheduler.step(val_mae)

        # Check for improvement and save best model
        if val_mae < best_val_mae - EARLY_STOP_DELTA:
            best_val_mae = val_mae
            epochs_without_improvement = 0
            best_model_state = model.state_dict().copy()
        else:
            epochs_without_improvement += 1

        # Early stopping
        if epochs_without_improvement >= EARLY_STOP_PATIENCE:
            print(f"      Early stopping at epoch {epoch}")
            break

        if epoch % 10 == 0:
            print(f"      Epoch {epoch}/{EPOCHS} - Val MAE: {val_mae:.4f} (Best: {best_val_mae:.4f})")

    # Load best model for final predictions
    model.load_state_dict(best_model_state)

    # Get detailed predictions and metrics
    val_preds, val_targets = get_predictions_single(model, val_loader, device)
    val_metrics = compute_detailed_metrics(val_preds, val_targets)

    # Save checkpoint
    checkpoint_dir = project_root / "Results" / "comparison" / "checkpoints" / f"{backbone}_top_view"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"fold{fold_idx}.pth"
    torch.save(best_model_state, checkpoint_path)

    # Save training history
    history_dir = project_root / "Results" / "comparison" / "training_history" / f"{backbone}_top_view"
    history_dir.mkdir(parents=True, exist_ok=True)
    history_path = history_dir / f"fold{fold_idx}.json"
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)

    # Save predictions
    pred_dir = project_root / "Results" / "comparison" / "predictions" / f"{backbone}_top_view"
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_path = pred_dir / f"fold{fold_idx}.csv"
    pred_df = pd.DataFrame({
        "target": val_targets,
        "prediction": val_preds
    })
    pred_df.to_csv(pred_path, index=False)

    return best_val_mae, val_metrics, training_history


def train_fusion_experiment(backbone, fusion_type, fold_idx, train_df, val_df, device):
    """Train fusion experiment with full metric tracking."""
    print(f"\n    Training {backbone} - {fusion_type.capitalize()} - Fold {fold_idx+1}/{N_FOLDS}")

    if fusion_type == "late":
        model = LateFusionRegressor(backbone_name=backbone, pretrained=True).to(device)
    else:  # feature
        model = FeatureFusionRegressor(backbone_name=backbone, pretrained=True).to(device)

    # Save temp CSVs
    temp_dir = project_root / "temp" / "all_models"
    temp_dir.mkdir(parents=True, exist_ok=True)

    train_csv = temp_dir / f"train_{fusion_type}_{backbone}_f{fold_idx}.csv"
    val_csv = temp_dir / f"val_{fusion_type}_{backbone}_f{fold_idx}.csv"
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)

    # Data loaders
    root_dir = project_root / "Dataset_Processed"
    mask_dir = project_root / "Segmentation" / "masks"
    train_ds = ChickenAgePairedDataset(train_csv, root_dir, get_default_transforms(True), use_masks=True, mask_dir=mask_dir)
    val_ds = ChickenAgePairedDataset(val_csv, root_dir, get_default_transforms(False), use_masks=True, mask_dir=mask_dir)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Training setup
    criterion = nn.MSELoss()
    lr = get_lr_for_backbone(backbone)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

    def warmup_lambda(epoch):
        if epoch < WARMUP_EPOCHS:
            return (epoch + 1) / WARMUP_EPOCHS
        return 1.0

    warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)
    plateau_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=False)

    best_val_mae = float("inf")
    epochs_without_improvement = 0
    training_history = []
    best_model_state = None

    for epoch in range(1, EPOCHS + 1):
        train_mae = train_one_epoch_fusion(model, train_loader, criterion, optimizer, device)
        val_mae = validate_fusion(model, val_loader, device)

        # Save epoch history
        training_history.append({
            "epoch": epoch,
            "train_mae": float(train_mae),
            "val_mae": float(val_mae)
        })

        # Learning rate scheduling
        if epoch <= WARMUP_EPOCHS:
            warmup_scheduler.step()
        else:
            plateau_scheduler.step(val_mae)

        # Check for improvement and save best model
        if val_mae < best_val_mae - EARLY_STOP_DELTA:
            best_val_mae = val_mae
            epochs_without_improvement = 0
            best_model_state = model.state_dict().copy()
        else:
            epochs_without_improvement += 1

        # Early stopping
        if epochs_without_improvement >= EARLY_STOP_PATIENCE:
            print(f"      Early stopping at epoch {epoch}")
            break

        if epoch % 10 == 0:
            print(f"      Epoch {epoch}/{EPOCHS} - Val MAE: {val_mae:.4f} (Best: {best_val_mae:.4f})")

    # Load best model for final predictions
    model.load_state_dict(best_model_state)

    # Get detailed predictions and metrics
    val_preds, val_targets = get_predictions_fusion(model, val_loader, device)
    val_metrics = compute_detailed_metrics(val_preds, val_targets)

    # Save checkpoint
    checkpoint_dir = project_root / "Results" / "comparison" / "checkpoints" / f"{backbone}_{fusion_type}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"fold{fold_idx}.pth"
    torch.save(best_model_state, checkpoint_path)

    # Save training history
    history_dir = project_root / "Results" / "comparison" / "training_history" / f"{backbone}_{fusion_type}"
    history_dir.mkdir(parents=True, exist_ok=True)
    history_path = history_dir / f"fold{fold_idx}.json"
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)

    # Save predictions
    pred_dir = project_root / "Results" / "comparison" / "predictions" / f"{backbone}_{fusion_type}"
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_path = pred_dir / f"fold{fold_idx}.csv"
    pred_df = pd.DataFrame({
        "target": val_targets,
        "prediction": val_preds
    })
    pred_df.to_csv(pred_path, index=False)

    return best_val_mae, val_metrics, training_history


def load_progress():
    """Load existing progress."""
    progress_file = project_root / "Results" / "comparison" / "full_progress.json"
    if progress_file.exists():
        with open(progress_file, 'r') as f:
            return json.load(f)
    return {}


def save_progress(results):
    """Save progress."""
    results_dir = project_root / "Results" / "comparison"
    results_dir.mkdir(parents=True, exist_ok=True)

    progress_file = results_dir / "full_progress.json"
    with open(progress_file, 'w') as f:
        json.dump(results, f, indent=2)


def main():
    set_random_seeds(RANDOM_SEED)

    print("="*80)
    print("TRAIN ALL 27 MODEL CONFIGURATIONS - FULL VERSION")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Random seed: {RANDOM_SEED}")
    print(f"  Epochs: {EPOCHS} (with early stopping)")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"\nSaving:")
    print(f"  - Model checkpoints for each fold")
    print(f"  - Per-epoch training history")
    print(f"  - Detailed predictions (for graphs)")
    print(f"  - All metrics (MAE, RMSE, R2, correlation)")
    print(f"\nTraining plan:")
    print(f"  - 9 TOP View models (single-view)")
    print(f"  - 9 Late Fusion models (two-view)")
    print(f"  - 9 Feature Fusion models (two-view)")
    print(f"  Total: 27 models × {N_FOLDS} folds = {27 * N_FOLDS} experiments")
    print("="*80)

    # Prepare labels
    labels_csv = project_root / "Labels" / "labels.csv"
    if not labels_csv.exists():
        print("\nCombining train/val/test splits...")
        train_df = pd.read_csv(project_root / "Labels" / "train.csv")
        val_df = pd.read_csv(project_root / "Labels" / "val.csv")
        test_df = pd.read_csv(project_root / "Labels" / "test.csv")
        combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
        combined_csv = project_root / "Labels" / "combined_all.csv"
        combined_df.to_csv(combined_csv, index=False)
        labels_csv = combined_csv

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}\n")

    # Load existing progress
    all_results = load_progress()

    total_experiments = len(BACKBONES) * len(FUSION_TYPES) * N_FOLDS
    completed = sum(len(res.get("folds", [])) for res in all_results.values())

    print(f"Progress: {completed}/{total_experiments} experiments already completed\n")

    # Train all models
    for backbone in BACKBONES:
        print(f"\n{'='*80}")
        print(f"BACKBONE: {backbone.upper()}")
        print(f"{'='*80}")

        for fusion_type in FUSION_TYPES:
            key = f"{backbone}_{fusion_type}"

            # Initialize results
            if key not in all_results:
                all_results[key] = {"folds": []}

            folds_completed = len(all_results[key]["folds"])

            if folds_completed >= N_FOLDS:
                print(f"  Skipping {backbone} - {fusion_type.capitalize()} (already completed)")
                continue

            print(f"\n  Training: {backbone.upper()} + {fusion_type.upper()}")

            # Create CV splits
            if fusion_type == "top_view":
                splits = create_cv_splits_single(labels_csv, n_splits=N_FOLDS)
            else:
                splits = create_cv_splits_paired(labels_csv, n_splits=N_FOLDS)

            # Train remaining folds
            for fold_idx in range(folds_completed, N_FOLDS):
                train_df, val_df = splits[fold_idx]

                try:
                    start_time = time.time()

                    if fusion_type == "top_view":
                        mae, metrics, history = train_top_view_experiment(
                            backbone, fold_idx, train_df, val_df, device
                        )
                    else:
                        mae, metrics, history = train_fusion_experiment(
                            backbone, fusion_type, fold_idx, train_df, val_df, device
                        )

                    elapsed = time.time() - start_time

                    # Save fold result
                    fold_result = {
                        "fold": fold_idx,
                        "mae": mae,
                        "metrics": metrics,
                        "num_epochs": len(history),
                        "training_time_min": elapsed / 60
                    }
                    all_results[key]["folds"].append(fold_result)

                    # Compute aggregate statistics
                    fold_maes = [f["mae"] for f in all_results[key]["folds"]]
                    all_results[key]["mean_mae"] = float(np.mean(fold_maes))
                    all_results[key]["std_mae"] = float(np.std(fold_maes))

                    # Save progress
                    save_progress(all_results)

                    completed += 1
                    remaining = total_experiments - completed

                    print(f"\n      Fold {fold_idx+1} complete:")
                    print(f"        MAE: {mae:.4f}")
                    print(f"        RMSE: {metrics['rmse']:.4f}")
                    print(f"        R2: {metrics['r2']:.4f}")
                    print(f"        Time: {elapsed/60:.1f} min")
                    print(f"      Progress: {completed}/{total_experiments} ({completed/total_experiments*100:.1f}%)")
                    if remaining > 0:
                        print(f"      Estimated remaining: {remaining * elapsed / 3600:.1f} hours")

                except Exception as e:
                    print(f"\n      [FAILED] Fold {fold_idx+1} failed: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

    print("\n" + "="*80)
    print("ALL MODELS TRAINED!")
    print("="*80)
    print(f"\nResults saved to: Results/comparison/")
    print(f"  - full_progress.json (summary)")
    print(f"  - checkpoints/[model]/fold[N].pth")
    print(f"  - training_history/[model]/fold[N].json")
    print(f"  - predictions/[model]/fold[N].csv")
    print(f"\nNext steps:")
    print(f"  1. Run Scripts/evaluating/evaluate_all_models.py for analysis")
    print(f"  2. Use saved checkpoints for Grad-CAM or additional analysis")


if __name__ == "__main__":
    main()
