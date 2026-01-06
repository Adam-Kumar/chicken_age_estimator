"""train_best_model.py
Train the best model: ConvNeXt-B Feature Fusion with 3-fold CV.

Uses same methodology as train_all_models.py for consistency:
- 3-fold cross-validation from labels.csv
- Averaged validation metrics across folds
- ConvNeXt-B backbone with Feature Fusion strategy

Output:
- Checkpoints: checkpoints/convnext_b_feature_fold{0,1,2}.pth
- Results: Results/best_model/metrics.json (averaged across folds with training history)

HYPERPARAMETER TUNING SUGGESTIONS:
See configuration section below - modify and re-run to improve performance.
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import time
import json
import random
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR, CosineAnnealingLR
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

from Model.model import FeatureFusionRegressor
from Model.dataset import ChickenAgePairedDataset, get_default_transforms
from torch.utils.data import DataLoader


# ============================================================================
# CONFIGURATION - HYPERPARAMETER TUNING OPTIONS
# ============================================================================

# Random seed for reproducibility
RANDOM_SEED = 42

BACKBONE = "convnext_b"
N_FOLDS = 3

# === CORE HYPERPARAMETERS ===
EPOCHS = 50

# TUNE: Batch size
# Options: 8 (more regularization), 16 (baseline), 24, 32 (faster but less stable)
BATCH_SIZE = 16

# TUNE: Learning rate (MOST IMPORTANT!)
# For ConvNeXt-B, try:
#   5e-6: Very conservative, better generalization (RECOMMENDED for small dataset)
#   1e-5: Conservative, reduce overfitting
#   2e-5: Baseline (current)
#   3e-5: More aggressive
LR = 2e-5

# TUNE: Weight decay (L2 regularization)
# Options: 5e-3, 1e-2 (baseline), 2e-2, 5e-2 (aggressive regularization)
WEIGHT_DECAY = 1e-2

# TUNE: Warmup epochs
# Options: 0 (no warmup), 3 (baseline), 5 (longer warmup)
WARMUP_EPOCHS = 3

# TUNE: Early stopping
# Patience: 7 (aggressive), 10 (baseline), 15 (patient)
# Delta: Minimum improvement to count
EARLY_STOP_PATIENCE = 10
EARLY_STOP_DELTA = 0.001

# Gradient clipping (usually keep at 1.0)
GRADIENT_CLIP_NORM = 1.0

# === ADVANCED OPTIONS ===

# Learning rate scheduler
# Options: "plateau" (baseline), "cosine" (smooth decay)
LR_SCHEDULER = "plateau"

# ============================================================================
# SUGGESTED TUNING STRATEGY:
#
# If overfitting (val < test):
#   1. Reduce LR to 1e-5 or 5e-6
#   2. Increase WEIGHT_DECAY to 2e-2
#   3. Reduce BATCH_SIZE to 8
#   4. Try LR_SCHEDULER = "cosine"
#
# If underfitting (both val and test high):
#   1. Increase LR to 3e-5
#   2. Increase BATCH_SIZE to 24 or 32
#   3. Reduce WEIGHT_DECAY to 5e-3
#   4. Increase EPOCHS to 70
# ============================================================================


def set_random_seeds(seed=42):
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Make CuDNN deterministic (may reduce performance slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Random seeds set to {seed} for reproducibility")


def create_cv_splits(labels_csv, n_splits=3, random_state=42):
    """Create K-fold splits at chicken level (same as train_all_models.py)."""
    df = pd.read_csv(labels_csv)
    df_top = df[df['view'].str.upper() == 'TOP VIEW'].copy()
    unique_chickens = df_top['piece_id'].unique()

    print(f"Creating {n_splits}-fold CV from {len(unique_chickens)} chickens")

    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits = []

    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(unique_chickens)):
        train_chickens = unique_chickens[train_idx]
        val_chickens = unique_chickens[val_idx]
        train_df = df_top[df_top['piece_id'].isin(train_chickens)]
        val_df = df_top[df_top['piece_id'].isin(val_chickens)]

        print(f"  Fold {fold_idx+1}: {len(train_chickens)} train, {len(val_chickens)} val chickens")
        splits.append((train_df, val_df))

    return splits


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch with gradient clipping."""
    model.train()
    total_mae = 0.0
    num_batches = 0

    for (x_top, x_side), y in loader:
        x_top, x_side, y = x_top.to(device), x_side.to(device), y.to(device)

        optimizer.zero_grad()
        preds = model(x_top, x_side)
        loss = criterion(preds, y)
        loss.backward()

        if GRADIENT_CLIP_NORM is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP_NORM)

        optimizer.step()

        total_mae += (preds - y).abs().mean().item()
        num_batches += 1

    return total_mae / num_batches


def validate(model, loader, device):
    """Validate the model."""
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


def train_single_fold(fold_idx, train_df, val_df, device):
    """Train a single fold."""
    print(f"\n{'='*80}")
    print(f"FOLD {fold_idx + 1}/{N_FOLDS}")
    print(f"{'='*80}")

    # Save temp CSVs
    temp_dir = project_root / "temp" / "best_model_cv"
    temp_dir.mkdir(parents=True, exist_ok=True)

    train_csv = temp_dir / f"train_fold{fold_idx}.csv"
    val_csv = temp_dir / f"val_fold{fold_idx}.csv"
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)

    # Create datasets
    root_dir = project_root / "Dataset_Processed"
    mask_dir = project_root / "Segmentation" / "masks"

    train_ds = ChickenAgePairedDataset(train_csv, root_dir, get_default_transforms(True), use_masks=True, mask_dir=mask_dir)
    val_ds = ChickenAgePairedDataset(val_csv, root_dir, get_default_transforms(False), use_masks=True, mask_dir=mask_dir)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")

    # Create model
    model = FeatureFusionRegressor(backbone_name=BACKBONE, pretrained=True).to(device)

    # Training setup
    criterion = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Learning rate scheduler
    if LR_SCHEDULER == "plateau":
        def warmup_lambda(epoch):
            if epoch < WARMUP_EPOCHS:
                return (epoch + 1) / WARMUP_EPOCHS
            return 1.0

        warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)
        plateau_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=False)
    elif LR_SCHEDULER == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR/100)

    # Track best model and training history
    best_val_mae = float("inf")
    best_epoch = 0
    best_model_state = None
    epochs_without_improvement = 0
    training_history = []  # Store per-epoch train/val MAE

    print("\nTraining...")
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        train_mae = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_mae = validate(model, val_loader, device)

        # Record training history
        training_history.append({
            'epoch': epoch,
            'train_mae': float(train_mae),
            'val_mae': float(val_mae)
        })

        # Learning rate scheduling
        if LR_SCHEDULER == "plateau":
            if epoch <= WARMUP_EPOCHS:
                warmup_scheduler.step()
            else:
                plateau_scheduler.step(val_mae)
        elif LR_SCHEDULER == "cosine":
            scheduler.step()

        # Check for improvement
        if val_mae < best_val_mae - EARLY_STOP_DELTA:
            best_val_mae = val_mae
            best_epoch = epoch
            best_model_state = model.state_dict().copy()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch % 10 == 0 or epoch == 1:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch:2d}/{EPOCHS} - Train: {train_mae:.4f}, "
                  f"Val: {val_mae:.4f}, LR: {current_lr:.2e} "
                  f"(Best: {best_val_mae:.4f} @ {best_epoch})")

        # Early stopping
        if epochs_without_improvement >= EARLY_STOP_PATIENCE:
            print(f"  Early stopped at epoch {epoch}")
            break

    training_time = time.time() - start_time

    print(f"\nFold {fold_idx+1} Results:")
    print(f"  Best Val MAE: {best_val_mae:.4f} (epoch {best_epoch})")
    print(f"  Training time: {training_time/60:.1f} min")

    # Save checkpoint
    checkpoint_dir = project_root / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{BACKBONE}_feature_fold{fold_idx}.pth"
    torch.save(best_model_state, checkpoint_path)

    return {
        'val_mae': float(best_val_mae),
        'best_epoch': int(best_epoch),
        'training_time': float(training_time),
        'training_history': training_history,  # Per-epoch train/val MAE
    }


def main():
    # Set random seeds for reproducibility
    set_random_seeds(RANDOM_SEED)

    print("="*80)
    print(f"TRAINING: {BACKBONE.upper()} Feature Fusion (3-Fold CV)")
    print("="*80)
    print(f"\nHyperparameters:")
    print(f"  Random seed: {RANDOM_SEED}")
    print(f"  Backbone: {BACKBONE.upper()}")
    print(f"  Folds: {N_FOLDS}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LR:.2e}")
    print(f"  Weight decay: {WEIGHT_DECAY:.2e}")
    print(f"  Warmup: {WARMUP_EPOCHS} epochs")
    print(f"  Early stop: patience={EARLY_STOP_PATIENCE}, delta={EARLY_STOP_DELTA}")
    print(f"  LR scheduler: {LR_SCHEDULER}")
    print("="*80)

    # Load labels and create CV splits
    labels_csv = project_root / "Labels" / "labels.csv"
    if not labels_csv.exists():
        raise FileNotFoundError(f"labels.csv not found at {labels_csv}")

    splits = create_cv_splits(labels_csv, n_splits=N_FOLDS)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}\n")

    # Train all folds
    fold_results = []
    total_start = time.time()

    for fold_idx, (train_df, val_df) in enumerate(splits):
        result = train_single_fold(fold_idx, train_df, val_df, device)
        fold_results.append(result)

    total_time = time.time() - total_start

    # Compute averaged metrics
    val_maes = [r['val_mae'] for r in fold_results]
    mean_val_mae = np.mean(val_maes)
    std_val_mae = np.std(val_maes)

    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"\nPer-Fold Results:")
    for i, result in enumerate(fold_results):
        print(f"  Fold {i+1}: Val MAE = {result['val_mae']:.4f}")

    print(f"\n** FINAL RESULT (3-Fold CV): {mean_val_mae:.4f} ± {std_val_mae:.4f} days **")
    print(f"\nTotal time: {total_time/60:.1f} minutes")

    # Save results
    results_dir = project_root / "Results" / "best_model"
    results_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "model": f"{BACKBONE.upper()} Feature Fusion",
        "strategy": "Feature Fusion",
        "n_folds": N_FOLDS,
        "mean_val_mae": float(mean_val_mae),
        "std_val_mae": float(std_val_mae),
        "fold_results": fold_results,
        "hyperparameters": {
            "backbone": BACKBONE,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LR,
            "weight_decay": WEIGHT_DECAY,
            "warmup_epochs": WARMUP_EPOCHS,
            "early_stop_patience": EARLY_STOP_PATIENCE,
            "lr_scheduler": LR_SCHEDULER,
        }
    }

    metrics_file = results_dir / "metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\nOutputs:")
    print(f"  - Checkpoints: checkpoints/{BACKBONE}_feature_fold{{0,1,2}}.pth")
    print(f"  - Metrics: {metrics_file}")
    print("\nNext steps:")
    print("  - Run Scripts/training/train_other_strategies.py for strategy comparison")
    print("  - Run Scripts/evaluating/evaluate_best_model.py for detailed analysis")
    print("\nTo improve performance, try:")
    print("  - Reduce LR to 1e-5 or 5e-6 (if overfitting)")
    print("  - Increase WEIGHT_DECAY to 2e-2 (if overfitting)")
    print("  - Try LR_SCHEDULER = 'cosine' (smoother learning)")


if __name__ == "__main__":
    main()
