"""train_other_strategies.py
Train ConvNeXt-B with alternative fusion strategies using 3-fold CV.

Trains comparison strategies for the best model:
- TOP view only (single-view baseline)
- Late Fusion (view-aware: separate models, averaged predictions)

Uses 3-fold cross-validation matching train_best_model.py methodology.

Output:
- Checkpoints: checkpoints/convnext_b_{strategy}_fold{0,1,2}.pth
- Results: Results/other_strategies/convnext_b/other_strategies_metrics.json (averaged across folds)

For comparison, run Scripts/evaluating/compare_best_model.py to compare all strategies
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
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

from Model.model import ResNetRegressor, LateFusionRegressor
from Model.dataset import ChickenAgeDataset, ChickenAgePairedDataset, get_default_transforms
from torch.utils.data import DataLoader


# Configuration - Same hyperparameters as train_best_model.py
RANDOM_SEED = 42  # For reproducibility
BACKBONE = "convnext_b"
N_FOLDS = 3
EPOCHS = 50
BATCH_SIZE = 16
LR = 2e-5
WEIGHT_DECAY = 1e-2
WARMUP_EPOCHS = 3
EARLY_STOP_PATIENCE = 10
EARLY_STOP_DELTA = 0.001
GRADIENT_CLIP_NORM = 1.0


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

    print(f"Random seeds set to {seed} for reproducibility\n")


def create_cv_splits(labels_csv, n_splits=3, random_state=42):
    """Create K-fold splits at chicken level (same as train_best_model.py)."""
    df = pd.read_csv(labels_csv)
    df_top = df[df['view'].str.upper() == 'TOP VIEW'].copy()
    unique_chickens = df_top['piece_id'].unique()

    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits = []

    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(unique_chickens)):
        train_chickens = unique_chickens[train_idx]
        val_chickens = unique_chickens[val_idx]
        train_df = df_top[df_top['piece_id'].isin(train_chickens)]
        val_df = df_top[df_top['piece_id'].isin(val_chickens)]
        splits.append((train_df, val_df))

    return splits


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

        if GRADIENT_CLIP_NORM is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP_NORM)

        optimizer.step()
        total_mae += (preds - y).abs().mean().item()
        num_batches += 1

    return total_mae / num_batches


def train_one_epoch_late(model, loader, criterion, optimizer, device):
    """Train late fusion model for one epoch."""
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


def validate_late(model, loader, device):
    """Validate late fusion model."""
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


def train_strategy_fold(strategy_name, view_filter, fold_idx, train_df, val_df, device, is_fusion=False):
    """Train a single strategy on a single fold."""
    # Save temp CSVs
    temp_dir = project_root / "temp" / "other_strategies_cv"
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Create datasets based on strategy type
    root_dir = project_root / "Dataset_Processed"
    mask_dir = project_root / "Segmentation" / "masks"

    if is_fusion:
        # Late Fusion: paired dataset
        train_csv = temp_dir / f"train_{strategy_name.lower().replace(' ', '_')}_f{fold_idx}.csv"
        val_csv = temp_dir / f"val_{strategy_name.lower().replace(' ', '_')}_f{fold_idx}.csv"
        train_df.to_csv(train_csv, index=False)
        val_df.to_csv(val_csv, index=False)

        train_ds = ChickenAgePairedDataset(train_csv, root_dir, get_default_transforms(True), use_masks=True, mask_dir=mask_dir)
        val_ds = ChickenAgePairedDataset(val_csv, root_dir, get_default_transforms(False), use_masks=True, mask_dir=mask_dir)
        model = LateFusionRegressor(backbone_name=BACKBONE, pretrained=True).to(device)
    else:
        # Single view (TOP only): filter by view
        train_filtered = train_df.copy()  # Already filtered to TOP view from splits
        val_filtered = val_df.copy()

        train_csv = temp_dir / f"train_{strategy_name.lower().replace(' ', '_')}_f{fold_idx}.csv"
        val_csv = temp_dir / f"val_{strategy_name.lower().replace(' ', '_')}_f{fold_idx}.csv"
        train_filtered.to_csv(train_csv, index=False)
        val_filtered.to_csv(val_csv, index=False)

        train_ds = ChickenAgeDataset(train_csv, root_dir, get_default_transforms(True), use_masks=True, mask_dir=mask_dir)
        val_ds = ChickenAgeDataset(val_csv, root_dir, get_default_transforms(False), use_masks=True, mask_dir=mask_dir)
        model = ResNetRegressor(backbone_name=BACKBONE, pretrained=True).to(device)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Training setup
    criterion = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    def warmup_lambda(epoch):
        if epoch < WARMUP_EPOCHS:
            return (epoch + 1) / WARMUP_EPOCHS
        return 1.0

    warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)
    plateau_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=False)

    best_val_mae = float("inf")
    best_epoch = 0
    best_model_state = None
    epochs_without_improvement = 0

    # Training loop
    for epoch in range(1, EPOCHS + 1):
        if is_fusion:
            train_mae = train_one_epoch_late(model, train_loader, criterion, optimizer, device)
            val_mae = validate_late(model, val_loader, device)
        else:
            train_mae = train_one_epoch_single(model, train_loader, criterion, optimizer, device)
            val_mae = validate_single(model, val_loader, device)

        # Learning rate scheduling
        if epoch <= WARMUP_EPOCHS:
            warmup_scheduler.step()
        else:
            plateau_scheduler.step(val_mae)

        # Check for improvement
        if val_mae < best_val_mae - EARLY_STOP_DELTA:
            best_val_mae = val_mae
            best_epoch = epoch
            best_model_state = model.state_dict().copy()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Early stopping
        if epochs_without_improvement >= EARLY_STOP_PATIENCE:
            break

    # Save checkpoint
    checkpoint_dir = project_root / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint_name = f"{BACKBONE}_{strategy_name.lower().replace(' ', '_')}_fold{fold_idx}.pth"
    checkpoint_path = checkpoint_dir / checkpoint_name
    torch.save(best_model_state, checkpoint_path)

    return {
        'val_mae': float(best_val_mae),
        'best_epoch': int(best_epoch),
    }


def main():
    # Set random seeds for reproducibility
    set_random_seeds(RANDOM_SEED)

    print("="*80)
    print(f"TRAINING OTHER STRATEGIES: {BACKBONE.upper()} (3-Fold CV)")
    print("="*80)
    print(f"\nStrategies:")
    print(f"  1. TOP view only")
    print(f"  2. Late Fusion")
    print(f"\nConfiguration:")
    print(f"  Random seed: {RANDOM_SEED}")
    print(f"  Backbone: {BACKBONE.upper()}")
    print(f"  Folds: {N_FOLDS}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Learning rate: {LR:.2e}")
    print(f"  Batch size: {BATCH_SIZE}")
    print("="*80)

    # Load labels and create CV splits
    labels_csv = project_root / "Labels" / "labels.csv"
    if not labels_csv.exists():
        raise FileNotFoundError(f"labels.csv not found at {labels_csv}")

    print(f"\nCreating {N_FOLDS}-fold CV splits...")
    splits = create_cv_splits(labels_csv, n_splits=N_FOLDS)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Define strategies
    strategies = [
        ("TOP View", "TOP VIEW", False),
        ("Late Fusion", None, True),
    ]

    all_results = {}

    # Train each strategy on all folds
    for strategy_name, view_filter, is_fusion in strategies:
        print("\n" + "="*80)
        print(f"STRATEGY: {strategy_name}")
        print("="*80)

        fold_results = []

        for fold_idx, (train_df, val_df) in enumerate(splits):
            print(f"\n  Fold {fold_idx + 1}/{N_FOLDS}...", end=" ")

            result = train_strategy_fold(
                strategy_name, view_filter, fold_idx, train_df, val_df, device, is_fusion
            )

            fold_results.append(result)
            print(f"Val MAE: {result['val_mae']:.4f}")

        # Compute averaged metrics
        val_maes = [r['val_mae'] for r in fold_results]
        mean_val_mae = np.mean(val_maes)
        std_val_mae = np.std(val_maes)

        print(f"\n  ** {strategy_name}: {mean_val_mae:.4f} ± {std_val_mae:.4f} days **")

        # Store results
        strategy_key = strategy_name.lower().replace(' ', '_')
        all_results[strategy_key] = {
            "strategy": strategy_name,
            "mean_val_mae": float(mean_val_mae),
            "std_val_mae": float(std_val_mae),
            "n_folds": N_FOLDS,
            "fold_results": fold_results,
        }

    # Save all results
    results_dir = project_root / "Results" / "other_strategies" / BACKBONE
    results_dir.mkdir(parents=True, exist_ok=True)

    results_file = results_dir / "other_strategies_metrics.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Print final summary
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"\nAveraged Results ({N_FOLDS}-Fold CV):")
    for strategy_key, results in all_results.items():
        print(f"\n  {results['strategy']}:")
        print(f"    MAE: {results['mean_val_mae']:.4f} ± {results['std_val_mae']:.4f} days")

    print(f"\nOutputs:")
    print(f"  - Checkpoints: checkpoints/{BACKBONE}_{{strategy}}_fold{{0,1,2}}.pth")
    print(f"  - Metrics: {results_file}")
    print("\nNext steps:")
    print("  - Run Scripts/evaluating/compare_best_model.py to compare all strategies")


if __name__ == "__main__":
    main()
