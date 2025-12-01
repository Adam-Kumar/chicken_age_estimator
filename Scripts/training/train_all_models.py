"""train_all_models.py
Master script to train ALL 27 model configurations with 3-fold cross-validation.

Training plan:
- 9 Baseline models (single-view TOP) → Results/baseline_cv_results.json
- 18 Fusion models (9 backbones × 2 fusion types) → Results/comparison/csv/progress.json

Total: 27 models × 3 folds = 81 experiments
Estimated time: 40-70 hours with GPU

Features:
- Resumable: Skips completed experiments
- Progress tracking: Saves after each fold
- Comprehensive: All backbones and fusion strategies
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import json
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

from Model.model import ResNetRegressor, LateFusionRegressor, FeatureFusionRegressor
from Model.dataset import ChickenAgeDataset, ChickenAgePairedDataset, get_default_transforms
from torch.utils.data import DataLoader


# Configuration
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
N_FOLDS = 3
EPOCHS = 30
BATCH_SIZE = 8


def get_lr_for_backbone(backbone):
    """Get appropriate learning rate for backbone."""
    if "vit" in backbone or "swin" in backbone:
        return 5e-5  # Lower for transformers
    elif "convnext" in backbone:
        return 8e-5  # Slightly higher than pure transformers
    else:
        return 1e-4  # Standard for CNNs


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
    """Create K-fold splits for single-view (TOP only)."""
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
    """Create K-fold splits for paired view (TOP+SIDE)."""
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


def train_baseline_experiment(backbone, fold_idx, train_df, val_df, device):
    """Train baseline (single-view TOP) experiment."""
    print(f"\n    Training {backbone} - Baseline - Fold {fold_idx+1}/{N_FOLDS}")

    model = ResNetRegressor(backbone_name=backbone, pretrained=True).to(device)

    # Save temp CSVs
    temp_dir = project_root / "Scripts" / "temp_all_models"
    temp_dir.mkdir(exist_ok=True)

    train_csv = temp_dir / f"train_baseline_{backbone}_f{fold_idx}.csv"
    val_csv = temp_dir / f"val_baseline_{backbone}_f{fold_idx}.csv"
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)

    # Data loaders
    root_dir = project_root / "Dataset_Processed"
    train_ds = ChickenAgeDataset(train_csv, root_dir, get_default_transforms(True))
    val_ds = ChickenAgeDataset(val_csv, root_dir, get_default_transforms(False))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Training setup
    criterion = nn.MSELoss()
    lr = get_lr_for_backbone(backbone)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_mae = float("inf")

    for epoch in range(1, EPOCHS + 1):
        train_mae = train_one_epoch_single(model, train_loader, criterion, optimizer, device)
        val_mae = validate_single(model, val_loader, device)
        scheduler.step()

        if val_mae < best_val_mae:
            best_val_mae = val_mae

        if epoch % 10 == 0:
            print(f"      Epoch {epoch}/{EPOCHS} - Val MAE: {val_mae:.4f} (Best: {best_val_mae:.4f})")

    return best_val_mae


def train_fusion_experiment(backbone, fusion_type, fold_idx, train_df, val_df, device):
    """Train fusion (late or feature) experiment."""
    print(f"\n    Training {backbone} - {fusion_type.capitalize()} - Fold {fold_idx+1}/{N_FOLDS}")

    if fusion_type == "late":
        model = LateFusionRegressor(backbone_name=backbone, pretrained=True).to(device)
    else:  # feature
        model = FeatureFusionRegressor(backbone_name=backbone, pretrained=True).to(device)

    # Save temp CSVs
    temp_dir = project_root / "Scripts" / "temp_all_models"
    temp_dir.mkdir(exist_ok=True)

    train_csv = temp_dir / f"train_{fusion_type}_{backbone}_f{fold_idx}.csv"
    val_csv = temp_dir / f"val_{fusion_type}_{backbone}_f{fold_idx}.csv"
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)

    # Data loaders (paired dataset)
    root_dir = project_root / "Dataset_Processed"
    train_ds = ChickenAgePairedDataset(train_csv, root_dir, get_default_transforms(True))
    val_ds = ChickenAgePairedDataset(val_csv, root_dir, get_default_transforms(False))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Training setup
    criterion = nn.MSELoss()
    lr = get_lr_for_backbone(backbone)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_mae = float("inf")

    for epoch in range(1, EPOCHS + 1):
        train_mae = train_one_epoch_fusion(model, train_loader, criterion, optimizer, device)
        val_mae = validate_fusion(model, val_loader, device)
        scheduler.step()

        if val_mae < best_val_mae:
            best_val_mae = val_mae

        if epoch % 10 == 0:
            print(f"      Epoch {epoch}/{EPOCHS} - Val MAE: {val_mae:.4f} (Best: {best_val_mae:.4f})")

    return best_val_mae


def load_baseline_progress():
    """Load baseline results."""
    results_file = project_root / "Results" / "baseline_cv_results.json"
    if results_file.exists():
        with open(results_file, 'r') as f:
            return json.load(f)
    return {}


def save_baseline_progress(results):
    """Save baseline results."""
    results_dir = project_root / "Results"
    results_dir.mkdir(exist_ok=True)

    results_file = results_dir / "baseline_cv_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)


def load_fusion_progress():
    """Load fusion results."""
    results_file = project_root / "Results" / "comparison" / "csv" / "progress.json"
    if results_file.exists():
        with open(results_file, 'r') as f:
            return json.load(f)
    return {}


def save_fusion_progress(results):
    """Save fusion results."""
    results_dir = project_root / "Results" / "comparison" / "csv"
    results_dir.mkdir(parents=True, exist_ok=True)

    results_file = results_dir / "progress.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)


def main():
    print("="*80)
    print("TRAIN ALL 27 MODEL CONFIGURATIONS")
    print("="*80)
    print(f"\nTraining plan:")
    print(f"  - 9 Baseline models (single-view TOP)")
    print(f"  - 9 Late Fusion models (two-view)")
    print(f"  - 9 Feature Fusion models (two-view)")
    print(f"  Total: 27 models × {N_FOLDS} folds = {27 * N_FOLDS} experiments")
    print(f"  Estimated time: 40-70 hours with GPU")
    print("="*80)

    # Prepare combined labels
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
    baseline_results = load_baseline_progress()
    fusion_results = load_fusion_progress()

    total_experiments = len(BACKBONES) * len(FUSION_TYPES) * N_FOLDS
    completed = 0

    # Count already completed
    for backbone in BACKBONES:
        for fusion in FUSION_TYPES:
            key = f"{backbone}_{fusion}"
            if fusion == "baseline":
                if key in baseline_results and len(baseline_results[key].get("fold_results", [])) == N_FOLDS:
                    completed += N_FOLDS
            else:
                if key in fusion_results and len(fusion_results[key].get("fold_results", [])) == N_FOLDS:
                    completed += N_FOLDS

    print(f"Progress: {completed}/{total_experiments} experiments already completed\n")

    # Train all models
    for backbone in BACKBONES:
        print(f"\n{'='*80}")
        print(f"BACKBONE: {backbone.upper()}")
        print(f"{'='*80}")

        for fusion_type in FUSION_TYPES:
            key = f"{backbone}_{fusion_type}"

            # Initialize results
            if fusion_type == "baseline":
                if key not in baseline_results:
                    baseline_results[key] = {"fold_results": []}
                fold_results = baseline_results[key]["fold_results"]
            else:
                if key not in fusion_results:
                    fusion_results[key] = {"fold_results": []}
                fold_results = fusion_results[key]["fold_results"]

            start_fold = len(fold_results)

            if start_fold >= N_FOLDS:
                print(f"  Skipping {backbone} - {fusion_type.capitalize()} (already completed)")
                continue

            print(f"\n  Training: {backbone.upper()} + {fusion_type.upper()}")

            # Create CV splits
            if fusion_type == "baseline":
                splits = create_cv_splits_single(labels_csv, n_splits=N_FOLDS)
            else:
                splits = create_cv_splits_paired(labels_csv, n_splits=N_FOLDS)

            # Train remaining folds
            for fold_idx in range(start_fold, N_FOLDS):
                train_df, val_df = splits[fold_idx]

                try:
                    start_time = time.time()

                    if fusion_type == "baseline":
                        mae = train_baseline_experiment(backbone, fold_idx, train_df, val_df, device)
                    else:
                        mae = train_fusion_experiment(backbone, fusion_type, fold_idx, train_df, val_df, device)

                    elapsed = time.time() - start_time

                    # Save result
                    fold_results.append(mae)
                    if fusion_type == "baseline":
                        baseline_results[key]["fold_results"] = fold_results
                        baseline_results[key]["mean"] = np.mean(fold_results)
                        baseline_results[key]["std"] = np.std(fold_results)
                        save_baseline_progress(baseline_results)
                    else:
                        fusion_results[key]["fold_results"] = fold_results
                        fusion_results[key]["mean"] = np.mean(fold_results)
                        fusion_results[key]["std"] = np.std(fold_results)
                        save_fusion_progress(fusion_results)

                    completed += 1
                    remaining = total_experiments - completed

                    print(f"\n      Fold {fold_idx+1} complete: MAE = {mae:.4f} ({elapsed/60:.1f} min)")
                    print(f"      Progress: {completed}/{total_experiments} ({completed/total_experiments*100:.1f}%)")
                    if remaining > 0:
                        print(f"      Estimated remaining time: {remaining * elapsed / 3600:.1f} hours")

                except Exception as e:
                    print(f"\n      [FAILED] Fold {fold_idx+1} failed: {e}")
                    continue

    print("\n" + "="*80)
    print("ALL MODELS TRAINED!")
    print("="*80)
    print(f"\nResults saved to:")
    print(f"  - Baseline: Results/baseline_cv_results.json")
    print(f"  - Fusion: Results/comparison/csv/progress.json")
    print(f"\nNext steps:")
    print(f"  1. Run Scripts/evaluate_all_models.py to compare all 27 configurations")
    print(f"  2. Run Scripts/evaluate_best_model.py for detailed best model analysis")


if __name__ == "__main__":
    main()
