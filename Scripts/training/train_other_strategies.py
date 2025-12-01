"""train_other_strategies.py
Train ConvNeXt-T with alternative fusion strategies for comparison with best model.

Trains 4 strategies on 70:15:15 split:
- TOP view only (single-view)
- SIDE view only (single-view)
- Baseline (view-agnostic: TOP+SIDE mixed)
- Late Fusion (view-aware: separate models, averaged predictions)

All use ConvNeXt-T backbone for fair comparison with Feature Fusion (best model).
Feature Fusion is trained separately in train_best_model.py.

Output:
- Checkpoints: checkpoints/convnext_t_{strategy}_holdout.pth

For evaluation, run Scripts/evaluating/evaluate_other_strategies.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import pandas as pd

from Model.model import ResNetRegressor, LateFusionRegressor
from Model.dataset import ChickenAgeDataset, ChickenAgePairedDataset, get_default_transforms
from torch.utils.data import DataLoader


# Configuration
BACKBONE = "convnext_t"
EPOCHS = 30
BATCH_SIZE = 8
LR = 8e-5
WEIGHT_DECAY = 1e-2


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




def train_single_view(strategy_name, view_filter, train_csv, val_csv, test_csv, device):
    """Train single-view strategy (TOP or SIDE only)."""
    print(f"\n{'='*80}")
    print(f"Training: {strategy_name}")
    print(f"{'='*80}\n")

    # Create model
    model = ResNetRegressor(backbone_name=BACKBONE, pretrained=True).to(device)

    # Load data
    root_dir = project_root / "Dataset_Processed"

    # Filter for specific view
    train_df = pd.read_csv(train_csv)
    train_df = train_df[train_df['view'].str.upper() == view_filter.upper()]
    val_df = pd.read_csv(val_csv)
    val_df = val_df[val_df['view'].str.upper() == view_filter.upper()]
    test_df = pd.read_csv(test_csv)
    test_df = test_df[test_df['view'].str.upper() == view_filter.upper()]

    # Save filtered CSVs
    temp_dir = project_root / "Scripts" / "temp_other_strategies"
    temp_dir.mkdir(exist_ok=True)

    filtered_train_csv = temp_dir / f"train_{strategy_name.lower().replace(' ', '_')}.csv"
    filtered_val_csv = temp_dir / f"val_{strategy_name.lower().replace(' ', '_')}.csv"
    filtered_test_csv = temp_dir / f"test_{strategy_name.lower().replace(' ', '_')}.csv"

    train_df.to_csv(filtered_train_csv, index=False)
    val_df.to_csv(filtered_val_csv, index=False)
    test_df.to_csv(filtered_test_csv, index=False)

    train_ds = ChickenAgeDataset(filtered_train_csv, root_dir, get_default_transforms(True))
    val_ds = ChickenAgeDataset(filtered_val_csv, root_dir, get_default_transforms(False))
    test_ds = ChickenAgeDataset(filtered_test_csv, root_dir, get_default_transforms(False))

    print(f"Train: {len(train_ds)} samples")
    print(f"Val: {len(val_ds)} samples")
    print(f"Test: {len(test_ds)} samples\n")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Training setup
    criterion = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_mae = float("inf")
    best_epoch = 0
    best_model_state = None

    # Training loop
    print("Training...")
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        train_mae = train_one_epoch_single(model, train_loader, criterion, optimizer, device)
        val_mae = validate_single(model, val_loader, device)
        scheduler.step()

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch
            best_model_state = model.state_dict().copy()

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:2d}/{EPOCHS} - Train MAE: {train_mae:.4f}, Val MAE: {val_mae:.4f} (Best: {best_val_mae:.4f} @ epoch {best_epoch})")

    training_time = time.time() - start_time

    print(f"\nTraining complete - Best Val MAE: {best_val_mae:.4f} (epoch {best_epoch})")
    print(f"Training time: {training_time/60:.1f} minutes")

    # Save checkpoint
    checkpoint_dir = project_root / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint_name = f"convnext_t_{strategy_name.lower().replace(' ', '_')}_holdout.pth"
    checkpoint_path = checkpoint_dir / checkpoint_name
    torch.save(best_model_state, checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")


def train_baseline(train_csv, val_csv, test_csv, device):
    """Train baseline view-agnostic strategy (TOP+SIDE mixed)."""
    print(f"\n{'='*80}")
    print(f"Training: Baseline (View-Agnostic)")
    print(f"{'='*80}\n")

    # Create model
    model = ResNetRegressor(backbone_name=BACKBONE, pretrained=True).to(device)

    # Load data (all views mixed)
    root_dir = project_root / "Dataset_Processed"

    train_ds = ChickenAgeDataset(train_csv, root_dir, get_default_transforms(True))
    val_ds = ChickenAgeDataset(val_csv, root_dir, get_default_transforms(False))
    test_ds = ChickenAgeDataset(test_csv, root_dir, get_default_transforms(False))

    print(f"Train: {len(train_ds)} samples")
    print(f"Val: {len(val_ds)} samples")
    print(f"Test: {len(test_ds)} samples\n")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Training setup
    criterion = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_mae = float("inf")
    best_epoch = 0
    best_model_state = None

    # Training loop
    print("Training...")
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        train_mae = train_one_epoch_single(model, train_loader, criterion, optimizer, device)
        val_mae = validate_single(model, val_loader, device)
        scheduler.step()

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch
            best_model_state = model.state_dict().copy()

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:2d}/{EPOCHS} - Train MAE: {train_mae:.4f}, Val MAE: {val_mae:.4f} (Best: {best_val_mae:.4f} @ epoch {best_epoch})")

    training_time = time.time() - start_time

    print(f"\nTraining complete - Best Val MAE: {best_val_mae:.4f} (epoch {best_epoch})")
    print(f"Training time: {training_time/60:.1f} minutes")

    # Save checkpoint
    checkpoint_dir = project_root / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint_path = checkpoint_dir / "convnext_t_baseline_holdout.pth"
    torch.save(best_model_state, checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")


def train_late_fusion(train_csv, val_csv, test_csv, device):
    """Train late fusion strategy (separate models, averaged predictions)."""
    print(f"\n{'='*80}")
    print(f"Training: Late Fusion")
    print(f"{'='*80}\n")

    # Create model
    model = LateFusionRegressor(backbone_name=BACKBONE, pretrained=True).to(device)

    # Load data (paired TOP+SIDE)
    root_dir = project_root / "Dataset_Processed"

    train_ds = ChickenAgePairedDataset(train_csv, root_dir, get_default_transforms(True))
    val_ds = ChickenAgePairedDataset(val_csv, root_dir, get_default_transforms(False))
    test_ds = ChickenAgePairedDataset(test_csv, root_dir, get_default_transforms(False))

    print(f"Train: {len(train_ds)} samples")
    print(f"Val: {len(val_ds)} samples")
    print(f"Test: {len(test_ds)} samples\n")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Training setup
    criterion = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_mae = float("inf")
    best_epoch = 0
    best_model_state = None

    # Training loop
    print("Training...")
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        train_mae = train_one_epoch_late(model, train_loader, criterion, optimizer, device)
        val_mae = validate_late(model, val_loader, device)
        scheduler.step()

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch
            best_model_state = model.state_dict().copy()

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:2d}/{EPOCHS} - Train MAE: {train_mae:.4f}, Val MAE: {val_mae:.4f} (Best: {best_val_mae:.4f} @ epoch {best_epoch})")

    training_time = time.time() - start_time

    print(f"\nTraining complete - Best Val MAE: {best_val_mae:.4f} (epoch {best_epoch})")
    print(f"Training time: {training_time/60:.1f} minutes")

    # Save checkpoint
    checkpoint_dir = project_root / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint_path = checkpoint_dir / "convnext_t_late_fusion_holdout.pth"
    torch.save(best_model_state, checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")


def main():
    print("="*80)
    print("TRAINING OTHER STRATEGIES: ConvNeXt-T (70:15:15 Split)")
    print("="*80)
    print(f"\nStrategies to train:")
    print(f"  1. TOP view only")
    print(f"  2. SIDE view only")
    print(f"  3. Baseline (view-agnostic)")
    print(f"  4. Late Fusion (view-aware)")
    print(f"\nConfiguration:")
    print(f"  Backbone: {BACKBONE.upper()}")
    print(f"  Split: 70% train, 15% val, 15% test")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Estimated time: 2-3 hours")
    print("="*80)

    # Load data splits
    train_csv = project_root / "Labels" / "train.csv"
    val_csv = project_root / "Labels" / "val.csv"
    test_csv = project_root / "Labels" / "test.csv"

    if not all([train_csv.exists(), val_csv.exists(), test_csv.exists()]):
        raise FileNotFoundError("train.csv, val.csv, or test.csv not found in Labels/")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}\n")

    # Train TOP view only
    train_single_view("TOP View", "TOP VIEW", train_csv, val_csv, test_csv, device)

    # Train SIDE view only
    train_single_view("SIDE View", "SIDE VIEW", train_csv, val_csv, test_csv, device)

    # Train Baseline (view-agnostic)
    train_baseline(train_csv, val_csv, test_csv, device)

    # Train Late Fusion
    train_late_fusion(train_csv, val_csv, test_csv, device)

    # Print final summary
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print("\nCheckpoints saved:")
    print("  - checkpoints/convnext_t_top_view_holdout.pth")
    print("  - checkpoints/convnext_t_side_view_holdout.pth")
    print("  - checkpoints/convnext_t_baseline_holdout.pth")
    print("  - checkpoints/convnext_t_late_fusion_holdout.pth")
    print("\nNext steps:")
    print("  - Run Scripts/evaluating/evaluate_best_model.py to evaluate Feature Fusion")
    print("  - Run Scripts/evaluating/compare_best_model.py to compare all 5 strategies")


if __name__ == "__main__":
    main()
