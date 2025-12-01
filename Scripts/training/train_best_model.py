"""train_best_model.py
Train the best model: ConvNeXt-T Feature Fusion with held-out test set.

This script trains the champion model using the original 70:15:15 split:
- Backbone: ConvNeXt-T
- Fusion: Feature Fusion (concatenate features from TOP+SIDE views)
- Train: 70% of chickens (for training)
- Val: 15% of chickens (for validation/early stopping)
- Test: 15% of chickens (held-out, not used during training)

Output:
- Checkpoint: checkpoints/convnext_t_feature_holdout.pth

For evaluation, run Scripts/evaluating/evaluate_best_model.py
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

from Model.model import FeatureFusionRegressor
from Model.dataset import ChickenAgePairedDataset, get_default_transforms
from torch.utils.data import DataLoader


# Configuration
BACKBONE = "convnext_t"
EPOCHS = 30
BATCH_SIZE = 8
LR = 8e-5
WEIGHT_DECAY = 1e-2


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_mae = 0.0
    num_batches = 0

    for (x_top, x_side), y in loader:
        x_top, x_side, y = x_top.to(device), x_side.to(device), y.to(device)

        optimizer.zero_grad()
        preds = model(x_top, x_side)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_mae += (preds - y).abs().mean().item()
        num_batches += 1

    return total_loss / num_batches, total_mae / num_batches


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



def main():
    print("="*80)
    print("TRAINING BEST MODEL: ConvNeXt-T Feature Fusion (70:15:15 Split)")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Backbone: {BACKBONE.upper()}")
    print(f"  Fusion: Feature Fusion (TOP+SIDE)")
    print(f"  Split: 70% train, 15% val, 15% test (held-out)")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Learning rate: {LR}")
    print(f"  Batch size: {BATCH_SIZE}")
    print("="*80)

    # Load data splits
    train_csv = project_root / "Labels" / "train.csv"
    val_csv = project_root / "Labels" / "val.csv"
    test_csv = project_root / "Labels" / "test.csv"

    if not all([train_csv.exists(), val_csv.exists(), test_csv.exists()]):
        raise FileNotFoundError("train.csv, val.csv, or test.csv not found in Labels/")

    # Count chickens in each split
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    train_chickens = train_df[train_df['view'].str.upper() == 'TOP VIEW']['piece_id'].nunique()
    val_chickens = val_df[val_df['view'].str.upper() == 'TOP VIEW']['piece_id'].nunique()
    test_chickens = test_df[test_df['view'].str.upper() == 'TOP VIEW']['piece_id'].nunique()

    print(f"\nDataset splits (at chicken level):")
    print(f"  Train: {len(train_df)} samples ({train_chickens} chickens)")
    print(f"  Val:   {len(val_df)} samples ({val_chickens} chickens)")
    print(f"  Test:  {len(test_df)} samples ({test_chickens} chickens) - HELD OUT")
    print(f"  Total: {train_chickens + val_chickens + test_chickens} chickens")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}\n")

    # Create datasets
    root_dir = project_root / "Dataset_Processed"
    train_ds = ChickenAgePairedDataset(train_csv, root_dir, get_default_transforms(True))
    val_ds = ChickenAgePairedDataset(val_csv, root_dir, get_default_transforms(False))
    test_ds = ChickenAgePairedDataset(test_csv, root_dir, get_default_transforms(False))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Create model
    model = FeatureFusionRegressor(backbone_name=BACKBONE, pretrained=True).to(device)

    # Training setup
    criterion = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Track history
    train_losses = []
    train_maes = []
    val_maes = []
    best_val_mae = float("inf")
    best_epoch = 0
    best_model_state = None

    print("="*80)
    print("TRAINING")
    print("="*80)

    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_mae = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_mae = validate(model, val_loader, device)
        scheduler.step()

        train_losses.append(train_loss)
        train_maes.append(train_mae)
        val_maes.append(val_mae)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch
            best_model_state = model.state_dict().copy()

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:2d}/{EPOCHS} - "
                  f"Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.4f}, "
                  f"Val MAE: {val_mae:.4f} (Best: {best_val_mae:.4f} @ epoch {best_epoch})")

    training_time = time.time() - start_time

    print(f"\nTraining complete - Best Val MAE: {best_val_mae:.4f} (epoch {best_epoch})")
    print(f"Training time: {training_time/60:.1f} minutes")

    # Save best model checkpoint
    checkpoint_dir = project_root / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint_path = checkpoint_dir / "convnext_t_feature_holdout.pth"

    torch.save(best_model_state, checkpoint_path)
    print(f"\nSaved best model checkpoint to: {checkpoint_path}")

    # Print final summary
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"\nFinal Results:")
    print(f"  Best Validation MAE: {best_val_mae:.4f} days (epoch {best_epoch})")
    print(f"  Training time: {training_time/60:.1f} minutes")
    print(f"\nOutputs:")
    print(f"  - Model checkpoint: {checkpoint_path}")
    print("\nNext steps:")
    print("  - Run Scripts/evaluating/evaluate_best_model.py to evaluate the model")
    print("  - Run Scripts/evaluating/compare_best_model.py to compare with other strategies")


if __name__ == "__main__":
    main()
