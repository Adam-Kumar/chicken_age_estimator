"""train_custom.py
Train a specific model configuration with command-line arguments.

Usage:
    python Scripts/train_custom.py --backbone convnext_t --fusion feature
    python Scripts/train_custom.py --backbone resnet50 --fusion late
    python Scripts/train_custom.py --backbone efficientnet_b0 --fusion baseline

Arguments:
    --backbone: One of [efficientnet_b0, resnet18, resnet50, resnet101,
                        vit_b_16, swin_t, swin_b, convnext_t, convnext_b]
    --fusion: One of [baseline, late, feature]
    --folds: Number of CV folds (default: 3)
    --epochs: Number of epochs (default: 30)
    --batch_size: Batch size (default: 8)

Output:
    - Checkpoint: checkpoints/{backbone}_{fusion}_best.pth
    - Training curves: Results/custom_training/graphs/training_curves_{backbone}_{fusion}.png
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

from Model.model import BaselineFusionRegressor, LateFusionRegressor, FeatureFusionRegressor
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
    total_loss = 0.0
    total_mae = 0.0
    num_batches = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        preds = model(x)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_mae += (preds - y).abs().mean().item()
        num_batches += 1

    return total_loss / num_batches, total_mae / num_batches


def train_one_epoch_fusion(model, loader, criterion, optimizer, device):
    """Train fusion model for one epoch."""
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
    print(f"Found {len(unique_chickens)} unique chickens for CV")

    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits = []

    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(unique_chickens)):
        train_chickens = unique_chickens[train_idx]
        val_chickens = unique_chickens[val_idx]

        train_df = df[df['piece_id'].isin(train_chickens)]
        val_df = df[df['piece_id'].isin(val_chickens)]

        print(f"Fold {fold_idx + 1}: Train={len(train_df)} samples ({len(train_chickens)} chickens), "
              f"Val={len(val_df)} samples ({len(val_chickens)} chickens)")

        splits.append((train_df, val_df))

    return splits


def create_cv_splits_paired(labels_csv, n_splits=3, random_state=42):
    """Create K-fold splits for paired view (TOP+SIDE)."""
    df = pd.read_csv(labels_csv)
    df_top = df[df['view'].str.upper() == 'TOP VIEW'].copy()

    unique_chickens = df_top['piece_id'].unique()
    print(f"Found {len(unique_chickens)} unique chickens for CV")

    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits = []

    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(unique_chickens)):
        train_chickens = unique_chickens[train_idx]
        val_chickens = unique_chickens[val_idx]

        train_df = df_top[df_top['piece_id'].isin(train_chickens)]
        val_df = df_top[df_top['piece_id'].isin(val_chickens)]

        print(f"Fold {fold_idx + 1}: Train={len(train_df)} samples ({len(train_chickens)} chickens), "
              f"Val={len(val_df)} samples ({len(val_chickens)} chickens)")

        splits.append((train_df, val_df))

    return splits


def train_one_fold(args, fold_idx, train_df, val_df, device):
    """Train one fold."""
    print(f"\n{'='*80}")
    print(f"Fold {fold_idx + 1}/{args.folds}")
    print(f"{'='*80}")

    # Create model
    if args.fusion == "baseline":
        model = BaselineFusionRegressor(backbone_name=args.backbone, pretrained=True).to(device)
    elif args.fusion == "late":
        model = LateFusionRegressor(backbone_name=args.backbone, pretrained=True).to(device)
    else:  # feature
        model = FeatureFusionRegressor(backbone_name=args.backbone, pretrained=True).to(device)

    # Save temp CSVs
    temp_dir = project_root / "Scripts" / "temp_custom"
    temp_dir.mkdir(exist_ok=True)

    train_csv = temp_dir / f"train_{args.backbone}_{args.fusion}_f{fold_idx}.csv"
    val_csv = temp_dir / f"val_{args.backbone}_{args.fusion}_f{fold_idx}.csv"
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)

    # Data loaders (all fusion types now use paired dataset)
    root_dir = project_root / "Dataset_Processed"
    train_ds = ChickenAgePairedDataset(train_csv, root_dir, get_default_transforms(True))
    val_ds = ChickenAgePairedDataset(val_csv, root_dir, get_default_transforms(False))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Training setup
    criterion = nn.MSELoss()
    lr = get_lr_for_backbone(args.backbone)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Track history
    train_losses = []
    train_maes = []
    val_maes = []
    best_val_mae = float("inf")
    best_epoch = 0
    best_model_state = None

    for epoch in range(1, args.epochs + 1):
        # All models now use paired views (baseline, late, feature)
        train_loss, train_mae = train_one_epoch_fusion(model, train_loader, criterion, optimizer, device)
        val_mae = validate_fusion(model, val_loader, device)

        scheduler.step()

        train_losses.append(train_loss)
        train_maes.append(train_mae)
        val_maes.append(val_mae)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch
            best_model_state = model.state_dict().copy()

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:2d}/{args.epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.4f}, "
                  f"Val MAE: {val_mae:.4f} (Best: {best_val_mae:.4f} @ epoch {best_epoch})")

    print(f"\nFold {fold_idx + 1} complete - Best Val MAE: {best_val_mae:.4f} (epoch {best_epoch})")

    return best_val_mae, (train_losses, train_maes, val_maes), best_model_state


def plot_training_curves(args, all_fold_histories):
    """Plot training curves for all folds."""
    results_dir = project_root / "Results" / "custom_training" / "graphs"
    results_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    colors = ['steelblue', 'coral', 'mediumseagreen', 'gold', 'purple']

    for fold_idx, (train_losses, train_maes, val_maes) in enumerate(all_fold_histories):
        epochs = range(1, len(train_losses) + 1)
        color = colors[fold_idx % len(colors)]

        # Plot loss
        axes[0].plot(epochs, train_losses, color=color, alpha=0.7,
                    label=f'Fold {fold_idx+1}', linewidth=2)
        axes[0].set_xlabel('Epoch', fontweight='bold')
        axes[0].set_ylabel('MSE Loss', fontweight='bold')
        axes[0].set_title('Training Loss', fontweight='bold', fontsize=14)
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # Plot train MAE
        axes[1].plot(epochs, train_maes, color=color, alpha=0.7,
                    label=f'Fold {fold_idx+1}', linewidth=2)
        axes[1].set_xlabel('Epoch', fontweight='bold')
        axes[1].set_ylabel('MAE (days)', fontweight='bold')
        axes[1].set_title('Training MAE', fontweight='bold', fontsize=14)
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        # Plot val MAE
        axes[2].plot(epochs, val_maes, color=color, alpha=0.7,
                    label=f'Fold {fold_idx+1}', linewidth=2)
        axes[2].set_xlabel('Epoch', fontweight='bold')
        axes[2].set_ylabel('MAE (days)', fontweight='bold')
        axes[2].set_title('Validation MAE', fontweight='bold', fontsize=14)
        axes[2].legend()
        axes[2].grid(alpha=0.3)

    plt.suptitle(f'{args.backbone.upper()} {args.fusion.capitalize()} Fusion - Training Curves ({args.folds}-Fold CV)',
                fontweight='bold', fontsize=16, y=1.02)
    plt.tight_layout()

    filename = f"training_curves_{args.backbone}_{args.fusion}.png"
    plt.savefig(results_dir / filename, dpi=150, bbox_inches='tight')
    print(f"\nSaved training curves to: {results_dir / filename}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Train a specific model configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python Scripts/train_custom.py --backbone convnext_t --fusion feature
  python Scripts/train_custom.py --backbone resnet50 --fusion late --folds 5
  python Scripts/train_custom.py --backbone efficientnet_b0 --fusion baseline --epochs 50
        """
    )

    parser.add_argument("--backbone", type=str, required=True, choices=BACKBONES,
                       help="Backbone architecture")
    parser.add_argument("--fusion", type=str, required=True, choices=FUSION_TYPES,
                       help="Fusion strategy")
    parser.add_argument("--folds", type=int, default=3,
                       help="Number of CV folds (default: 3)")
    parser.add_argument("--epochs", type=int, default=30,
                       help="Number of epochs (default: 30)")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size (default: 8)")

    args = parser.parse_args()

    print("="*80)
    print(f"CUSTOM MODEL TRAINING: {args.backbone.upper()} + {args.fusion.upper()}")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Backbone: {args.backbone}")
    print(f"  Fusion: {args.fusion}")
    print(f"  Folds: {args.folds}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {get_lr_for_backbone(args.backbone)} (auto-selected)")
    print("="*80)

    # Prepare data
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

    # Create CV splits (all fusion types use paired views now)
    splits = create_cv_splits_paired(labels_csv, n_splits=args.folds)

    # Train all folds
    fold_maes = []
    fold_histories = []
    best_overall_state = None
    best_overall_mae = float("inf")

    total_start_time = time.time()

    for fold_idx in range(args.folds):
        train_df, val_df = splits[fold_idx]

        fold_start = time.time()
        best_mae, history, model_state = train_one_fold(args, fold_idx, train_df, val_df, device)
        fold_elapsed = time.time() - fold_start

        fold_maes.append(best_mae)
        fold_histories.append(history)

        if best_mae < best_overall_mae:
            best_overall_mae = best_mae
            best_overall_state = model_state

        print(f"Fold {fold_idx + 1} time: {fold_elapsed/60:.1f} minutes")

    total_elapsed = time.time() - total_start_time

    # Save best model checkpoint
    checkpoint_dir = project_root / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{args.backbone}_{args.fusion}_best.pth"

    torch.save(best_overall_state, checkpoint_path)
    print(f"\nSaved best model checkpoint to: {checkpoint_path}")

    # Plot training curves
    plot_training_curves(args, fold_histories)

    # Print final results
    mean_mae = np.mean(fold_maes)
    std_mae = np.std(fold_maes)
    min_mae = np.min(fold_maes)
    max_mae = np.max(fold_maes)

    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"\nCross-Validation Results ({args.folds}-Fold):")
    print(f"  Fold MAEs: {[f'{mae:.4f}' for mae in fold_maes]}")
    print(f"  Mean MAE: {mean_mae:.4f} ± {std_mae:.4f}")
    print(f"  Min MAE:  {min_mae:.4f}")
    print(f"  Max MAE:  {max_mae:.4f}")
    print(f"\nTotal training time: {total_elapsed/3600:.2f} hours")
    print(f"\nOutputs:")
    print(f"  - Model checkpoint: checkpoints/{args.backbone}_{args.fusion}_best.pth")
    print(f"  - Training curves: Results/custom_training/graphs/training_curves_{args.backbone}_{args.fusion}.png")


if __name__ == "__main__":
    main()
