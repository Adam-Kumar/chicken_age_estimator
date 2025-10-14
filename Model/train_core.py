"""train.py
Training script for chicken drumette age regression.

Usage:
    Simply run this script directly in your IDE:
    
    ```
    python train.py
    ```
    
    Or customize parameters by editing the DEFAULT_* constants below.

Outputs:
- checkpoints/   best.pth  (best on val MAE)
- prints training/validation metrics per epoch
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Tuple, List

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Ensure the parent directory is on PYTHONPATH when running as a script
# so that `Model` can be imported when executed via `python Model/train.py`.
import sys
from pathlib import Path as _Path

_project_root = _Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from Model.dataset import create_dataloaders  # noqa: E402
from Model.model import build_resnet_regressor  # noqa: E402

# Default parameters - edit these directly instead of using command line args
DEFAULT_TRAIN_CSV = str(_project_root / "Labels" / "train.csv")
DEFAULT_VAL_CSV = str(_project_root / "Labels" / "val.csv")
DEFAULT_TEST_CSV = str(_project_root / "Labels" / "test.csv")
DEFAULT_ROOT_DIR = str(_project_root / "Dataset_Processed")
DEFAULT_EPOCHS = 30
DEFAULT_BATCH_SIZE = 32
DEFAULT_LR = 1e-4
DEFAULT_WEIGHT_DECAY = 1e-2
DEFAULT_FREEZE_BACKBONE = False
DEFAULT_CHECKPOINT_DIR = str(_project_root / "Model" / "checkpoints")
DEFAULT_SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser(description="Train chicken age regressor")
    parser.add_argument("--train_csv", default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--val_csv", default=DEFAULT_VAL_CSV)
    parser.add_argument("--test_csv", default=DEFAULT_TEST_CSV)
    parser.add_argument("--root_dir", default=DEFAULT_ROOT_DIR)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--freeze_backbone", action="store_true", default=DEFAULT_FREEZE_BACKBONE)
    parser.add_argument("--model_type", type=str, default="baseline",
                        choices=["baseline", "late_fusion", "feature_fusion"],
                        help="Model architecture: baseline (single view), late_fusion (avg predictions), feature_fusion (concat features)")
    parser.add_argument("--fusion", action="store_true", help="[DEPRECATED] Use --model_type feature_fusion instead")
    parser.add_argument("--checkpoint_dir", default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def set_seed(seed: int):
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:  # noqa: D401, D413
    return (pred - target).abs().mean()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimiser: torch.optim.Optimizer,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    running_mae = 0.0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimiser.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimiser.step()

        running_loss += loss.item() * images.size(0)
        running_mae += mae(outputs, labels).item() * images.size(0)

    n = len(loader.dataset)  # type: ignore[arg-type]
    return running_loss / n, running_mae / n


# ---------------- fusion helpers -----------------


def train_one_epoch_fusion(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimiser: torch.optim.Optimizer,
):
    model.train()
    running_loss = 0.0
    running_mae = 0.0
    for (img_top, img_side), labels in loader:
        img_top, img_side, labels = img_top.to(DEVICE), img_side.to(DEVICE), labels.to(DEVICE)
        optimiser.zero_grad()
        outputs = model(img_top, img_side)
        loss = criterion(outputs, labels)
        loss.backward()
        optimiser.step()

        running_loss += loss.item() * labels.size(0)
        running_mae += mae(outputs, labels).item() * labels.size(0)

    n = len(loader.dataset)  # type: ignore[arg-type]
    return running_loss / n, running_mae / n


def evaluate_fusion(model: nn.Module, loader: DataLoader, criterion: nn.Module):
    model.eval()
    running_loss = 0.0
    running_mae = 0.0
    with torch.no_grad():
        for (img_top, img_side), labels in loader:
            img_top, img_side, labels = img_top.to(DEVICE), img_side.to(DEVICE), labels.to(DEVICE)
            outputs = model(img_top, img_side)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * labels.size(0)
            running_mae += mae(outputs, labels).item() * labels.size(0)

    n = len(loader.dataset)  # type: ignore[arg-type]
    return running_loss / n, running_mae / n


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    running_mae = 0.0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            running_mae += mae(outputs, labels).item() * images.size(0)

    n = len(loader.dataset)  # type: ignore[arg-type]
    return running_loss / n, running_mae / n


def main():
    # You can either use command line args or just edit the DEFAULT_* constants above
    args = parse_args()
    set_seed(args.seed)

    # Handle deprecated --fusion flag
    if args.fusion:
        print("Warning: --fusion flag is deprecated. Use --model_type feature_fusion instead.")
        args.model_type = "feature_fusion"

    cpu_cnt = os.cpu_count() or 4
    num_workers = max(2, cpu_cnt // 2)

    # Determine if we need fusion dataloaders (for late_fusion and feature_fusion)
    use_fusion_data = args.model_type in ["late_fusion", "feature_fusion"]

    if use_fusion_data:
        from Model.dataset import create_dataloaders_fusion  # noqa: E402
        from Model.model import FeatureFusionRegressor, LateFusionRegressor  # noqa: E402

        train_loader, val_loader, test_loader = create_dataloaders_fusion(
            args.train_csv,
            args.val_csv,
            args.test_csv,
            root_dir=args.root_dir,
            batch_size=args.batch_size,
            num_workers=num_workers,
        )

        if args.model_type == "late_fusion":
            model: nn.Module = LateFusionRegressor(pretrained=True, freeze_backbone=args.freeze_backbone)
            print("Training LateFusionRegressor (averaging predictions from TOP and SIDE views)")
        else:  # feature_fusion
            model = FeatureFusionRegressor(pretrained=True, freeze_backbone=args.freeze_backbone)
            print("Training FeatureFusionRegressor (concatenating features from TOP and SIDE views)")
    else:
        train_loader, val_loader, test_loader = create_dataloaders(
            args.train_csv,
            args.val_csv,
            args.test_csv,
            root_dir=args.root_dir,
            batch_size=args.batch_size,
            num_workers=num_workers,
        )
        model = build_resnet_regressor(pretrained=True, freeze_backbone=args.freeze_backbone)
        print("Training ResNetRegressor (baseline single-view model)")
    model.to(DEVICE)

    criterion = nn.L1Loss()
    optimiser = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimiser, T_max=args.epochs)

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    best_val_mae = float("inf")
    history_train: List[float] = []
    history_val: List[float] = []

    print(f"Training on {DEVICE}...")
    print(f"Dataset sizes: Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")  # type: ignore[arg-type]

    for epoch in range(1, args.epochs + 1):
        if use_fusion_data:
            train_loss, train_mae = train_one_epoch_fusion(model, train_loader, criterion, optimiser)
            val_loss, val_mae = evaluate_fusion(model, val_loader, criterion)
        else:
            train_loss, train_mae = train_one_epoch(model, train_loader, criterion, optimiser)
            val_loss, val_mae = evaluate(model, val_loader, criterion)
        scheduler.step()

        history_train.append(train_mae)
        history_val.append(val_mae)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss {train_loss:.4f} MAE {train_mae:.3f} | "
            f"val loss {val_loss:.4f} MAE {val_mae:.3f}"
        )

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            # Use model-specific checkpoint names
            ckpt_name = f"best_{args.model_type}.pth"
            ckpt_path = Path(args.checkpoint_dir) / ckpt_name
            # Safety: ensure directory exists (handles edge cases with relative paths)
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state": model.state_dict(),
                "model_type": args.model_type,
                "epoch": epoch,
                "val_mae": val_mae
            }, ckpt_path)
            print(f"  -> New best MAE. Checkpoint saved to {ckpt_path}")

    # Plot training curve
    plt.figure(figsize=(6,4))
    plt.plot(range(1, args.epochs+1), history_train, label="Train MAE")
    plt.plot(range(1, args.epochs+1), history_val, label="Val MAE")
    plt.xlabel("Epoch")
    plt.ylabel("MAE (days)")
    plt.title(f"Training / Validation MAE ({args.model_type})")
    plt.legend()
    metrics_plot_path = _project_root / "Results" / "training_curves" / f"train_val_mae_{args.model_type}.png"
    metrics_plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(metrics_plot_path)
    print(f"Training curve saved to {metrics_plot_path}")

    # Evaluate best model on test set
    ckpt_name = f"best_{args.model_type}.pth"
    ckpt = torch.load(Path(args.checkpoint_dir) / ckpt_name, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    if use_fusion_data:
        test_loss, test_mae = evaluate_fusion(model, test_loader, criterion)
    else:
        test_loss, test_mae = evaluate(model, test_loader, criterion)
    print(f"Test  loss {test_loss:.4f}  MAE {test_mae:.3f} (best epoch {ckpt['epoch']})")


if __name__ == "__main__":
    main() 