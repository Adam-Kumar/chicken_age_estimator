"""dataset.py
PyTorch Dataset and DataLoader helpers for the chicken drumette age regression task.
"""

from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode

__all__ = [
    "ChickenAgeDataset",
    "ChickenAgePairedDataset",
    "get_default_transforms",
    "create_dataloaders",
    "create_dataloaders_fusion",
]


class ChickenAgeDataset(Dataset):
    """Dataset that loads images and day labels from a CSV."""

    def __init__(
        self,
        csv_file: str | Path,
        root_dir: str | Path = "Dataset_Processed",
        transforms: Optional[Callable] = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.df = pd.read_csv(csv_file)
        if "relative_path" not in self.df.columns or "day" not in self.df.columns:
            raise ValueError("CSV must contain relative_path and day columns")
        self.transforms = transforms

    def __len__(self) -> int:  # noqa: D401
        return len(self.df)

    def __getitem__(self, idx: int) -> Any:  # noqa: D401
        row = self.df.iloc[idx]
        img_path = self.root_dir / row["relative_path"]
        image = Image.open(img_path).convert("RGB")
        label = torch.tensor(float(row["day"]), dtype=torch.float32)
        if self.transforms:
            image = self.transforms(image)
        return image, label


# ------------------------------ transforms ------------------------------------

def get_default_transforms(train=True):
    if train:
        return T.Compose([
            # Slight random crop & resize (90-100 % of original)
            T.RandomResizedCrop(
                224, scale=(0.9, 1.0), ratio=(0.95, 1.05),
                interpolation=InterpolationMode.BILINEAR
            ),
            T.RandomHorizontalFlip(),
            T.RandomApply([
                T.RandomRotation(10, interpolation=InterpolationMode.BILINEAR)
            ], p=0.3),
            T.ColorJitter(
                brightness=0.1, contrast=0.1,
                saturation=0.1, hue=0.02
            ),
            T.RandomApply([T.GaussianBlur(5, sigma=(0.1, 0.5))], p=0.2),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
    else:
        return T.Compose([
            T.Resize(224),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])


# --------------------------- dataloader helper --------------------------------

def create_dataloaders(
    train_csv: str | Path,
    val_csv: str | Path,
    test_csv: str | Path,
    root_dir: str | Path = "Dataset_Processed",
    batch_size: int = 32,
    num_workers: int = 4,
):
    train_ds = ChickenAgeDataset(train_csv, root_dir, get_default_transforms(True))
    val_ds = ChickenAgeDataset(val_csv, root_dir, get_default_transforms(False))
    test_ds = ChickenAgeDataset(test_csv, root_dir, get_default_transforms(False))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader 


# -----------------------------------------------------------------------------
# Paired dataset for feature-fusion (TOP + SIDE views)
# -----------------------------------------------------------------------------


class ChickenAgePairedDataset(Dataset):
    """Returns (img_top, img_side), day label."""

    def __init__(
        self,
        csv_file: str | Path,
        root_dir: str | Path = "Dataset_Processed",
        transforms: Optional[Callable] = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.transforms = transforms
        df = pd.read_csv(csv_file)
        # Keep only top view rows, assume matching side exists
        self.df_top = df[df["view"].str.upper() == "TOP VIEW"].reset_index(drop=True)
        if self.df_top.empty:
            raise ValueError("CSV must contain TOP VIEW rows for paired dataset")

    def __len__(self) -> int:
        return len(self.df_top)

    def _side_path_from_top(self, rel_path: str) -> str:
        return rel_path.replace("TOP VIEW", "SIDE VIEW")

    def __getitem__(self, idx: int):
        row = self.df_top.iloc[idx]
        top_path = self.root_dir / row["relative_path"]
        side_relative = self._side_path_from_top(row["relative_path"])
        side_path = self.root_dir / side_relative

        img_top = Image.open(top_path).convert("RGB")
        img_side = Image.open(side_path).convert("RGB")

        if self.transforms:
            img_top = self.transforms(img_top)
            img_side = self.transforms(img_side)

        label = torch.tensor(float(row["day"]), dtype=torch.float32)
        return (img_top, img_side), label


def create_dataloaders_fusion(
    train_csv: str | Path,
    val_csv: str | Path,
    test_csv: str | Path,
    root_dir: str | Path = "Dataset_Processed",
    batch_size: int = 16,
    num_workers: int = 4,
):
    train_ds = ChickenAgePairedDataset(train_csv, root_dir, get_default_transforms(True))
    val_ds = ChickenAgePairedDataset(val_csv, root_dir, get_default_transforms(False))
    test_ds = ChickenAgePairedDataset(test_csv, root_dir, get_default_transforms(False))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader 