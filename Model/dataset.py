"""dataset.py
PyTorch Dataset and DataLoader helpers for the chicken drumette age regression task.
"""

from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd
from PIL import Image
import numpy as np
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


def apply_mask_to_image(image: Image.Image, mask_path: Path, bg_value: int = 255) -> Image.Image:
    """Apply binary mask to image, setting background to bg_value (default white).

    Args:
        image: PIL Image (RGB)
        mask_path: Path to mask file (binary: 255=foreground, 0=background)
        bg_value: Background color (0=black, 255=white)

    Returns:
        Masked PIL Image
    """
    if not mask_path.exists():
        # If mask doesn't exist, return original image
        return image

    # Load mask
    mask = Image.open(mask_path).convert('L')  # Grayscale
    mask_np = np.array(mask)

    # Convert to binary (threshold at 127)
    mask_binary = (mask_np > 127).astype(np.uint8)

    # Apply mask to image
    img_np = np.array(image)
    masked = img_np.copy()

    # Set background pixels to bg_value
    masked[mask_binary == 0] = bg_value

    return Image.fromarray(masked)


class ChickenAgeDataset(Dataset):
    """Dataset that loads images and day labels from a CSV."""

    def __init__(
        self,
        csv_file: str | Path,
        root_dir: str | Path = "Dataset_Processed",
        transforms: Optional[Callable] = None,
        use_masks: bool = False,
        mask_dir: Optional[str | Path] = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.df = pd.read_csv(csv_file)
        if "relative_path" not in self.df.columns or "day" not in self.df.columns:
            raise ValueError("CSV must contain relative_path and day columns")
        self.transforms = transforms
        self.use_masks = use_masks
        self.mask_dir = Path(mask_dir) if mask_dir else Path("Segmentation/masks")

    def __len__(self) -> int:  # noqa: D401
        return len(self.df)

    def __getitem__(self, idx: int) -> Any:  # noqa: D401
        row = self.df.iloc[idx]
        img_path = self.root_dir / row["relative_path"]
        image = Image.open(img_path).convert("RGB")

        # Apply mask if enabled
        if self.use_masks:
            mask_path = self.mask_dir / row["relative_path"]
            image = apply_mask_to_image(image, mask_path, bg_value=255)

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
    use_masks: bool = False,
    mask_dir: Optional[str | Path] = None,
):
    train_ds = ChickenAgeDataset(train_csv, root_dir, get_default_transforms(True), use_masks, mask_dir)
    val_ds = ChickenAgeDataset(val_csv, root_dir, get_default_transforms(False), use_masks, mask_dir)
    test_ds = ChickenAgeDataset(test_csv, root_dir, get_default_transforms(False), use_masks, mask_dir)

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
        use_masks: bool = False,
        mask_dir: Optional[str | Path] = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.transforms = transforms
        self.use_masks = use_masks
        self.mask_dir = Path(mask_dir) if mask_dir else Path("Segmentation/masks")
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

        # Apply masks if enabled
        if self.use_masks:
            mask_top_path = self.mask_dir / row["relative_path"]
            mask_side_path = self.mask_dir / side_relative
            img_top = apply_mask_to_image(img_top, mask_top_path, bg_value=255)
            img_side = apply_mask_to_image(img_side, mask_side_path, bg_value=255)

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
    use_masks: bool = False,
    mask_dir: Optional[str | Path] = None,
):
    train_ds = ChickenAgePairedDataset(train_csv, root_dir, get_default_transforms(True), use_masks, mask_dir)
    val_ds = ChickenAgePairedDataset(val_csv, root_dir, get_default_transforms(False), use_masks, mask_dir)
    test_ds = ChickenAgePairedDataset(test_csv, root_dir, get_default_transforms(False), use_masks, mask_dir)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader 