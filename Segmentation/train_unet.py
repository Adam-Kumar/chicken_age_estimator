"""train_unet.py

Train U-Net segmentation model on manually annotated masks.

After annotating ~30 images with annotate_masks.py, train a lightweight
U-Net to segment chicken from plate/background.

Usage:
    python Segmentation/train_unet.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt


class UNet(nn.Module):
    """Lightweight U-Net for binary segmentation."""

    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()

        # Encoder
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)

        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)

        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = self.conv_block(1024, 512)

        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = self.conv_block(512, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = self.conv_block(256, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = self.conv_block(128, 64)

        # Output
        self.out = nn.Conv2d(64, out_channels, 1)

        self.pool = nn.MaxPool2d(2, 2)

    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))

        # Decoder with skip connections
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.dec4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)

        out = self.out(dec1)
        return out


class SegmentationDataset(Dataset):
    """Dataset for loading images and masks."""

    def __init__(self, image_dir, mask_dir, image_size=256, augment=False):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.image_size = image_size
        self.augment = augment

        # Find all annotated images
        self.samples = []
        for view in ['TOP VIEW', 'SIDE VIEW']:
            for day in range(1, 8):
                mask_day_dir = self.mask_dir / view / f'Day {day}'
                if mask_day_dir.exists():
                    for mask_path in mask_day_dir.glob('*.jpg'):
                        img_path = self.image_dir / view / f'Day {day}' / mask_path.name
                        if img_path.exists():
                            self.samples.append((img_path, mask_path))

        print(f"Found {len(self.samples)} annotated images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        # Load image and mask
        image = np.array(Image.open(img_path).convert('RGB'))
        mask = np.array(Image.open(mask_path).convert('L'))

        # Resize
        image = cv2.resize(image, (self.image_size, self.image_size))
        mask = cv2.resize(mask, (self.image_size, self.image_size))

        # Augmentation
        if self.augment:
            # Random horizontal flip
            if np.random.rand() > 0.5:
                image = np.fliplr(image)
                mask = np.fliplr(mask)

            # Random rotation
            if np.random.rand() > 0.5:
                angle = np.random.randint(-15, 15)
                h, w = image.shape[:2]
                M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
                image = cv2.warpAffine(image, M, (w, h))
                mask = cv2.warpAffine(mask, M, (w, h))

            # Random brightness
            if np.random.rand() > 0.5:
                factor = np.random.uniform(0.8, 1.2)
                image = np.clip(image * factor, 0, 255).astype(np.uint8)

        # Normalize image
        image = image.astype(np.float32) / 255.0
        image = (image - 0.5) / 0.5  # [-1, 1]

        # Normalize mask
        mask = (mask > 127).astype(np.float32)

        # To tensor (C, H, W)
        image = torch.from_numpy(image.transpose(2, 0, 1))
        mask = torch.from_numpy(mask).unsqueeze(0)

        return image, mask


def dice_loss(pred, target, smooth=1.0):
    """Dice loss for segmentation."""
    pred = torch.sigmoid(pred)
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)

    intersection = (pred_flat * target_flat).sum()
    dice = (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)

    return 1.0 - dice


def train_unet(train_loader, val_loader, model, device, epochs=50, lr=1e-4):
    """Train U-Net model."""

    optimizer = optim.Adam(model.parameters(), lr=lr)
    bce_loss = nn.BCEWithLogitsLoss()

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    checkpoint_dir = project_root / "Segmentation" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for images, masks in pbar:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            # Combined loss: BCE + Dice
            loss = bce_loss(outputs, masks) + dice_loss(outputs, masks)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = bce_loss(outputs, masks) + dice_loss(outputs, masks)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_dir / "unet_best.pth")
            print(f"[OK] Saved best model (val_loss: {val_loss:.4f})")

    # Save final model
    torch.save(model.state_dict(), checkpoint_dir / "unet_final.pth")

    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('U-Net Training Curves')
    plt.savefig(checkpoint_dir / "training_curves.png", dpi=150, bbox_inches='tight')
    plt.close()

    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best model saved to: {checkpoint_dir / 'unet_best.pth'}")
    print(f"Training curves saved to: {checkpoint_dir / 'training_curves.png'}")


def main():
    print("="*80)
    print("U-NET SEGMENTATION MODEL TRAINING")
    print("="*80)
    print()

    # Check for annotated masks
    mask_dir = project_root / "Segmentation" / "training_masks"
    if not mask_dir.exists():
        print("[ERROR] No training masks found!")
        print("\nPlease annotate images first:")
        print("  python Segmentation/annotate_masks.py --num_images 30")
        return

    # Count annotated images
    num_masks = len(list(mask_dir.glob('**/*.jpg')))
    if num_masks < 20:
        print(f"[WARNING] Only {num_masks} annotated masks found!")
        print("Recommended: 30+ annotated images for good performance")
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            return

    print(f"Found {num_masks} annotated training masks\n")

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load dataset
    dataset = SegmentationDataset(
        image_dir=project_root / "Dataset_Processed",
        mask_dir=mask_dir,
        image_size=224,
        augment=True
    )

    # Train/val split (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)

    print(f"\nTrain samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}\n")

    # Create model
    model = UNet(in_channels=3, out_channels=1).to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}\n")

    # Train
    train_unet(train_loader, val_loader, model, device, epochs=50, lr=1e-4)

    print("\nNext step: Generate masks for all images")
    print("  python Segmentation/generate_masks_unet.py")


if __name__ == "__main__":
    main()
