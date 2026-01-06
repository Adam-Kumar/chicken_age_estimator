"""generate_masks_unet.py

Generate segmentation masks for ALL 1148 images (224x224) using trained U-Net model.

Run this after training U-Net on annotated samples.

Usage:
    python Segmentation/generate_masks_unet.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm


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


def segment_with_unet(image, model, device):
    """Segment 224x224 image using trained U-Net."""
    # Normalize
    image_normalized = image.astype(np.float32) / 255.0
    image_normalized = (image_normalized - 0.5) / 0.5

    # To tensor
    image_tensor = torch.from_numpy(image_normalized.transpose(2, 0, 1)).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        output = model(image_tensor)
        output = torch.sigmoid(output)
        mask = (output[0, 0].cpu().numpy() > 0.5).astype(np.uint8) * 255

    # Post-processing: keep largest component
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (labels == largest_label).astype(np.uint8) * 255

    # Light smoothing
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    return mask


def generate_all_masks(dataset_dir, output_dir, model, device):
    """Generate masks for all images using trained U-Net."""
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)

    # Find all images
    image_files = []
    for view in ['TOP VIEW', 'SIDE VIEW']:
        for day in range(1, 8):
            day_dir = dataset_dir / view / f'Day {day}'
            if day_dir.exists():
                image_files.extend(list(day_dir.glob('*.jpg')))

    print(f"Found {len(image_files)} images to process")
    print(f"Output directory: {output_dir}\n")

    successful = 0
    failed = 0

    for img_path in tqdm(image_files, desc="Generating masks"):
        try:
            # Load image
            image = np.array(Image.open(img_path).convert('RGB'))

            # Generate mask
            mask = segment_with_unet(image, model, device)

            # Save mask (preserve directory structure)
            relative_path = img_path.relative_to(dataset_dir)
            output_path = output_dir / relative_path
            output_path.parent.mkdir(parents=True, exist_ok=True)

            Image.fromarray(mask).save(output_path)
            successful += 1

        except Exception as e:
            tqdm.write(f"Failed: {img_path}: {e}")
            failed += 1

    print(f"\n{'='*80}")
    print("MASK GENERATION COMPLETE")
    print("="*80)
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Masks saved to: {output_dir}")

    if failed == 0:
        print("\n[OK] All masks generated successfully!")
        print("\nNext steps:")
        print("1. Update dataset loaders to use masks")
        print("2. Retrain models with masked data")
        print("3. Compare before/after performance")


def main():
    print("="*80)
    print("BATCH MASK GENERATION - U-NET MODEL")
    print("="*80)
    print()

    # Check for trained model
    checkpoint_path = project_root / "Segmentation" / "checkpoints" / "unet_best.pth"
    if not checkpoint_path.exists():
        print("[ERROR] Trained U-Net model not found!")
        print("\nPlease train model first:")
        print("  python Segmentation/train_unet.py")
        return

    print(f"Loading trained model from: {checkpoint_path}\n")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Load model
    model = UNet(in_channels=3, out_channels=1).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    print("[OK] Model loaded\n")

    # Check for approval
    response = input("Ready to generate masks for all 1148 images? (y/n): ")
    if response.lower() != 'y':
        return

    # Generate masks
    dataset_dir = project_root / "Dataset_Processed"
    output_dir = project_root / "Segmentation" / "masks"

    generate_all_masks(dataset_dir, output_dir, model, device)


if __name__ == "__main__":
    main()
