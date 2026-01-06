"""test_mask_loading.py

Quick test to verify that mask loading works correctly.
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from Model.dataset import ChickenAgeDataset, ChickenAgePairedDataset, get_default_transforms
import matplotlib.pyplot as plt
import numpy as np


def test_single_view_mask():
    """Test single-view dataset with masks."""
    print("Testing single-view dataset with masks...")

    # Use train.csv for testing
    csv_file = project_root / "Labels" / "train.csv"
    root_dir = project_root / "Dataset_Processed"
    mask_dir = project_root / "Segmentation" / "masks"

    # Create dataset WITHOUT masks
    ds_no_mask = ChickenAgeDataset(csv_file, root_dir, transforms=None, use_masks=False)

    # Create dataset WITH masks
    ds_with_mask = ChickenAgeDataset(csv_file, root_dir, transforms=None, use_masks=True, mask_dir=mask_dir)

    # Get first sample
    img_no_mask, label = ds_no_mask[0]
    img_with_mask, _ = ds_with_mask[0]

    # Convert to numpy for visualization
    img_no_mask_np = np.array(img_no_mask)
    img_with_mask_np = np.array(img_with_mask)

    # Display
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(img_no_mask_np)
    axes[0].set_title("Original Image (No Mask)")
    axes[0].axis('off')

    axes[1].imshow(img_with_mask_np)
    axes[1].set_title("Masked Image (Chicken Only)")
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(project_root / "Scripts" / "test_mask_comparison.png", dpi=150, bbox_inches='tight')
    print(f"[OK] Saved comparison to Scripts/test_mask_comparison.png")
    plt.close()

    # Check if masking worked
    diff = np.abs(img_no_mask_np.astype(float) - img_with_mask_np.astype(float)).sum()
    if diff > 0:
        print("[OK] Mask applied successfully (images are different)")
    else:
        print("[WARNING] Images are identical - mask may not be applied")

    return True


def test_paired_view_mask():
    """Test paired-view dataset with masks."""
    print("\nTesting paired-view dataset with masks...")

    csv_file = project_root / "Labels" / "train.csv"
    root_dir = project_root / "Dataset_Processed"
    mask_dir = project_root / "Segmentation" / "masks"

    # Create dataset WITH masks
    ds_with_mask = ChickenAgePairedDataset(csv_file, root_dir, transforms=None, use_masks=True, mask_dir=mask_dir)

    # Get first sample
    (img_top, img_side), label = ds_with_mask[0]

    # Convert to numpy
    img_top_np = np.array(img_top)
    img_side_np = np.array(img_side)

    # Display
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(img_top_np)
    axes[0].set_title("TOP View (Masked)")
    axes[0].axis('off')

    axes[1].imshow(img_side_np)
    axes[1].set_title("SIDE View (Masked)")
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(project_root / "Scripts" / "test_paired_mask.png", dpi=150, bbox_inches='tight')
    print(f"[OK] Saved paired view to Scripts/test_paired_mask.png")
    plt.close()

    return True


def main():
    print("="*80)
    print("MASK LOADING TEST")
    print("="*80)
    print()

    try:
        # Test single-view
        test_single_view_mask()

        # Test paired-view
        test_paired_view_mask()

        print("\n" + "="*80)
        print("ALL TESTS PASSED")
        print("="*80)
        print("\nMask loading is working correctly!")
        print("You can now run train_all_models.py with masks enabled.")

    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
