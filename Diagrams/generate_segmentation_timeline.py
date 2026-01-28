"""Generate Segmentation Timeline Image

Displays segmentation masks and masked results for a single chicken across 7 days.
Shows both TOP and SIDE views with masks on top and segmented images below.
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import numpy as np
from PIL import Image
import argparse

def apply_mask_to_image(image_path, mask_path):
    """Apply mask to image, setting background to white."""
    img = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")

    # Convert to numpy arrays
    img_array = np.array(img)
    mask_array = np.array(mask)

    # Create mask (1 where chicken, 0 where background)
    mask_binary = (mask_array > 127).astype(np.uint8)

    # Apply mask: set background to white
    masked_img = img_array.copy()
    for c in range(3):  # RGB channels
        masked_img[:, :, c] = img_array[:, :, c] * mask_binary + 255 * (1 - mask_binary)

    return masked_img

def generate_segmentation_timeline(chicken_id, output_path=None):
    """
    Generate a timeline showing segmentation masks and results for one chicken.

    Parameters:
    -----------
    chicken_id : int
        Chicken ID (1-82)
    output_path : str, optional
        Output path. If None, uses default naming
    """
    # Setup paths
    project_root = Path(__file__).parent.parent
    dataset_dir = project_root / "Dataset_Processed"
    mask_dir = project_root / "Segmentation" / "masks"

    # Create figure with 4 rows and 7 columns
    # Row 1: TOP masks, Row 2: TOP masked images
    # Row 3: SIDE masks, Row 4: SIDE masked images
    fig, axes = plt.subplots(4, 7, figsize=(21, 12))

    views = ['TOP VIEW', 'SIDE VIEW']
    view_labels = ['Top (Dorsal)', 'Side (Lateral)']

    for view_idx, (view, view_label) in enumerate(zip(views, view_labels)):
        base_row = view_idx * 2  # 0 for TOP, 2 for SIDE

        for day in range(1, 8):
            img_path = dataset_dir / view / f"Day {day}" / f"{chicken_id}.jpg"
            mask_path = mask_dir / view / f"Day {day}" / f"{chicken_id}.jpg"

            if not img_path.exists() or not mask_path.exists():
                print(f"Warning: Image or mask not found for {view} Day {day}")
                continue

            # Load mask
            mask_img = mpimg.imread(mask_path)

            # Display mask in first row for this view
            ax_mask = axes[base_row, day - 1]
            ax_mask.imshow(mask_img, cmap='gray')
            ax_mask.axis('off')

            # Apply mask and get masked image
            masked_img = apply_mask_to_image(img_path, mask_path)

            # Display masked image in second row for this view
            ax_masked = axes[base_row + 1, day - 1]
            ax_masked.imshow(masked_img)
            ax_masked.axis('off')

            # Add day label below last row only
            if view_idx == 1:
                ax_masked.set_xlabel(f'Day {day}', fontsize=12, weight='bold', labelpad=8)

            # Add labels on first column
            if day == 1:
                ax_mask.set_ylabel(f'{view_label}\nMask', fontsize=11, weight='bold',
                                  rotation=0, labelpad=70, ha='right', va='center')
                ax_masked.set_ylabel(f'{view_label}\nMasked', fontsize=11, weight='bold',
                                    rotation=0, labelpad=70, ha='right', va='center')

    # Add main title
    fig.suptitle(f'Chicken #{chicken_id} - Segmentation Examples',
                 fontsize=20, weight='bold', y=0.98)

    # Add subtitle
    fig.text(0.5, 0.02,
             'Binary masks (rows 1 & 3) and masked images with background removed (rows 2 & 4)',
             ha='center', fontsize=12, style='italic', color='#666666')

    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    # Save figure
    if output_path is None:
        output_path = project_root / "Diagrams" / f"chicken_{chicken_id}_segmentation.png"
    else:
        output_path = Path(output_path)

    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"Segmentation timeline saved to: {output_path}")
    plt.close()

    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate chicken segmentation timeline visualization')
    parser.add_argument('--chicken_id', type=int, default=5,
                       help='Chicken ID (1-82), default: 5')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path (optional)')

    args = parser.parse_args()

    print(f"Generating segmentation timeline for Chicken #{args.chicken_id}")
    print("-" * 60)

    generate_segmentation_timeline(args.chicken_id, args.output)

    print("-" * 60)
    print("Done!")
