"""Generate Chicken Timeline Image (Combined)

Displays the progression of a single chicken across 7 days.
Shows both TOP and SIDE views in a 2×7 grid.
Supports both original high-resolution and processed 224×224 images.
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import argparse


def generate_chicken_timeline(chicken_id, use_processed=False, output_path=None):
    """
    Generate a timeline showing both TOP and SIDE views for one chicken.

    Parameters:
    -----------
    chicken_id : int
        Chicken ID (1-82)
    use_processed : bool
        If True, use processed 224×224 images. If False, use original images.
    output_path : str, optional
        Output path. If None, uses default naming
    """
    # Setup paths
    project_root = Path(__file__).parent.parent

    # Choose dataset based on flag
    if use_processed:
        dataset_name = "Dataset_Processed"
        output_suffix = "_processed"
    else:
        dataset_name = "Dataset_Original"
        output_suffix = ""

    # Create figure with 2 rows (TOP and SIDE) and 7 columns (days)
    fig, axes = plt.subplots(2, 7, figsize=(21, 6), gridspec_kw={'hspace': 0.02, 'wspace': 0.02})

    views = ['TOP VIEW', 'SIDE VIEW']
    view_labels = ['Top', 'Side']

    for view_idx, (view, view_label) in enumerate(zip(views, view_labels)):
        dataset_dir = project_root / dataset_name / view

        for day in range(1, 8):
            img_path = dataset_dir / f"Day {day}" / f"{chicken_id}.jpg"

            if not img_path.exists():
                print(f"Warning: Image not found: {img_path}")
                continue

            # Load image
            img = mpimg.imread(img_path)

            # Display image
            ax = axes[view_idx, day - 1]
            ax.imshow(img, aspect='auto')
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            # Add day label above top row only
            if view_idx == 0:
                ax.set_title(f'Day {day}', fontsize=12, weight='bold', pad=8)

            # Add view label on first column only
            if day == 1:
                ax.set_ylabel(view_label, fontsize=14, weight='bold',
                             rotation=90, labelpad=10)


    # Save figure
    if output_path is None:
        output_path = project_root / "Diagrams" / f"chicken_{chicken_id}_timeline{output_suffix}.png"
    else:
        output_path = Path(output_path)

    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"Timeline saved to: {output_path}")
    plt.close()

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate chicken dual-view timeline visualization')
    parser.add_argument('--chicken_id', type=int, default=7,
                       help='Chicken ID (1-82), default: 7')
    parser.add_argument('--original_only', action='store_true',
                       help='Generate only original dataset timeline')
    parser.add_argument('--processed_only', action='store_true',
                       help='Generate only processed dataset timeline')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path (optional, only used with _only flags)')

    args = parser.parse_args()

    print(f"Generating dual-view timeline for Chicken #{args.chicken_id}")
    print("-" * 60)

    if args.original_only:
        generate_chicken_timeline(args.chicken_id, use_processed=False, output_path=args.output)
    elif args.processed_only:
        generate_chicken_timeline(args.chicken_id, use_processed=True, output_path=args.output)
    else:
        # Default: generate both
        print("Generating original dataset timeline...")
        generate_chicken_timeline(args.chicken_id, use_processed=False)
        print()
        print("Generating processed dataset timeline...")
        generate_chicken_timeline(args.chicken_id, use_processed=True)

    print("-" * 60)
    print("Done!")
