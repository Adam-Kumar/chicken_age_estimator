"""Generate Chicken Timeline Image

Displays the progression of a single chicken across 7 days.
Shows both TOP and SIDE views in a 2×7 grid using original high-resolution images.
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import argparse

def generate_chicken_timeline(chicken_id, output_path=None):
    """
    Generate a timeline showing both TOP and SIDE views for one chicken.

    Parameters:
    -----------
    chicken_id : int
        Chicken ID (1-82)
    output_path : str, optional
        Output path. If None, uses default naming
    """
    # Setup paths
    project_root = Path(__file__).parent.parent

    # Create figure with 2 rows (TOP and SIDE) and 7 columns (days)
    fig, axes = plt.subplots(2, 7, figsize=(21, 7))

    views = ['TOP VIEW', 'SIDE VIEW']
    view_labels = ['Top (Dorsal)', 'Side (Lateral)']

    for view_idx, (view, view_label) in enumerate(zip(views, view_labels)):
        dataset_dir = project_root / "Dataset_Original" / view

        for day in range(1, 8):
            img_path = dataset_dir / f"Day {day}" / f"{chicken_id}.jpg"

            if not img_path.exists():
                print(f"Warning: Image not found: {img_path}")
                continue

            # Load image
            img = mpimg.imread(img_path)

            # Display image
            ax = axes[view_idx, day - 1]
            ax.imshow(img)
            ax.axis('off')

            # Add day label below bottom row only
            if view_idx == 1:
                ax.set_xlabel(f'Day {day}', fontsize=12, weight='bold', labelpad=8)

            # Add view label on first column only
            if day == 1:
                ax.set_ylabel(view_label, fontsize=12, weight='bold',
                             rotation=0, labelpad=60, ha='right', va='center')

    # Add main title
    fig.suptitle(f'Chicken #{chicken_id} - Multi-View Temporal Progression',
                 fontsize=20, weight='bold', y=0.98)

    # Add subtitle
    fig.text(0.5, 0.02, 'Paired TOP and SIDE view images across 7-day post-slaughter timeline',
             ha='center', fontsize=12, style='italic', color='#666666')

    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    # Save figure
    if output_path is None:
        output_path = project_root / "Diagrams" / f"chicken_{chicken_id}_timeline.png"
    else:
        output_path = Path(output_path)

    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"Timeline saved to: {output_path}")
    plt.close()

    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate chicken dual-view timeline visualization')
    parser.add_argument('--chicken_id', type=int, default=5,
                       help='Chicken ID (1-82), default: 5')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path (optional)')

    args = parser.parse_args()

    print(f"Generating dual-view timeline for Chicken #{args.chicken_id}")
    print("-" * 60)

    generate_chicken_timeline(args.chicken_id, args.output)

    print("-" * 60)
    print("Done!")
