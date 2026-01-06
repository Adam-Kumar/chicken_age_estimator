"""annotate_masks.py

Interactive tool to create ground truth segmentation masks for U-Net training.

Uses GrabCut semi-automatic segmentation on 224x224 processed images.

Usage:
    python Segmentation/annotate_masks.py --num_images 30
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import argparse
import random
import numpy as np
from PIL import Image
import cv2


class MaskAnnotator:
    """Interactive mask annotation tool using GrabCut."""

    def __init__(self, image_path, output_path):
        self.image_path = image_path
        self.output_path = output_path

        # Load image
        self.image = np.array(Image.open(image_path).convert('RGB'))
        self.display_image = self.image.copy()

        # Points for GrabCut
        self.fg_points = []  # Foreground (chicken)
        self.bg_points = []  # Background (plate/lightbox)

        self.mask = None
        self.base_mask = None  # Cache GrabCut result
        self.saved = False

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks."""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Left click = foreground point (chicken)
            self.fg_points.append((x, y))
            cv2.circle(self.display_image, (x, y), 3, (0, 255, 0), -1)  # Green
            cv2.imshow('Annotate', self.display_image)
            print(f"Foreground point added: ({x}, {y})")

        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right click = background point (plate)
            self.bg_points.append((x, y))
            cv2.circle(self.display_image, (x, y), 3, (0, 0, 255), -1)  # Red
            cv2.imshow('Annotate', self.display_image)
            print(f"Background point added: ({x}, {y})")

    def run_grabcut(self, auto_mode=False):
        """Run GrabCut and cache the result."""
        if not auto_mode and len(self.fg_points) == 0:
            print("Need at least one foreground point!")
            return None

        print("Running GrabCut...")

        h, w = self.image.shape[:2]
        bgd_model = np.zeros((1, 65), dtype=np.float64)
        fgd_model = np.zeros((1, 65), dtype=np.float64)

        try:
            if auto_mode:
                # AUTO MODE: Use center rectangle (no manual points)
                print("  AUTO mode: Using center rectangle (60% of image)")
                margin = 0.20  # 20% margin on each side = 60% center
                rect = (int(w * margin), int(h * margin),
                       int(w * (1 - 2*margin)), int(h * (1 - 2*margin)))
                mask_grabcut = np.zeros((h, w), dtype=np.uint8)
                cv2.grabCut(self.image, mask_grabcut, rect, bgd_model, fgd_model,
                           5, cv2.GC_INIT_WITH_RECT)
            else:
                # MANUAL MODE: Use user points
                print(f"  MANUAL mode: {len(self.fg_points)} FG points, {len(self.bg_points)} BG points")
                mask_grabcut = np.zeros((h, w), dtype=np.uint8)
                mask_grabcut[:] = cv2.GC_PR_BGD  # Probably background

                # Add foreground points
                for x, y in self.fg_points:
                    cv2.circle(mask_grabcut, (x, y), 5, cv2.GC_FGD, -1)

                # Add background points
                for x, y in self.bg_points:
                    cv2.circle(mask_grabcut, (x, y), 10, cv2.GC_BGD, -1)

                cv2.grabCut(self.image, mask_grabcut, None, bgd_model, fgd_model,
                           5, cv2.GC_INIT_WITH_MASK)

            # Create binary mask
            mask = np.where((mask_grabcut == cv2.GC_FGD) | (mask_grabcut == cv2.GC_PR_FGD),
                           255, 0).astype(np.uint8)

            # Keep largest component
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            if num_labels > 1:
                largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                mask = (labels == largest_label).astype(np.uint8) * 255

            # Light smoothing
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

            print("[OK] GrabCut complete")
            return mask

        except Exception as e:
            print(f"GrabCut failed: {e}")
            return None

    def apply_erosion(self, base_mask, erosion_size):
        """Apply erosion to remove plate edges."""
        mask = base_mask.copy()

        if erosion_size > 0:
            kernel = np.ones((erosion_size, erosion_size), np.uint8)
            mask = cv2.erode(mask, kernel, iterations=1)

        # Fill small holes
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

        return mask

    def manual_refinement(self, mask):
        """Manual painting mode to refine mask."""
        print("\n" + "="*80)
        print("MANUAL REFINEMENT MODE")
        print("="*80)
        print("\nControls:")
        print("  Left-click + drag:  Paint to ADD (include chicken)")
        print("  Right-click + drag: Paint to REMOVE (exclude plate)")
        print("  Mouse wheel / [/]:  Adjust brush size")
        print("  's':                Save and exit")
        print("  'r':                Reset to original")
        print("  'q':                Cancel\n")

        working_mask = mask.copy()
        original_mask = mask.copy()
        brush_size = 10
        drawing = False
        draw_mode = 'add'

        def paint_callback(event, x, y, flags, param):
            nonlocal drawing, draw_mode, working_mask, brush_size

            if event == cv2.EVENT_LBUTTONDOWN:
                drawing = True
                draw_mode = 'add'
                cv2.circle(working_mask, (x, y), brush_size, 255, -1)

            elif event == cv2.EVENT_RBUTTONDOWN:
                drawing = True
                draw_mode = 'remove'
                cv2.circle(working_mask, (x, y), brush_size, 0, -1)

            elif event == cv2.EVENT_MOUSEMOVE:
                if drawing:
                    value = 255 if draw_mode == 'add' else 0
                    cv2.circle(working_mask, (x, y), brush_size, value, -1)

            elif event == cv2.EVENT_LBUTTONUP or event == cv2.EVENT_RBUTTONUP:
                drawing = False

            elif event == cv2.EVENT_MOUSEWHEEL:
                if flags > 0:
                    brush_size = min(brush_size + 2, 50)
                else:
                    brush_size = max(brush_size - 2, 3)
                print(f"Brush size: {brush_size}px")

        cv2.namedWindow('Refine Mask')
        cv2.setMouseCallback('Refine Mask', paint_callback)

        while True:
            # Show overlay
            display = self.image.copy()
            mask_colored = np.zeros_like(display)
            mask_colored[working_mask > 0] = [0, 255, 0]
            display = cv2.addWeighted(display, 0.7, mask_colored, 0.3, 0)

            # Add instructions
            cv2.putText(display, f"Brush: {brush_size}px | L-click=Add, R-click=Remove",
                       (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            cv2.putText(display, "s=Save, r=Reset, q=Cancel, [/]=Brush",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

            cv2.imshow('Refine Mask', display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('s'):
                print("[OK] Refinement saved")
                cv2.destroyWindow('Refine Mask')
                return working_mask

            elif key == ord('r'):
                working_mask = original_mask.copy()
                print("Reset to original")

            elif key == ord('q'):
                print("Cancelled")
                cv2.destroyWindow('Refine Mask')
                return original_mask

            elif key == ord('['):
                brush_size = max(brush_size - 2, 3)
                print(f"Brush size: {brush_size}px")

            elif key == ord(']'):
                brush_size = min(brush_size + 2, 50)
                print(f"Brush size: {brush_size}px")

        return working_mask

    def run(self):
        """Run interactive annotation."""
        print("\n" + "="*80)
        print(f"Annotating: {self.image_path.name}")
        print("="*80)
        print("\nControls:")
        print("  'a':         AUTO mode - generate mask with default rectangle (NO POINTS NEEDED)")
        print("  Left-click:  Foreground point (GREEN) - click on chicken")
        print("  Right-click: Background point (RED) - click on plate edges")
        print("  's':         Preview mask with GrabCut (manual mode)")
        print("  'r':         Reset points")
        print("  'q':         Quit")
        print("\nTip: Try 'a' first for automatic mask generation!\n")

        cv2.namedWindow('Annotate')
        cv2.setMouseCallback('Annotate', self.mouse_callback)
        cv2.imshow('Annotate', self.display_image)

        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == ord('a'):
                # AUTO MODE: Generate mask with default rectangle
                if self.base_mask is None:
                    self.base_mask = self.run_grabcut(auto_mode=True)
                    if self.base_mask is None:
                        continue

                # Erosion adjustment
                erosion_options = [0, 2, 3, 5, 7, 10]
                current_erosion_idx = 1  # Start with 2px

                while True:
                    erosion_size = erosion_options[current_erosion_idx]
                    self.mask = self.apply_erosion(self.base_mask, erosion_size)

                    # Show preview
                    preview = self.image.copy()
                    mask_colored = np.zeros_like(preview)
                    mask_colored[self.mask > 0] = [0, 255, 0]
                    preview = cv2.addWeighted(preview, 0.7, mask_colored, 0.3, 0)

                    # Show masked result
                    masked = self.image.copy()
                    masked[self.mask == 0] = 255

                    # Side by side
                    combined = np.hstack([preview, masked])

                    cv2.putText(combined, f"AUTO mode | Erosion: {erosion_size}px",
                               (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    cv2.putText(combined, "e=edit, +/-=erosion, s=save, r=redo, q=skip",
                               (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

                    cv2.imshow('Preview', combined)
                    key2 = cv2.waitKey(0) & 0xFF

                    if key2 == ord('s'):
                        # Save mask
                        self.output_path.parent.mkdir(parents=True, exist_ok=True)
                        Image.fromarray(self.mask).save(self.output_path)
                        print(f"[OK] Saved: {self.output_path}")
                        self.saved = True
                        cv2.destroyAllWindows()
                        return self.saved

                    elif key2 == ord('e'):
                        # Manual refinement
                        cv2.destroyWindow('Preview')
                        refined = self.manual_refinement(self.mask)
                        self.base_mask = refined
                        self.mask = refined
                        current_erosion_idx = 0

                    elif key2 == ord('+') or key2 == ord('='):
                        current_erosion_idx = min(current_erosion_idx + 1, len(erosion_options) - 1)

                    elif key2 == ord('-') or key2 == ord('_'):
                        current_erosion_idx = max(current_erosion_idx - 1, 0)

                    elif key2 == ord('r'):
                        # Redo
                        cv2.destroyWindow('Preview')
                        self.reset()
                        cv2.imshow('Annotate', self.display_image)
                        break

                    elif key2 == ord('q'):
                        # Skip
                        cv2.destroyAllWindows()
                        return False

            elif key == ord('s'):
                # MANUAL MODE: Run GrabCut with user points
                if self.base_mask is None:
                    self.base_mask = self.run_grabcut(auto_mode=False)
                    if self.base_mask is None:
                        continue

                # Erosion adjustment
                erosion_options = [0, 2, 3, 5, 7, 10]
                current_erosion_idx = 1  # Start with 2px

                while True:
                    erosion_size = erosion_options[current_erosion_idx]
                    self.mask = self.apply_erosion(self.base_mask, erosion_size)

                    # Show preview
                    preview = self.image.copy()
                    mask_colored = np.zeros_like(preview)
                    mask_colored[self.mask > 0] = [0, 255, 0]
                    preview = cv2.addWeighted(preview, 0.7, mask_colored, 0.3, 0)

                    # Show masked result
                    masked = self.image.copy()
                    masked[self.mask == 0] = 255

                    # Side by side
                    combined = np.hstack([preview, masked])

                    cv2.putText(combined, f"Erosion: {erosion_size}px",
                               (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    cv2.putText(combined, "e=edit, +/-=erosion, s=save, r=redo, q=skip",
                               (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

                    cv2.imshow('Preview', combined)
                    key2 = cv2.waitKey(0) & 0xFF

                    if key2 == ord('s'):
                        # Save mask
                        self.output_path.parent.mkdir(parents=True, exist_ok=True)
                        Image.fromarray(self.mask).save(self.output_path)
                        print(f"[OK] Saved: {self.output_path}")
                        self.saved = True
                        cv2.destroyAllWindows()
                        return self.saved

                    elif key2 == ord('e'):
                        # Manual refinement
                        cv2.destroyWindow('Preview')
                        refined = self.manual_refinement(self.mask)
                        self.base_mask = refined
                        self.mask = refined
                        current_erosion_idx = 0

                    elif key2 == ord('+') or key2 == ord('='):
                        current_erosion_idx = min(current_erosion_idx + 1, len(erosion_options) - 1)

                    elif key2 == ord('-') or key2 == ord('_'):
                        current_erosion_idx = max(current_erosion_idx - 1, 0)

                    elif key2 == ord('r'):
                        # Redo
                        cv2.destroyWindow('Preview')
                        self.reset()
                        cv2.imshow('Annotate', self.display_image)
                        break

                    elif key2 == ord('q'):
                        # Skip
                        cv2.destroyAllWindows()
                        return False

            elif key == ord('r'):
                self.reset()
                cv2.imshow('Annotate', self.display_image)

            elif key == ord('q'):
                cv2.destroyAllWindows()
                break

        cv2.destroyAllWindows()
        return self.saved

    def reset(self):
        """Reset annotation."""
        self.fg_points = []
        self.bg_points = []
        self.base_mask = None
        self.display_image = self.image.copy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_images', type=int, default=30,
                       help='Number of images to annotate')
    args = parser.parse_args()

    print("="*80)
    print("MASK ANNOTATION TOOL - U-NET TRAINING")
    print("="*80)
    print(f"\nTarget: {args.num_images} annotated images")
    print("Dataset: 224x224 processed images\n")

    # Setup paths
    dataset_dir = project_root / "Dataset_Processed"
    output_dir = project_root / "Segmentation" / "training_masks"

    # Find all images
    image_files = []
    for view in ['TOP VIEW', 'SIDE VIEW']:
        for day in range(1, 8):
            day_dir = dataset_dir / view / f'Day {day}'
            if day_dir.exists():
                image_files.extend(list(day_dir.glob('*.jpg')))

    print(f"Found {len(image_files)} total images")

    # Random sample
    random.shuffle(image_files)
    selected = image_files[:args.num_images]

    print(f"Selected {len(selected)} images for annotation\n")

    # Check existing annotations
    existing = []
    for img_path in selected:
        relative_path = img_path.relative_to(dataset_dir)
        output_path = output_dir / relative_path
        if output_path.exists():
            existing.append(output_path)

    if existing:
        print(f"Found {len(existing)} existing annotations")
        print("Will skip already annotated images\n")

    # Annotate
    annotated = 0
    skipped = 0

    for i, img_path in enumerate(selected):
        relative_path = img_path.relative_to(dataset_dir)
        output_path = output_dir / relative_path

        # Skip if exists
        if output_path.exists():
            print(f"[{i+1}/{len(selected)}] Skipping (exists): {img_path.name}")
            skipped += 1
            continue

        print(f"\n[{i+1}/{len(selected)}] Annotating...")

        annotator = MaskAnnotator(img_path, output_path)
        saved = annotator.run()

        if saved:
            annotated += 1
        else:
            print(f"Skipped: {img_path.name}")

    print("\n" + "="*80)
    print("ANNOTATION COMPLETE")
    print("="*80)
    print(f"Newly annotated: {annotated}")
    print(f"Skipped (existing): {skipped}")
    print(f"Total annotations: {annotated + skipped}")
    print(f"\nMasks saved to: {output_dir}")

    if annotated + skipped >= 20:
        print("\n[OK] Ready to train U-Net!")
        print("\nNext step:")
        print("  python Segmentation/train_unet.py")
    else:
        print(f"\n[WARNING] Only {annotated + skipped} annotations")
        print("Recommended: 30+ for good performance")


if __name__ == "__main__":
    main()
