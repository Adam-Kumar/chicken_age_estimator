import os
from PIL import Image

# === CONFIGURATION ===
INPUT_ROOT = "Dataset Original"
OUTPUT_ROOT = "Processed"
TEST_MODE = False  # Set to True to process only one image from each day for testing
CROP_PARAMS_TOP = {
    "top": 0.36,    # Crop 36% from the top
    "bottom": 0.31, # Crop 31% from the bottom
    "left": 0.35,   # Crop 35% from the left
    "right": 0.33   # Crop 33% from the right
}
CROP_PARAMS_SIDE = {
    "top": 0.27,    # Crop 25% from the top
    "bottom": 0.29, # Crop 25% from the bottom
    "left": 0.25,   # Crop 25% from the left
    "right": 0.23   # Crop 25% from the right
}
RESIZE_TO = (224, 224)
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
SQUARE_CROP = {
    "zoom": 1.0,      # 1.0 = largest possible square, >1.0 = zoom in, <1.0 = zoom out (if possible)
    "x_shift": 0.0,   # -1.0 (left) to 1.0 (right), 0.0 = center
    "y_shift": 0.0    # -1.0 (top) to 1.0 (bottom), 0.0 = center
}

def crop_image(img, crop_params):
    w, h = img.size
    left = int(w * crop_params["left"])
    right = w - int(w * crop_params["right"])
    top = int(h * crop_params["top"])
    bottom = h - int(h * crop_params["bottom"])
    return img.crop((left, top, right, bottom))

def crop_to_square_with_zoom_and_shift(img, zoom=1.0, x_shift=0.0, y_shift=0.0):
    w, h = img.size
    side = min(w, h)
    # Adjust side for zoom
    side = int(side / zoom)
    # Clamp side to image dimensions
    side = min(side, w, h)
    # Calculate center
    center_x = w // 2
    center_y = h // 2
    # Calculate max shift in pixels
    max_x_shift = (w - side) // 2
    max_y_shift = (h - side) // 2
    # Apply shift: -1.0 = far left/top, 1.0 = far right/bottom
    shift_x = int(x_shift * max_x_shift)
    shift_y = int(y_shift * max_y_shift)
    left = center_x - side // 2 + shift_x
    top = center_y - side // 2 + shift_y
    # Clamp to image bounds
    left = max(0, min(left, w - side))
    top = max(0, min(top, h - side))
    right = left + side
    bottom = top + side
    return img.crop((left, top, right, bottom))

def process_images(input_root, output_root, crop_params_top, crop_params_side, resize_to, test_mode=False):
    for view in os.listdir(input_root):
        view_path = os.path.join(input_root, view)
        if not os.path.isdir(view_path):
            continue
        # Choose crop params based on view
        if view.upper() == "TOP VIEW":
            crop_params = crop_params_top
        elif view.upper() == "SIDE VIEW":
            crop_params = crop_params_side
        else:
            print(f"Unknown view: {view}, skipping.")
            continue
        for day in os.listdir(view_path):
            day_path = os.path.join(view_path, day)
            if not os.path.isdir(day_path):
                continue
            # In test mode, save to a Test subfolder
            output_day_path = os.path.join(output_root, "Test", view, day) if test_mode else os.path.join(output_root, view, day)
            os.makedirs(output_day_path, exist_ok=True)
            processed_one = False
            for fname in os.listdir(day_path):
                if not fname.lower().endswith(IMAGE_EXTENSIONS):
                    continue
                in_fpath = os.path.join(day_path, fname)
                out_fpath = os.path.join(output_day_path, fname)
                try:
                    img = Image.open(in_fpath)
                    img = crop_image(img, crop_params)
                    img = crop_to_square_with_zoom_and_shift(
                        img,
                        zoom=SQUARE_CROP["zoom"],
                        x_shift=SQUARE_CROP["x_shift"],
                        y_shift=SQUARE_CROP["y_shift"]
                    )
                    img = img.resize(resize_to, Image.LANCZOS)
                    img.save(out_fpath)
                    print(f"Processed: {out_fpath}")
                except Exception as e:
                    print(f"Failed to process {in_fpath}: {e}")
                # In test mode, process only one image per day
                if test_mode:
                    processed_one = True
                    break
            # No need to break outer loop; we want one per day in test mode

if __name__ == "__main__":
    process_images(INPUT_ROOT, OUTPUT_ROOT, CROP_PARAMS_TOP, CROP_PARAMS_SIDE, RESIZE_TO, test_mode=TEST_MODE)
    print("Processing complete.")
