# fmt: off
import os
from pathlib import Path

import pandas as pd

# Attempt to reuse sanitisation helper if the module exists in path. We keep a runtime
# fallback plus a "type: ignore" so linters don't complain when the module isn't around.
try:
    from fix_labels import sanitize_view  # type: ignore  # noqa: E402, F401
except ModuleNotFoundError:
    def sanitize_view(v: str) -> str:  # noqa: D401
        v = v.strip().upper()
        if v in {"TOP", "TOP VIEW"}:
            return "TOP VIEW"
        if v in {"SIDE", "SIDE VIEW"}:
            return "SIDE VIEW"
        return v
# fmt: on

# === CONFIGURATION ===
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # Go up from labels/ to project root
DATA_ROOT = PROJECT_ROOT / "Dataset_Processed"  # root folder containing TOP VIEW/ SIDE VIEW subfolders
OUTPUT_CSV = str(Path(__file__).with_name("labels.csv"))  # always inside labels folder

# Optional: adjust to ignore the Test folder when test mode outputs exist
IGNORE_TEST_SUBFOLDER = True

# Assumptions:
# - Folder structure: Processed/<VIEW>/Day X/<filename>.jpg
# - <filename> without extension is the piece id (1..82)
# - Day folders are named "Day Y" where Y is the day label (1..7)

# Collect label rows -----------------------------------------------------------
rows: list[dict] = []
for view in os.listdir(DATA_ROOT):
    if IGNORE_TEST_SUBFOLDER and view == "Test":
        continue
    view_path = os.path.join(DATA_ROOT, view)
    if not os.path.isdir(view_path):
        continue

    for day_folder in os.listdir(view_path):
        day_path = os.path.join(view_path, day_folder)
        if not os.path.isdir(day_path):
            continue
        # Extract day index
        try:
            day_idx = int(day_folder.split()[1])  # "Day 3" → 3
        except (IndexError, ValueError):
            print(f"Skipping folder with unexpected name: {day_folder}")
            continue

        for fname in os.listdir(day_path):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            piece_id = os.path.splitext(fname)[0]  # "1.jpg" → "1"
            rel_path = os.path.join(view, day_folder, fname).replace("\\", "/")  # POSIX style
            rows.append({
                "relative_path": rel_path,
                "view": view,
                "day": day_idx,
                "piece_id": piece_id,
            })

# ----- sanitise & save --------------------------------------------------------

if not rows:
    raise SystemExit("No image files found; ensure DATA_ROOT is correct.")

df = pd.DataFrame(rows)

# Normalise / validate using same rules as fix_labels
df["view"] = df["view"].apply(sanitize_view)
df["day"] = pd.to_numeric(df["day"], errors="raise").astype(int)  # type: ignore[attr-defined]
df["piece_id"] = pd.to_numeric(df["piece_id"], errors="raise").astype(int)  # type: ignore[attr-defined]
df["relative_path"] = df["relative_path"].str.replace("\\\\", "/", regex=True)

df = df.sort_values(["view", "day", "piece_id"], ignore_index=True)

df.to_csv(OUTPUT_CSV, index=False)
print(f"{Path(OUTPUT_CSV).name} created with {len(df)} rows.") 