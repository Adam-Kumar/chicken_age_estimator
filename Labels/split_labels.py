"""split_labels.py
Splits labels.csv into train/val/test CSV files based on unique drumette IDs (piece_id).

Assumptions:
- labels.csv lives in the same folder as this script (labels/), unless you pass --labels_csv
- labels.csv columns: relative_path, view, day, piece_id

Outputs:
- train.csv, val.csv, test.csv saved next to labels.csv

Usage:
    python split_labels.py --train_ratio 0.7 --val_ratio 0.15 --seed 42
"""

import argparse
import csv
import random
from pathlib import Path

DEFAULT_TRAIN_RATIO = 0.7
DEFAULT_VAL_RATIO = 0.15


def load_labels(csv_path: Path):
    """Return list of rows (dict) from CSV."""
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows


def split_piece_ids(piece_ids, train_ratio, val_ratio, seed):
    """Split piece_ids list into train/val/test sets."""
    random.Random(seed).shuffle(piece_ids)
    n = len(piece_ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_ids = set(piece_ids[:n_train])
    val_ids = set(piece_ids[n_train : n_train + n_val])
    test_ids = set(piece_ids[n_train + n_val :])
    return train_ids, val_ids, test_ids


def write_split(rows, ids_set, out_path: Path):
    if not rows:
        return
    fieldnames = rows[0].keys()
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([r for r in rows if r["piece_id"] in ids_set])


def main():
    parser = argparse.ArgumentParser(description="Split labels.csv into train/val/test by drumette ID")
    default_csv = Path(__file__).with_name("labels.csv")
    parser.add_argument(
        "--labels_csv",
        default=str(default_csv),
        help="Path to labels.csv (default: labels/labels.csv)",
    )
    parser.add_argument("--train_ratio", type=float, default=DEFAULT_TRAIN_RATIO)
    parser.add_argument("--val_ratio", type=float, default=DEFAULT_VAL_RATIO)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    csv_path = Path(args.labels_csv)
    rows = load_labels(csv_path)
    if not rows:
        print("No rows found in labels.csv.")
        return

    piece_ids = sorted({r["piece_id"] for r in rows})
    train_ids, val_ids, test_ids = split_piece_ids(piece_ids, args.train_ratio, args.val_ratio, args.seed)

    write_split(rows, train_ids, csv_path.with_name("train.csv"))
    write_split(rows, val_ids, csv_path.with_name("val.csv"))
    write_split(rows, test_ids, csv_path.with_name("test.csv"))

    print(
        f"Split complete. Train: {len(train_ids)} IDs, Val: {len(val_ids)} IDs, Test: {len(test_ids)} IDs."
    )


if __name__ == "__main__":
    main() 