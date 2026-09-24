#!/usr/bin/env python3
"""Import the BPM1 camera captures + their labels into the pipeline dataset.

The source labels file (columns, no header) has 22 numeric columns; the
images carry no embedded ground truth, so we must be told which columns are
(x, y, sigma_x, sigma_y).

!!! GUESS WARNING !!!
The values below were not verified by the user -- chosen by best guess:
  - COL_X0, COL_Y0: the two position-like columns (col1 ~196 mm, col2 ~1.66 mm)
  - COL_SX, COL_SY : the swept response column 22 (2.6 .. 4.5)
  - PIXELS_PER_MM = 1.0 treats "1 mm == 1 px" until the real frame size in
    mm is known (e.g. if the 640 px frame is 100 mm wide, set PIXELS_PER_MM=6.4).

Row i of the source CSV maps 1:1 to image "BPM1_ i.png" (row order).

The pipeline's BeamDataset reads labels.csv with schema
    filename, x0, y0, sigma_x, sigma_y, theta, has_beam, category, split
and normalizes x0/IMG_WIDTH(=640), y0/IMG_HEIGHT(=480), sigma/MAX_SIGMA_NORM(=120).

Usage:
    python scripts/import_bpm_dataset.py
    python scripts/import_bpm_dataset.py --pixels-per-mm 6.4 --no-backup
Note: run from gaussian-project/ so `import config` works.
"""
import argparse
import csv
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

# ---- BEST-GUESS column mapping (EDIT if you know the real schema) ------------
COL_X0 = 1    # 1-based column number for beam x position (mm)
COL_Y0 = 2    # 1-based column number for beam y position (mm)
COL_SX = 22   # 1-based column number for beam width sigma_x (mm)
COL_SY = 22   # 1-based column number for beam width sigma_y (mm)
PIXELS_PER_MM = 1.0  # calibration: real pixels per mm; 1.0 == "no conversion"
# -------------------------------------------------------------------------------


def _sort_key(path):
    return int(Path(path).stem.split("_ ")[-1])


def main():
    parser = argparse.ArgumentParser(description="Import BPM1 images + labels into the dataset.")
    parser.add_argument("--source-images", default=r"C:\protonaccelerator")
    parser.add_argument("--source-labels", default=r"C:\Users\hp\Downloads\labels.csv")
    parser.add_argument("--dataset-dir", default=str(config.DATASET_DIR))
    parser.add_argument("--category", default="bpm")
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument("--pixels-per-mm", type=float, default=PIXELS_PER_MM)
    parser.add_argument("--backup", action="store_true", default=True)
    parser.add_argument("--no-backup", dest="backup", action="store_false")
    args = parser.parse_args()

    src_dir = Path(args.source_images)
    src_csv = Path(args.source_labels)
    dataset_dir = Path(args.dataset_dir)
    images_dir = dataset_dir / "images"
    mm2px = args.pixels_per_mm

    image_paths = sorted((p for p in src_dir.glob("BPM1_*.png")
                          if "_ " in p.stem and p.stem.rsplit("_ ", 1)[-1].isdigit()),
                         key=_sort_key)
    if not image_paths:
        print(f"No BPM1_*.png files found in {src_dir}")
        sys.exit(1)

    df = pd.read_csv(src_csv, header=None)
    if len(df) != len(image_paths):
        print(f"Mismatch: {len(image_paths)} images vs {len(df)} label rows -- aborting.")
        sys.exit(1)
    print(f"Found {len(image_paths)} images and {len(df)} label rows (row i -> image i).")

    # Backup any existing synthetic dataset so it can be restored later.
    if args.backup and (dataset_dir / "labels.csv").exists():
        backup_dir = dataset_dir / f"_synthetic_backup_{len(list(dataset_dir.glob('_synthetic_backup_*')))}"
        backup_dir.mkdir(parents=True, exist_ok=True)
        if (dataset_dir / "labels.csv").exists():
            shutil.move(str(dataset_dir / "labels.csv"), str(backup_dir / "labels.csv"))
        if (dataset_dir / "images").exists():
            shutil.move(str(dataset_dir / "images"), str(backup_dir / "images"))
        print(f"Existing synthetic dataset backed up to {backup_dir}")

    cat_dir = images_dir / args.category
    cat_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, (img_path, (_, label)) in enumerate(zip(image_paths, df.iterrows()), start=1):
        x0_px = float(label[COL_X0 - 1]) * mm2px
        y0_px = float(label[COL_Y0 - 1]) * mm2px
        sx_px = float(label[COL_SX - 1]) * mm2px
        sy_px = float(label[COL_SY - 1]) * mm2px
        filename = f"{args.category}/{i:05d}_{args.category}.png"
        shutil.copy(str(img_path), str(cat_dir / f"{i:05d}_{args.category}.png"))
        rows.append({
            "filename": filename,
            "x0": x0_px,
            "y0": y0_px,
            "sigma_x": sx_px,
            "sigma_y": sy_px,
            "theta": 0.0,
            "has_beam": 1,
            "category": args.category,
        })

    # Same 80/10/10 seeded shuffle the synthetic builder uses.
    random.Random(args.seed).shuffle(rows)
    n = len(rows)
    n_train = int(n * config.TRAIN_SPLIT)
    n_val = int(n * config.VAL_SPLIT)
    for j, row in enumerate(rows):
        row["split"] = "train" if j < n_train else ("val" if j < n_train + n_val else "test")

    labels_path = dataset_dir / "labels.csv"
    with open(labels_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Done. {len(rows)} BPM images copied to {cat_dir}")
    print(f"Labels written to {labels_path}")
    print("\nLabel ranges (converted to px with pixels_per_mm={:.3f}):".format(mm2px))
    lab = pd.read_csv(labels_path)
    for c in ["x0", "y0", "sigma_x", "sigma_y"]:
        print(f"  {c}: {lab[c].min():.2f} .. {lab[c].max():.2f}")
    print("\nSplit counts:", lab["split"].value_counts().to_dict())
    print("\nCAUTION (guess-based):")
    print("  - Mapping and mm->pixel calibration are UNVERIFIED; edit COL_*/PIXELS_PER_MM")
    print("    at the top of this script if the real schema is different.")
    print("  - sigma range is far below the synthetic [15, 90] px the CNNs were tuned for.")
    print("\nNext:  python main.py train --model tiny --epochs 30")


if __name__ == "__main__":
    main()