"""Builds the full synthetic dataset (images + labels.csv) to disk."""
import argparse
import csv
import random
from pathlib import Path

import numpy as np
from PIL import Image

import config
from .generator import (
    make_background_sample,
    make_noisy_sample,
    make_normal_sample,
    make_tilted_sample,
)

GENERATORS = {
    "normal": make_normal_sample,
    "tilted": make_tilted_sample,
    "noisy": make_noisy_sample,
    "background": make_background_sample,
}


def _assign_splits(rows, seed):
    random.Random(seed).shuffle(rows)
    n = len(rows)
    n_train = int(n * config.TRAIN_SPLIT)
    n_val = int(n * config.VAL_SPLIT)
    for i, row in enumerate(rows):
        if i < n_train:
            row["split"] = "train"
        elif i < n_train + n_val:
            row["split"] = "val"
        else:
            row["split"] = "test"
    return rows


def build_dataset(output_dir=None, counts=None, seed=config.RANDOM_SEED):
    output_dir = Path(output_dir or config.DATASET_DIR)
    images_dir = output_dir / "images"

    counts = counts or {
        "normal": config.NUM_NORMAL,
        "tilted": config.NUM_TILTED,
        "noisy": config.NUM_NOISY,
        "background": config.NUM_BACKGROUND,
    }
    for category in counts:
        (images_dir / category).mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    rows = []
    idx = 0
    for category, count in counts.items():
        gen_fn = GENERATORS[category]
        for _ in range(count):
            sample = gen_fn(rng)
            filename = f"{category}/{idx:05d}_{category}.png"
            Image.fromarray(sample.image).save(images_dir / filename)
            rows.append({
                "filename": filename,
                "x0": sample.x0,
                "y0": sample.y0,
                "sigma_x": sample.sigma_x,
                "sigma_y": sample.sigma_y,
                "theta": sample.theta,
                "has_beam": sample.has_beam,
                "category": sample.category,
            })
            idx += 1
            if idx % 100 == 0:
                print(f"  generated {idx} images...")

    rows = _assign_splits(rows, seed)

    labels_path = output_dir / "labels.csv"
    with open(labels_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Done. {len(rows)} images written to {images_dir}")
    print(f"Labels written to {labels_path}")
    return labels_path


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic Gaussian beam dataset.")
    parser.add_argument("--output-dir", default=str(config.DATASET_DIR))
    parser.add_argument("--normal", type=int, default=config.NUM_NORMAL)
    parser.add_argument("--tilted", type=int, default=config.NUM_TILTED)
    parser.add_argument("--noisy", type=int, default=config.NUM_NOISY)
    parser.add_argument("--background", type=int, default=config.NUM_BACKGROUND)
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    args = parser.parse_args()

    counts = {
        "normal": args.normal,
        "tilted": args.tilted,
        "noisy": args.noisy,
        "background": args.background,
    }
    build_dataset(output_dir=args.output_dir, counts=counts, seed=args.seed)


if __name__ == "__main__":
    main()
