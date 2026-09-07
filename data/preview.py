"""Sanity-check helper: plot a grid of generated samples from each category."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

import config


def save_preview_grid(labels_csv=None, images_dir=None, per_category=4, output=None):
    labels_csv = Path(labels_csv or config.LABELS_CSV)
    images_dir = Path(images_dir or config.IMAGES_DIR)
    output = str(output or config.DATASET_DIR / "preview.png")

    df = pd.read_csv(labels_csv)
    categories = df["category"].unique()

    fig, axes = plt.subplots(len(categories), per_category,
                              figsize=(3 * per_category, 3 * len(categories)))
    for row, category in enumerate(categories):
        subset = df[df["category"] == category].sample(
            n=min(per_category, len(df[df["category"] == category])), random_state=0
        )
        for col, (_, sample) in enumerate(subset.iterrows()):
            ax = axes[row, col] if len(categories) > 1 else axes[col]
            img = Image.open(images_dir / sample["filename"])
            ax.imshow(img, cmap="gray")
            title = category
            if sample["has_beam"]:
                title += f"\n(x={sample['x0']:.0f}, y={sample['y0']:.0f})"
            ax.set_title(title, fontsize=9)
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(output, dpi=120)
    plt.close(fig)
    print(f"Saved preview grid -> {output}")
    return output


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Save a preview grid of sample dataset images.")
    parser.add_argument("--labels-csv", default=str(config.LABELS_CSV))
    parser.add_argument("--images-dir", default=str(config.IMAGES_DIR))
    parser.add_argument("--per-category", type=int, default=4)
    parser.add_argument("--output", default=str(config.DATASET_DIR / "preview.png"))
    args = parser.parse_args()
    save_preview_grid(args.labels_csv, args.images_dir, args.per_category, args.output)


if __name__ == "__main__":
    main()
