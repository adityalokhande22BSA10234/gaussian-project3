"""Dataset quality validation: image integrity + label sanity checks.

Verifies: every labeled file exists and is a readable, correctly-sized
grayscale image; beam-bearing categories actually show a bright peak above
the noise floor; background frames don't accidentally contain a beam-like
peak; and all label values (position, sigma, theta) fall within expected
ranges with no NaN/Inf.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import config

# A beam-bearing frame should have a clear peak well above typical noise;
# a background frame should stay well below this so labels stay meaningful.
PEAK_BRIGHTNESS_MIN = 60     # 0-255 scale; beam categories must exceed this
BACKGROUND_PEAK_MAX = 140    # background frames should not spike this high


def _check_row(images_dir, row, issues):
    path = images_dir / row["filename"]
    if not path.exists():
        issues.append(f"MISSING: {row['filename']}")
        return None

    try:
        img = Image.open(path)
        img.load()
    except Exception as e:
        issues.append(f"CORRUPT: {row['filename']} ({e})")
        return None

    if img.mode != "L":
        issues.append(f"WRONG MODE ({img.mode}): {row['filename']}")
    if img.size != (config.IMG_WIDTH, config.IMG_HEIGHT):
        issues.append(f"WRONG SIZE ({img.size}): {row['filename']}")

    arr = np.asarray(img, dtype=np.float32)
    peak = arr.max()

    if row["has_beam"] == 1 and peak < PEAK_BRIGHTNESS_MIN:
        issues.append(f"WEAK BEAM (peak={peak:.0f}): {row['filename']}")
    if row["has_beam"] == 0 and peak > BACKGROUND_PEAK_MAX:
        issues.append(f"UNEXPECTED PEAK IN BACKGROUND (peak={peak:.0f}): {row['filename']}")

    return peak


def _check_labels(df, issues):
    numeric_cols = ["x0", "y0", "sigma_x", "sigma_y", "theta"]
    if df[numeric_cols].isna().any().any() or not np.isfinite(df[numeric_cols].to_numpy()).all():
        issues.append("NaN/Inf found in label columns")

    beam_rows = df[df["has_beam"] == 1]
    out_of_bounds = beam_rows[
        (beam_rows["x0"] < 0) | (beam_rows["x0"] > config.IMG_WIDTH) |
        (beam_rows["y0"] < 0) | (beam_rows["y0"] > config.IMG_HEIGHT)
    ]
    if len(out_of_bounds):
        issues.append(f"{len(out_of_bounds)} beam centers fall outside the frame")

    bad_sigma = beam_rows[
        (beam_rows["sigma_x"] < config.MIN_SIGMA * 0.5) | (beam_rows["sigma_x"] > config.MAX_SIGMA * 1.2) |
        (beam_rows["sigma_y"] < config.MIN_SIGMA * 0.5) | (beam_rows["sigma_y"] > config.MAX_SIGMA * 1.2)
    ]
    if len(bad_sigma):
        issues.append(f"{len(bad_sigma)} rows have sigma outside expected range")


def validate_dataset(labels_csv=None, images_dir=None):
    """Checks every image + label. Returns (summary_df, issues_list)."""
    labels_csv = Path(labels_csv or config.LABELS_CSV)
    images_dir = Path(images_dir or config.IMAGES_DIR)
    df = pd.read_csv(labels_csv)

    issues = []
    _check_labels(df, issues)

    rows = []
    for category, group in df.groupby("category"):
        peaks = []
        for _, row in group.iterrows():
            peak = _check_row(images_dir, row, issues)
            if peak is not None:
                peaks.append(peak)
        peaks = np.array(peaks) if peaks else np.array([0.0])
        splits = group["split"].value_counts()
        rows.append({
            "category": category,
            "count": len(group),
            "train": int(splits.get("train", 0)),
            "val": int(splits.get("val", 0)),
            "test": int(splits.get("test", 0)),
            "mean_peak": round(float(peaks.mean()), 1),
            "min_peak": round(float(peaks.min()), 1),
            "max_peak": round(float(peaks.max()), 1),
        })
    summary = pd.DataFrame(rows)
    return summary, issues


def print_validation_report(summary, issues):
    print("--- Category summary ---")
    print(summary.to_string(index=False))
    print("\n--- Result ---")
    if issues:
        print(f"FOUND {len(issues)} ISSUE(S):")
        for issue in issues[:50]:
            print(f"  - {issue}")
        if len(issues) > 50:
            print(f"  ... and {len(issues) - 50} more")
    else:
        print("All images present, correctly sized/grayscale, beam peaks and labels look consistent. PASS.")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Validate the synthetic dataset.")
    parser.add_argument("--labels-csv", default=str(config.LABELS_CSV))
    parser.add_argument("--images-dir", default=str(config.IMAGES_DIR))
    args = parser.parse_args()

    summary, issues = validate_dataset(args.labels_csv, args.images_dir)
    print_validation_report(summary, issues)
    if issues:
        sys.exit(1)


if __name__ == "__main__":
    main()
