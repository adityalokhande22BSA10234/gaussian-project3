"""Label extraction for the real BPM captures in `new_dataset`.

The captures are instrument frames (640x480 grayscale) rather than clean
synthetic Gaussians, so per-image labels are derived from each frame:

    1. light Gaussian blur to suppress hot pixels,
    2. the fixed frame border is zeroed (most captures carry bright edge bars),
    3. Otsu threshold on the remaining frame gives bright-pixel blobs,
    4. connected components are scored on roundness, ellipse fill and peak
       brightness, and the best-scoring blob is kept,
    5. background-subtracted intensity moments of that blob give the centre,
       the principal-axis sigmas, and the blob's rotation `theta`.

`build_label_index()` walks `train1/` and `test2/`, extracts one label per
PNG and writes the `labels.csv` the training pipeline expects. `train1` is
carved into train/val per gs class; `test2` is kept whole as the test split
because its classes do not overlap `train1`.
"""
import argparse
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import config

CSV_COLUMNS = [
    "filename", "x0", "y0", "sigma_x", "sigma_y", "theta",
    "has_beam", "category", "split",
]


def _mask_border(frame, margin_x, margin_y):
    """Zero out `margin_x` columns / `margin_y` rows on each edge."""
    frame[:margin_y, :] = 0
    frame[-margin_y:, :] = 0
    frame[:, :margin_x] = 0
    frame[:, -margin_x:] = 0
    return frame


def _blob_score(component, peak, min_area):
    """Rank a connected component: prefer compact, elliptical, bright blobs."""
    area = float(cv2.contourArea(component))
    if area < min_area:
        return 0.0

    (_, _), (w, h), _ = cv2.minAreaRect(component)
    if w <= 0 or h <= 0:
        return 0.0
    side_long, side_short = max(w, h), min(w, h)
    if side_long <= 0:
        return 0.0

    aspect = side_short / side_long                 # 1.0 == perfectly round
    fill = area / max(w * h, 1.0)                   # 1.0 == fills its rect
    fill_score = min(fill / config.EXTRACT_ELLIPSE_FILL, 1.0)
    brightness = peak / 255.0

    return (aspect ** 2) * fill_score * brightness


def _moments_to_params(mask, gray):
    """Intensity-weighted moments of the selected blob.

    Returns (x0, y0, sigma_x, sigma_y, theta) with sigmas clipped into
    [MIN_SIGMA/2, MAX_SIGMA_NORM] so the sigmoid-bounded model head can
    represent them.
    """
    weights = cv2.GaussianBlur(
        gray.astype(np.float32), (0, 0), config.EXTRACT_BLUR_SIGMA
    )
    weights = weights * (mask > 0)
    total = float(weights.sum())
    if total <= 0.0:
        h, w = mask.shape
        return w / 2.0, h / 2.0, config.MIN_SIGMA, config.MIN_SIGMA, 0.0

    ys, xs = np.nonzero(weights)
    vals = weights[ys, xs]

    x0 = float((xs * vals).sum() / total)
    y0 = float((ys * vals).sum() / total)

    # A centroid of an in-frame blob cannot fall outside the frame, so this
    # only ever triggers on a malformed input. Clamping keeps the label inside
    # the range the regression head can represent instead of letting an
    # unreachable target silently poison training.
    x0 = float(np.clip(x0, 0.0, config.IMG_WIDTH))
    y0 = float(np.clip(y0, 0.0, config.IMG_HEIGHT))

    var_x = float(((xs - x0) ** 2 * vals).sum() / total)
    var_y = float(((ys - y0) ** 2 * vals).sum() / total)
    cov_xy = float(((xs - x0) * (ys - y0) * vals).sum() / total)

    cov = np.array([[var_x, cov_xy], [cov_xy, var_y]], dtype=np.float64)
    theta = 0.5 * math.atan2(2.0 * cov_xy, var_x - var_y)

    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, 1.0)
    major_axis = eigvecs[:, 1]
    minor_axis = eigvecs[:, 0]
    sigma_major = math.sqrt(eigvals[1])
    sigma_minor = math.sqrt(eigvals[0])

    # Project the per-axis sigmas back onto the image x/y frame. The std along
    # each frame axis is the root-sum-of-squares of the per-axis contributions
    # (a linear sum would go negative for rotated blobs and clamp to the floor).
    sigma_x = math.hypot(major_axis[0] * sigma_major, minor_axis[0] * sigma_minor)
    sigma_y = math.hypot(major_axis[1] * sigma_major, minor_axis[1] * sigma_minor)

    lo = config.MIN_SIGMA * 0.5
    hi = config.MAX_SIGMA_NORM
    sigma_x = float(np.clip(sigma_x, lo, hi))
    sigma_y = float(np.clip(sigma_y, lo, hi))

    return x0, y0, sigma_x, sigma_y, float(theta)


def extract_label(image_path):
    """Derive (x0, y0, sigma_x, sigma_y, theta, score) for one capture.

    Returns None if no blob passes the minimum-area filter, i.e. the frame is
    effectively beamless.
    """
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    blurred = cv2.GaussianBlur(
        gray, (0, 0), config.EXTRACT_BLUR_SIGMA
    )
    masked = _mask_border(
        blurred.copy(), config.EXTRACT_BORDER_X, config.EXTRACT_BORDER_Y
    )

    if masked.max() <= 0:
        return None

    _, binary = cv2.threshold(masked, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    num, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    min_area = config.EXTRACT_MIN_BLOB_FRAC * gray.size

    best = None
    best_score = 0.0
    for idx in range(1, num):
        component_mask = (labels == idx).astype(np.uint8)
        contours, _ = cv2.findContours(
            component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue
        component = max(contours, key=cv2.contourArea)
        peak = float(gray[component_mask > 0].max())
        score = _blob_score(component, peak, min_area)
        if score > best_score:
            best_score = score
            best = component_mask

    if best is None:
        return None

    x0, y0, sigma_x, sigma_y, theta = _moments_to_params(best, gray)
    return x0, y0, sigma_x, sigma_y, theta, best_score


def iter_capture_images(images_dir=None):
    """Yield every capture PNG, i.e. `<split>/<gs class>/*.png`.

    The match is deliberately structural: only files inside the two split
    directories are considered. Generated artifacts that happen to live in the
    dataset root -- `labels.csv`, and especially the preview grid PNG, which is
    a tall montage rather than a frame -- are therefore never mistaken for
    captures and indexed as if they were.
    """
    images_dir = Path(images_dir or config.IMAGES_DIR)
    for split in (config.TRAIN_SUBDIR, config.TEST_SUBDIR):
        split_dir = images_dir / split
        if not split_dir.is_dir():
            continue
        yield from sorted(split_dir.rglob("*.png"))


def _assign_train_val_splits(frame, val_split, seed):
    """Carve `train1` into train/val per gs class, stratified."""
    rng = np.random.default_rng(seed)
    splits = []
    for _, group in frame.groupby("category", sort=True):
        idx = group.index.to_numpy().copy()
        rng.shuffle(idx)
        n_val = int(round(len(idx) * val_split))
        n_val = min(max(n_val, 1), len(idx) - 1)
        val_idx = set(idx[:n_val].tolist())
        splits.extend(
            "val" if i in val_idx else "train" for i in group.index
        )
    frame = frame.copy()
    frame["split"] = splits
    return frame


def build_label_index(images_dir=None, labels_csv=None, val_split=None, seed=None):
    """Extract labels for every capture under `images_dir` and write the CSV.

    Returns the written DataFrame. Rows are emitted in a deterministic order
    (sorted by filename) and always cover every PNG found, so the index stays
    a 1:1 map onto the dataset.
    """
    images_dir = Path(images_dir or config.IMAGES_DIR)
    labels_csv = Path(labels_csv or config.LABELS_CSV)
    val_split = config.VAL_SPLIT if val_split is None else val_split
    seed = config.RANDOM_SEED if seed is None else seed

    records = []
    failures = []
    for path in iter_capture_images(images_dir):
        relative = path.relative_to(images_dir).as_posix()
        category = path.parent.name
        try:
            extracted = extract_label(path)
        except Exception as exc:  # unreadable file -> record a beamless row
            failures.append(f"{relative}: {exc}")
            extracted = None

        if extracted is None:
            failures.append(f"{relative}: no beam-like blob found")
            h, w = config.IMG_HEIGHT, config.IMG_WIDTH
            records.append({
                "filename": relative,
                "x0": w / 2.0, "y0": h / 2.0,
                "sigma_x": config.MIN_SIGMA, "sigma_y": config.MIN_SIGMA,
                "theta": 0.0, "has_beam": 0, "category": category,
            })
        else:
            x0, y0, sigma_x, sigma_y, theta, _score = extracted
            records.append({
                "filename": relative,
                "x0": x0, "y0": y0,
                "sigma_x": sigma_x, "sigma_y": sigma_y,
                "theta": theta, "has_beam": 1, "category": category,
            })

    frame = pd.DataFrame.from_records(records, columns=CSV_COLUMNS)

    is_train_dir = frame["filename"].str.startswith(f"{config.TRAIN_SUBDIR}/")
    frame["split"] = "test"
    frame.loc[is_train_dir, "split"] = "train"
    frame = _assign_train_val_splits(
        frame[is_train_dir].copy(), val_split, seed
    )
    train_val_rows = frame
    test_rows = pd.DataFrame.from_records(records, columns=CSV_COLUMNS)
    test_rows = test_rows[~test_rows["filename"].str.startswith(f"{config.TRAIN_SUBDIR}/")]
    test_rows["split"] = "test"

    out = pd.concat([train_val_rows, test_rows], ignore_index=True)
    out = out.sort_values("filename").reset_index(drop=True)

    if len(out) != len(records):
        raise RuntimeError(
            f"Index row mismatch: {len(out)} rows for {len(records)} images"
        )

    labels_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(labels_csv, index=False)

    counts = out["split"].value_counts().to_dict()
    print(f"Wrote {len(out)} label rows -> {labels_csv}")
    print(f"  split counts: {counts}")
    print(f"  beamless rows: {int((out['has_beam'] == 0).sum())}")
    if failures:
        print(f"  {len(failures)} image(s) without a usable blob:")
        for message in failures[:20]:
            print(f"    - {message}")
        if len(failures) > 20:
            print(f"    ... and {len(failures) - 20} more")
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Build labels.csv for the real BPM captures."
    )
    parser.add_argument("--images-dir", default=str(config.IMAGES_DIR))
    parser.add_argument("--labels-csv", default=str(config.LABELS_CSV))
    parser.add_argument("--val-split", type=float, default=config.VAL_SPLIT)
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    args = parser.parse_args()
    build_label_index(args.images_dir, args.labels_csv, args.val_split, args.seed)


if __name__ == "__main__":
    main()