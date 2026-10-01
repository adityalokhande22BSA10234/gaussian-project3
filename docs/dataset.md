# Real capture dataset

The training set is **630 real instrument frames**, not rendered images. They
live in `new_dataset/` and are committed to the repo, so there is nothing to
generate or download:

```
new_dataset/
├── labels.csv                     # derived label index (regenerable)
├── preview.png                    # sample grid (gitignored, regenerate with `main.py preview`)
├── train1/                        # 420 frames, gs11-gs19
│   └── gs13/BPM1_ 1.png
└── test2/                         # 210 frames, gs21-gs26
    └── gs23/BPM3_ 12.png
```

## Image format

- 640x480 pixels, grayscale, 8-bit PNG
- The `filename` column in `labels.csv` is relative to `new_dataset/`, e.g.
  `test2/gs23/BPM3_ 12.png`
- Filenames contain spaces, which is why every path is handled with
  `pandas`/`pathlib` rather than naive string splitting.

## Labels are derived, not supplied

Real captures have no ground truth, so one label per frame is computed by
`data/real_dataset.py:extract_label`. Run it with:

```
python main.py labels
```

The pipeline, per frame:

1. **Blur** (`EXTRACT_BLUR_SIGMA = 3.0`) to suppress hot pixels.
2. **Mask the border** -- 40 px off each side column, 30 px off each top/bottom
   row. Most captures carry fixed bright edge bars that would otherwise win
   the blob contest below.
3. **Otsu threshold** the remaining frame into bright-pixel blobs.
4. **Score each connected component** and keep the winner
   (`data/real_dataset.py:_blob_score`):

   ```
   score = aspect^2 * fill_score * brightness
   ```

   where `aspect = short_side / long_side` of the component's `minAreaRect`
   (1.0 = perfectly round), `fill_score` is the contour area over its bounding
   rectangle normalised against pi/4 (what a thresholded elliptical Gaussian
   achieves), and `brightness` is the peak grey level over 255. Components
   under `EXTRACT_MIN_BLOB_FRAC` of the frame are rejected outright.
5. **Take intensity-weighted moments** of the winning blob
   (`data/real_dataset.py:_moments_to_params`) for the centre, the covariance
   matrix, and `theta`.

### Why sigmas are projected back to the image axes

The covariance eigen-decomposition gives principal-axis widths, but the model
regresses `(sigma_x, sigma_y)` and has no rotation output. The per-axis width is
therefore the root-sum-of-squares of both eigen-directions' contributions:

```python
sigma_x = math.hypot(major_axis[0] * sigma_major, minor_axis[0] * sigma_minor)
sigma_y = math.hypot(major_axis[1] * sigma_major, minor_axis[1] * sigma_minor)
```

An earlier version summed these linearly. That goes negative whenever the
covariance terms oppose, so the result clamped to the `MIN_SIGMA * 0.5` floor --
which silently pinned `sigma_x` to a single constant in **10 of 14 classes**.
The linear sum is not an orthogonal projection; `hypot` is.

Centres are clamped into the frame and sigmas into `[MIN_SIGMA * 0.5,
MAX_SIGMA_NORM]`, so a malformed frame cannot inject an unreachable target.

## Splits

`train1` and `test2` are the supplied split, and the two halves have
**disjoint gs classes**. `test2` is therefore held out whole rather than
subsampled -- a model can never have seen its class.

| Split | Source | Frames | Classes |
|---|---|---|---|
| `train` | `train1` | 379 | gs11-gs19 |
| `val` | `train1` | 41 | gs11-gs19 |
| `test` | `test2` | 210 | gs21, gs22, gs23, gs25, gs26 |

`train1` is carved into train/val per gs class at `--val-split` (default 0.1)
with a fixed seed, so every class is represented in both. All 630 frames are
used: no confidence filtering is applied, because dropping the hard frames
would hide exactly the problem worth measuring.

The test classes are unseen, so test MAE measures generalisation, not
memorisation. `--by-category` on `main.py evaluate` is what makes that
interpretable; a single aggregate test number hides which classes fail.

## Label schema (`labels.csv`)

| Column | Meaning |
|---|---|
| `filename` | Path relative to `new_dataset/` |
| `x0`, `y0` | Beam centre in pixels |
| `sigma_x`, `sigma_y` | Axis-aligned projected widths in pixels |
| `theta` | Blob rotation in radians, from the covariance eigen-decomposition. Metadata only -- not a regression target, since the model has no rotation output. |
| `has_beam` | `1` for every row here -- all 630 captures contain a beam. See the caveat below. |
| `category` | gs class folder name, e.g. `gs23` |
| `split` | `train` / `val` / `test` |

`has_beam` is uniformly `1`, so the model's presence output is a constant
`1.000` and carries no information. The head is kept because it is part of the
model's interface, but **presence accuracy must not be read as a quality
metric here**; MAE is the only meaningful one.

## Validation

`python main.py validate` checks that every referenced file exists and is a
valid 640x480 grayscale PNG, that no numeric label is NaN/Inf, that centres lie
inside the frame, and that sigmas are within range.

It does **not** check whether an indexed file is actually a beam frame. That gap
mattered: `preview.png` is a PNG sitting in the dataset root and was being
globbed as a capture, given a label of `y = 3772` on a 480 px frame, and
assigned to test. Validation passed throughout because nothing was corrupt --
the file was a valid image. `iter_capture_images` now matches only
`<train1|test2>/<class>/*.png`, so the structure itself enforces the invariant.

## Known limitations

- **Labels are a measurement, not ground truth.** Every target is derived by the
  same blob-selection heuristic the model must then reproduce. Where that
  heuristic picks a wrong blob, the label is simply wrong and the model is
  penalised for disagreeing with it. Roughly 74 of 630 frames (12%) derive a
  blob area more than double or less than half their own class median.
- **The gs folders mix acquisition sessions.** `gs23` contains filename groups
  whose derived `sigma_y` spans 15-117 px against a class median near 16 px --
  physically different beam setups sharing one folder name. `gs23` alone
  accounts for 85.1 px of `sigma_y` test MAE; the other four test classes sit
  between 1.7 and 49.7 px.
- **No beam-free frames.** Nothing here exercises the presence head. Any claim
  that the model can detect an empty frame is untested.
- **`preview.png` is not a capture** and is excluded structurally, not by
  convention.