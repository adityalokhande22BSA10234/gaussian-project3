# Architecture

## Pipeline overview

The project regresses 2D laser beam parameters -- centre `(x, y)` and spread
`(sigma_x, sigma_y)` -- from beam-camera frames. It trains on the **real
captures** in `new_dataset/`, using labels derived from those same frames:

1. **Label extraction** (`data/real_dataset.py`) -- for each capture, mask the
   frame border, Otsu-threshold, score the connected blobs on roundness x
   ellipse fill x brightness, and take intensity-weighted moments of the winner.
   Writes `new_dataset/labels.csv` plus the train/val/test assignment.
2. **CNN training** (`models/`, `training/`) -- a convolutional backbone
   (custom ConvNet or ResNet) with a sigmoid-bounded 4-output regression head,
   trained with a masked Smooth L1 loss against normalized targets.
3. **Inference** (`inference/`) -- feed the full frame resized to 128x128 and
   denormalize the predictions back to pixel units.

`main.py` exposes all of it behind one CLI; `app.py` exposes the same operations
as a Streamlit dashboard.

## Project structure

```
.
├── main.py                    # CLI entry point (all/labels/generate/validate/preview/train/evaluate/infer)
├── app.py                     # Streamlit dashboard; also launchable as plain `python app.py`
├── config.py                  # all tunable constants (paths, geometry, extraction, hyperparams)
├── requirements.txt
├── data/
│   ├── real_dataset.py        # label extraction from captures + labels.csv builder
│   ├── dataset.py             # PyTorch Dataset
│   ├── validate.py            # dataset quality checks (image integrity, label sanity)
│   └── preview.py             # saves a per-category sample grid image
├── models/
│   └── cnn.py                 # 3 CNN architectures: tiny / medium / resnet18
├── training/
│   ├── train.py               # training loop
│   └── evaluate.py            # pixel-space MAE, aggregate and per gs class
├── inference/
│   ├── preprocess.py          # background subtraction + full-frame resize
│   └── infer.py               # runs a trained model on an image
├── scripts/                   # thin CLI wrappers around each module
├── synthetic/                 # legacy generator, used only by `main.py generate`
├── docs/                      # this documentation
├── new_dataset/               # 630 real captures + derived labels.csv (committed)
├── dataset/                   # legacy generated synthetic images
├── checkpoints/               # trained weights, <model>_best.pt (gitignored)
└── Project_Progress_Report.pdf # step-by-step writeup of the work
```

## Module map

| Module | Responsibility |
|---|---|
| `config.py` | Single source of truth for paths, image geometry, label-extraction thresholds, and training hyperparameters. `DATASET_DIR` and `SYNTHETIC_DIR` are deliberately distinct so generated images can never mix into the real capture set. |
| `data/real_dataset.py` | Derives one label per capture (`extract_label`), scores candidate blobs (`_blob_score`), computes moments (`_moments_to_params`), enumerates captures (`iter_capture_images`), and writes `labels.csv` with the split assignment. |
| `data/dataset.py` | `BeamDataset(torch.utils.data.Dataset)` -- loads an image + normalized target from `labels.csv`, resizing the whole frame to 128x128. |
| `data/validate.py` | Checks every referenced image exists, is 640x480 grayscale, and that labels have no NaNs or out-of-range values. |
| `data/preview.py` | Saves a matplotlib grid of sample images per class for a visual check. |
| `models/cnn.py` | The three interchangeable CNN backbones + regression heads (see [models.md](models.md)). |
| `training/train.py` | Training loop: masked Smooth L1 + BCE, best-checkpoint saving, MPS/CUDA/CPU detection. Does not seed torch. |
| `training/evaluate.py` | Runs a checkpoint over a split; reports pixel-space MAE and, with `--by-category`, per-gs-class MAE. |
| `inference/preprocess.py` | Background subtraction and full-frame resize. `find_blob_bbox` is display-only. |
| `inference/infer.py` | Loads a checkpoint, preprocesses an image, returns `(x, y, sigma_x, sigma_y)` in pixel coordinates. |
| `main.py` | argparse CLI wiring all of the above, plus the default `all` pipeline (extract labels if missing -> train -> summary). |
| `app.py` | Streamlit dashboard: Home / Dataset / Training / Evaluation / Inference, with training and evaluation as background jobs. |

## Key design decisions

| Decision | Why |
|---|---|
| Train on the real captures, not the synthetic set | Synthetic frames are clean Gaussians; the captures are instrument output with edge bars, hot pixels, and variable gain. Training on the deployment distribution mattered far more than having exact ground truth. |
| Derive labels per frame instead of filtering frames | Real captures have no ground truth, so a target had to be computed somehow. All 630 frames are kept -- dropping the hard ones would hide the very failure mode worth measuring. |
| Hold `test2` out whole, not subsampled | `train1` and `test2` have disjoint gs classes, so any subsampling of `test2` still measures unseen-class generalization. Splitting it would waste that. |
| Carve train/val per gs class | Otherwise a rare class could land wholly in val and make validation loss meaningless. |
| Full frame at inference, matching training | Cropping to the blob before resizing was a train/inference distribution mismatch that made real-frame predictions incomparable to evaluation numbers. See [inference.md](inference.md). |
| Project sigmas with `hypot`, not a linear sum | A linear sum goes negative for rotated blobs and clamps to the floor, pinning `sigma_x` to one constant in 10 of 14 classes. |
| Match capture files structurally, not by convention | `iter_capture_images` accepts only `<train1\|test2>/<class>/*.png`, so a generated `preview.png` cannot be indexed as a frame. |
| Keep the synthetic generator, isolated | `main.py generate` still works for experimentation, but writes to `dataset/`, never `new_dataset/`. |

## Reading the results honestly

`docs/dataset.md` documents why aggregate test MAE overstates the problem. The
short version: beam widths are recovered to ~1.6 px on validation, test `x`
error rises to ~82 px, and per-class evaluation shows most of that is
concentrated in `gs23`, whose own derived labels are outliers. The three
architectures score within 0.4 px of each other across a 72x parameter range,
which points at label quality rather than capacity as the limiting factor.
Treat presence accuracy as uninformative -- it is a constant on this data.