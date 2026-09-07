# Architecture

## Pipeline overview

The project implements a 3-stage pipeline for regressing 2D Gaussian laser
beam parameters -- center `(x, y)` and spread `(sigma_x, sigma_y)` -- from
camera images, using only synthetic training data:

1. **Synthetic dataset generation & annotation** (`synthetic/`) -- render
   2D Gaussian beam images with diverse sizes, shifted centers,
   rotation/tilt, noise, and beam-free background frames, each with
   continuous ground-truth parameters.
2. **CNN model training** (`models/`, `training/`) -- a convolutional
   backbone (custom ConvNet or ResNet) feeds a regression head ending in 4
   sigmoid-bounded outputs, trained with a masked Smooth L1 loss against
   normalized ground truth.
3. **Real image preprocessing & inference** (`inference/`) -- background
   subtraction, ROI cropping, and running the trained model on a real
   camera capture to recover calibrated beam parameters.

`main.py` ties all three stages together behind a single CLI (see the
[repo-root README](../README.md) for how to run it).

## Project structure

```
.
├── main.py                    # CLI entry point (all/generate/validate/preview/train/evaluate/infer)
├── config.py                  # all tunable constants (paths, sizes, counts, hyperparams)
├── requirements.txt
├── synthetic/
│   ├── generator.py           # 2D Gaussian rendering + noise/gradient augmentations
│   └── dataset_builder.py     # writes images/labels.csv to disk
├── data/
│   ├── dataset.py             # PyTorch Dataset
│   ├── validate.py            # dataset quality checks (image integrity, label sanity)
│   └── preview.py             # saves a per-category sample grid image
├── models/
│   └── cnn.py                 # 3 CNN architectures: tiny / medium / resnet18
├── training/
│   ├── train.py                # training loop
│   └── evaluate.py            # pixel-space MAE + presence accuracy
├── inference/
│   ├── preprocess.py          # background subtraction + ROI cropping
│   └── infer.py                # runs a trained model on a real image
├── scripts/                   # thin CLI wrappers around each module (main.py covers the same ground)
├── docs/                      # this documentation
├── dataset/                   # generated images + labels.csv (gitignored)
└── checkpoints/               # trained model weights, <model>_best.pt (gitignored)
```

## Module map

| Module | Responsibility |
|---|---|
| `config.py` | Single source of truth for every tunable constant: image size, dataset counts, sigma ranges, training hyperparameters, checkpoint paths. |
| `synthetic/generator.py` | Renders a single 2D (optionally rotated) Gaussian image plus noise/gradient/hot-pixel augmentations. One function per dataset category. |
| `synthetic/dataset_builder.py` | Drives `generator.py` to build the full dataset, writes PNGs organized by category subfolder, and `labels.csv` with an 80/10/10 train/val/test split. |
| `data/dataset.py` | `BeamDataset(torch.utils.data.Dataset)` -- loads an image + normalized target tensor from `labels.csv`. |
| `data/validate.py` | Checks every image exists, is the right size/mode, has a sane peak brightness for its category, and that labels have no NaNs/out-of-range values. |
| `data/preview.py` | Saves a matplotlib grid of sample images per category, for a quick visual sanity check. |
| `models/cnn.py` | The three interchangeable CNN backbones + regression heads (see [models.md](models.md)). |
| `training/train.py` | Training loop: masked Smooth L1 + BCE loss, best-checkpoint saving, MPS/CUDA/CPU device auto-detection. |
| `training/evaluate.py` | Runs a checkpoint over a split and reports pixel-space MAE + beam-presence accuracy. |
| `inference/preprocess.py` | Background subtraction and ROI (region-of-interest) cropping for a real camera image. |
| `inference/infer.py` | Loads a checkpoint, preprocesses a real image, and returns calibrated `(x, y, sigma_x, sigma_y)` in real pixel coordinates. |
| `main.py` | argparse CLI wiring all of the above together, plus the default `all` pipeline (generate-if-missing -> train all models -> summary). |

## Deliberate deviations from the original spec

The original spec (pasted at the start of this project) gave example
values, not hard requirements. A few choices in this implementation
diverge from those examples, each for a specific reason:

| Spec example | This implementation | Why |
|---|---|---|
| ~800 normal / 75 tilted / 100-150 noisy / 100 background (~1,100 total) | 1,500 / 75 / 250 / 175 = **2,000 total** | Explicitly requested partway through development; still within the spec's stated "1,000-6,500" range. |
| Downsample to e.g. 224x224 | Downsampled to **128x128** | Keeps training compute light enough for a laptop CPU/MPS GPU; the spec gave 224 only as an example, not a requirement. |
| 4 regression outputs (x, y, sigma_x, sigma_y) | 4 regression outputs **+ a 5th beam-presence logit** | The spec's own dataset breakdown includes beam-free background/ambient frames. A pure 4-output regressor has no honest target for those frames (there's no beam to locate); the presence logit lets the model learn to say "no beam" instead of hallucinating coordinates. |
| Background subtraction via subtraction/thresholding | Subtracts a *measured* background frame when one is supplied; otherwise applies a **light median filter** instead of self-blur-subtraction | Testing found that blurring the image itself to estimate a "background" actively erases wide beams (sigma up to 90px, comparable to or larger than a typical blur kernel), destroying the very signal being measured. A median filter still removes isolated hot-pixel noise without this failure mode. |

One more fix worth noting: for `tilted` (rotated) samples, `sigma_x`/
`sigma_y` in `labels.csv` are the **axis-aligned projected widths**, not
the pre-rotation principal-axis widths used to render the ellipse. See
[dataset.md](dataset.md) for why.
