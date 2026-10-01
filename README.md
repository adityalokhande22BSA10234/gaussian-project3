# Gaussian Beam CNN Regression Pipeline

Trains a CNN to regress laser beam parameters -- center coordinates
`(x, y)` and spread widths `(sigma_x, sigma_y)` -- from a grayscale beam
camera capture, then runs that model on new images.

The training data is the real BPM capture set in `new_dataset/`. There is no
manual annotation step: labels are derived from each frame itself by
moment analysis of the dominant bright blob (see
[docs/dataset.md](docs/dataset.md)).

See [`docs/`](docs/) for architecture, dataset, model, and inference
details. This file only covers how to install and run the project.

## Requirements

- Python 3.9+
- See `requirements.txt`: numpy, pandas, pillow, opencv-python, torch,
  torchvision, matplotlib, streamlit

## Setup (after cloning)

```bash
git clone https://github.com/adityalokhande22BSA10234/gaussian_project_2.git
cd gaussian_project_2

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

The capture set is committed under `new_dataset/` (630 PNGs, 111MB), so
there is nothing to download.

## Dataset layout

```
new_dataset/
  train1/gs11 .. gs19     420 PNGs   -> carved into train/val
  test2/gs21 .. gs26      210 PNGs   -> held out whole as the test split
```

Every frame is 640x480 grayscale. `test2`'s gs classes never appear in
`train1`, so the test split measures generalization to *unseen* beam sizes --
test numbers are therefore expected to be worse than validation numbers.

## Running the project

### Quick start -- one command

```bash
python3 main.py
```

With no arguments, `main.py` runs the full pipeline:

1. Extracts `new_dataset/labels.csv` if it is missing or stale (one row per
   PNG, moments of the dominant blob give `x0`, `y0`, `sigma_x`, `sigma_y`,
   `theta`). The captures themselves are never modified.
2. Trains all three CNN models (`tiny`, `medium`, `resnet18`), 30 epochs
   each by default, checkpointing the best validation-loss weights to
   `checkpoints/<model>_best.pt`.
3. Evaluates each trained model on the held-out `test2` split and prints a
   summary table.

Re-running `python3 main.py` skips label extraction when the index already
matches the images on disk.

### Web dashboard

```bash
streamlit run app.py
```

Home / Dataset / Training / Evaluation / Inference pages wrap every CLI
stage. Label extraction and training run on a background thread and stream
their console output into the page. Inference can either upload a capture or
pick a frame straight from the dataset, and shows the predicted 1-sigma
ellipse overlaid on the frame.

### Full command reference

Every stage is also available as its own subcommand:

```bash
# Extract labels from the captured frames
python3 main.py labels --val-split 0.10 --seed 42

# Validate image integrity + label sanity
python3 main.py validate

# Save a preview grid image for a visual sanity check
python3 main.py preview

# Train a single model
python3 main.py train --model medium --epochs 30

# Evaluate a trained checkpoint on a split, with a per-gs-class breakdown
python3 main.py evaluate --model-path checkpoints/medium_best.pt --split test --by-category

# Run inference on a real beam image
python3 main.py infer --image path/to/capture.png --model-path checkpoints/medium_best.pt

# Run the full extract-if-missing + train-all-models pipeline explicitly
python3 main.py all --models tiny medium resnet18 --epochs 30
```

Run `python3 main.py <subcommand> --help` for every flag on a given
subcommand (dataset paths, batch size, learning rate, etc. are all
overridable).

The legacy synthetic generator is still available as `python3 main.py
generate`; it writes to `dataset/` and is never mixed into the real capture
set.

## Documentation

- [docs/architecture.md](docs/architecture.md) -- pipeline overview, project structure, module map, deviations from the original spec
- [docs/dataset.md](docs/dataset.md) -- capture set layout, label extraction, label schema, splits, quality validation
- [docs/models.md](docs/models.md) -- the three CNN architectures, I/O contract, loss functions, training/evaluation
- [docs/inference.md](docs/inference.md) -- full-frame preprocessing, denormalization, running inference
