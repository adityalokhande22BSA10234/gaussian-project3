# CNN models

## Input/output contract

All three models share one interface, defined in `models/cnn.py`:

- **Input**: a single-channel grayscale image tensor resized to
  `config.TRAIN_RESOLUTION` (128x128). Training and inference both use the
  **whole frame** resized this way -- see [inference.md](inference.md).
- **Output**: two tensors --
  - a **beam-presence logit** (sigmoid for a 0-1 probability)
  - **4 sigmoid-bounded regression values** `(x, y, sigma_x, sigma_y)`,
    normalized as `x0/IMG_WIDTH`, `y0/IMG_HEIGHT`,
    `sigma_x/MAX_SIGMA_NORM`, `sigma_y/MAX_SIGMA_NORM`.

`MAX_SIGMA_NORM` is 120, chosen above the 90 px `MAX_SIGMA` of the old synthetic
range so real captures with wider beams still land inside the sigmoid's range.

### The presence head is inert on real data

Every one of the 630 captures contains a beam, so `has_beam` is `1`
throughout. The presence logit therefore converges to a constant and presence
accuracy is a trivial `1.000` that says nothing about model quality. The head
is retained because it is part of the model interface and the training loop
still computes its loss, but **do not use presence accuracy to compare
checkpoints**. MAE is the only meaningful metric here.

## The three architectures

Selected via `--model`, or `--models` to train several at once.

| Model | Backbone | Params | Notes |
|---|---|---|---|
| `tiny` | 3-block custom ConvNet | 155,365 | Fastest; for quick iteration. |
| `medium` | 4-block custom ConvNet (default) | 623,461 | Balanced. |
| `resnet18` | torchvision ResNet-18, `conv1` swapped for 1-channel input | 11,172,805 | 18x `medium`'s parameters. Trained from scratch; ImageNet weights do not transfer usefully to single-channel beam frames. |

### `tiny` -- `TinyCNN`

```
Conv(1->16) -> BN -> ReLU -> MaxPool
Conv(16->32) -> BN -> ReLU -> MaxPool
Conv(32->64) -> BN -> ReLU
AdaptiveAvgPool(4x4)
Flatten -> Linear(1024->128) -> ReLU -> Dropout(0.2) -> Linear(128->5)
```

### `medium` -- `MediumCNN` (default)

```
Conv(1->16) -> BN -> ReLU -> MaxPool
Conv(16->32) -> BN -> ReLU -> MaxPool
Conv(32->64) -> BN -> ReLU -> MaxPool
Conv(64->128) -> BN -> ReLU
AdaptiveAvgPool(4x4)
Flatten -> Linear(2048->256) -> ReLU -> Dropout(0.3) -> Linear(256->5)
```

### `resnet18` -- `ResNet18Regressor`

`torchvision.models.resnet18(weights=None)` with `conv1` replaced by a
1-channel `Conv2d` (kernel 7, stride 2) and `fc` replaced by `Linear(512, 5)`.

## Loss function

`training/train.py` combines two losses per batch:

- **`SmoothL1Loss`** (reduction `'none'`) on the 4 regression outputs,
  **masked** to beam-present samples (`has_beam=1`) and averaged over just those
  elements -- beam-free frames have no meaningful `(x, y, sigma)` target.
- **`BCEWithLogitsLoss`** on the presence logit, over all samples.

```python
loss = bce(presence_logit, presence_target) + masked_smooth_l1(params_pred, params_target, presence_mask)
```

On this dataset the mask selects every sample, so the masking is currently
inert -- it is retained so beamless frames would be handled correctly if any
were added.

## Training

```
python main.py train --model <name> --epochs 30
python main.py train --models tiny medium resnet18 --epochs 30
```

- Optimizer: Adam, default learning rate `1e-3`
- Batch size: 16 (default)
- Device: auto-detects CUDA, then Apple Silicon MPS, then CPU
- Checkpointing: saves `checkpoints/<model>_best.pt` whenever validation loss
  improves, storing `model_state` + `model_name` so `evaluate`/`infer` can
  rebuild the right architecture automatically.
- **`train_model` does not seed torch**, so runs are not bit-reproducible.
  Metrics move by a few hundredths between runs, which matters when comparing
  architectures -- see below.

Checkpoints are gitignored; the three `*_best.pt` files here were trained on the
real captures by the author, not committed.

## Evaluation

```
python main.py evaluate --model-path checkpoints/<model>_best.pt --split test --by-category
```

reports:

- **MAE in pixels** for `x, y, sigma_x, sigma_y`, denormalized back to real
  pixel units.
- **MAE per gs class** with `--by-category`. This is the important one: a single
  aggregate test number hides which classes fail, and on this data it does.

### Results on the real captures

630 frames, 379 train / 41 val / 210 test, 30 epochs. Test classes are unseen
by construction.

| Model | val MAE | test MAE | test x | test sigma_x | test sigma_y |
|---|---|---|---|---|---|
| `tiny` | 7.48 | 41.38 | 79.5 | 21.4 | 36.0 |
| `medium` | 6.92 | 41.18 | 82.1 | 19.3 | 36.4 |
| `resnet18` | 7.54 | 41.24 | 79.6 | 20.6 | 35.7 |

(`medium` per-parameter: val x 20.07, y 5.04, sigma_x 1.61, sigma_y 0.97.)

Two things to read from this table:

1. **Capacity is not the bottleneck.** A 72x parameter range spans 0.4 px of
   aggregate test MAE -- within run-to-run noise given the missing seed. The
   limit is label quality, not model size.
2. **Validation MAE is roughly 5x optimistic.** Beam widths are recovered well
   (`sigma_x` 1.61 px on val), but `x` jumps from 20 to 82 px on unseen
   classes, and `sigma_y` from 1.0 to 36 px. Per-class evaluation shows the
   damage is concentrated in `gs23`, whose own derived labels are outliers, so
   this is largely a label problem rather than a generalization failure --
   but a single aggregate test number would have suggested the latter.