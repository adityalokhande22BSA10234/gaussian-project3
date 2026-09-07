# CNN models

## Input/output contract

All three models share the same interface, defined in `models/cnn.py`:

- **Input**: a single-channel (grayscale) image tensor, resized to
  `config.TRAIN_RESOLUTION` (128x128 by default).
- **Output**: two tensors --
  - a **beam-presence logit** (pass through `sigmoid` for a 0-1
    probability that a beam is present in the frame)
  - **4 sigmoid-bounded regression values** `(x, y, sigma_x, sigma_y)`,
    normalized: `x0/IMG_WIDTH`, `y0/IMG_HEIGHT`,
    `sigma_x/MAX_SIGMA_NORM`, `sigma_y/MAX_SIGMA_NORM`.

The presence logit exists because the dataset includes beam-free
`background` frames (see [dataset.md](dataset.md)); the 4 regression
outputs match the spec exactly.

## The three architectures

Selected via `--model` (single) or `--models` (multiple, for `all`).

| Model | Backbone | Params | Notes |
|---|---|---|---|
| `tiny` | 3-block custom ConvNet | ~155K | Fastest, lowest capacity -- good for quick iteration. |
| `medium` | 4-block custom ConvNet (default) | ~623K | Balanced accuracy/speed. |
| `resnet18` | torchvision ResNet-18, first conv swapped for 1-channel input | ~11.2M | Highest capacity, slowest, best accuracy in testing (see below). Trained from scratch -- ImageNet weights don't transfer to single-channel synthetic images. |

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
1-channel `Conv2d` (kernel 7, stride 2) and the final `fc` layer replaced
by `Linear(512, 5)`.

## Loss function

`training/train.py` combines two losses per batch:

- **`SmoothL1Loss`** (reduction='none') on the 4 regression outputs,
  **masked** to only beam-present samples (`has_beam=1`) and averaged
  over just those elements -- background frames contribute nothing to the
  regression loss, since they have no meaningful (x,y,sigma) target.
- **`BCEWithLogitsLoss`** on the presence logit, over *all* samples.

```python
loss = bce(presence_logit, presence_target) + masked_smooth_l1(params_pred, params_target, presence_mask)
```

This matches the spec's suggested Smooth L1 loss against normalized
ground truth, plus the presence term needed for the background category.

## Training

`python3 main.py train --model <name> --epochs 30` (or `all` for every
model):

- Optimizer: Adam, default learning rate `1e-3`
- Batch size: 16 (default)
- Device: auto-detects CUDA, then Apple Silicon MPS, then falls back to CPU
- Checkpointing: saves `checkpoints/<model>_best.pt` whenever validation
  loss improves; the checkpoint stores `model_state` + `model_name` so
  `evaluate`/`infer` can rebuild the right architecture automatically.

## Evaluation

`python3 main.py evaluate --model-path checkpoints/<model>_best.pt --split test`
reports:

- **Beam-presence accuracy**: fraction of samples where the presence
  prediction matches `has_beam`.
- **MAE (pixels)**: mean absolute error for `x, y, sigma_x, sigma_y`,
  computed only over beam-present samples, denormalized back to real
  pixel units.

Representative results from a 30-epoch run on the 2,000-image dataset
(1,600 train / 200 val / 200 test): all three models reached 1.000
presence accuracy; `resnet18` had the lowest position/sigma MAE, `tiny`
and `medium` were close behind at a fraction of the training cost.
