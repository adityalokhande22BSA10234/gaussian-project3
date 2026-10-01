# Preprocessing & inference

```
python main.py infer --image <path> --model-path checkpoints/<model>_best.pt
```

runs a trained model on a beam-camera image and returns
`(x, y, sigma_x, sigma_y)` in original-image pixels.

## Pipeline (`inference/preprocess.py` + `inference/infer.py`)

1. **Acquisition** -- `load_grayscale(path)` reads any image file
   (`cv2.imread` with `IMREAD_GRAYSCALE`), converting colour to grayscale
   automatically.

2. **Background subtraction** -- `subtract_background(image, background=None)`:
   - Pass a *measured* background frame (captured with the beam blocked) and it
     is subtracted directly via `cv2.subtract`.
   - Without one, it applies a light median filter (kernel 3) instead of trying
     to estimate a background by blurring the image. Self-blur-subtraction is
     unsafe here: beams reach `sigma` ~120 px, comparable to or larger than a
     typical blur kernel, so it erases the beam along with the ambient light.
     A median filter still kills isolated hot pixels and leaves smooth
     Gaussian blobs of any size untouched.

3. **Full-frame resize** -- `to_model_input(frame, resolution)` resizes the
   **whole frame** to 128x128 bilinear, scales to `[0, 1]`, shapes
   `(1, 1, 128, 128)`.

4. **Inference** -- the outputs are denormalized straight back to pixel units:
   `x = x_norm * IMG_WIDTH`, `y = y_norm * IMG_HEIGHT`,
   `sigma = sigma_norm * MAX_SIGMA_NORM`. No crop geometry enters the
   calculation, because no crop happened.

## The ROI crop that used to be here

`preprocess_for_inference` previously cropped to the brightest blob before
resizing (`threshold_roi`, ROI + 30 px padding) and rescaled the outputs by the
crop dimensions. That was a **train/inference distribution mismatch**: training
fed whole frames resized to 128x128 (`data/dataset.py:__getitem__`), while
inference fed a tightly-cropped beam stretched to fill the same 128x128. The
beam's apparent size, its position within the canvas, and its surrounding
context all differed from anything the network had seen, so predictions on real
frames were not comparable to training or evaluation numbers.

`find_blob_bbox` survives **purely for drawing an overlay** on a result. It
never feeds the network, and its return value is not an input to the rescaling.

## Tips for real camera captures

- **Grayscale, 8-bit**: 16-bit raw sensor data must be rescaled to 8-bit first.
- **Avoid saturation**: use neutral-density filters. A clipped, blown-out peak
  breaks the Gaussian assumption the moments rely on.
- **Fixed exposure/gain**: disable auto-exposure; it changes the beam's true
  intensity profile between frames.
- **Capture a real dark frame** if possible and pass it to
  `subtract_background()` -- more reliable than the no-background fallback for
  cluttered or non-uniformly-lit scenes.
- **Resolution need not be 640x480.** The full frame is resized to a square
  128x128, so any input size works. Note the aspect ratio is *not* preserved
  by the square resize; that is inherent to the training pipeline, not to
  inference.
- **Border bars** in real instrument frames can dominate a prediction. Label
  extraction masks the outer 40x30 px for that reason; inference deliberately
  does not, because the model is trained on unmasked frames.

## Known limitations

- Accuracy is bounded by the derived labels it was trained on. The test classes
  are unseen, and aggregate test MAE is dominated by `gs23`, whose own labels
  are largely outliers -- see [dataset.md](dataset.md#known-limitations).
- `train_model` does not seed torch, so repeat runs differ slightly and
  checkpoints are not bit-reproducible. `--seed` on `main.py labels` only
  controls the train/val carve.
- The presence output is a constant on this data (every frame has a beam), so
  it cannot be used to reject beamless input.