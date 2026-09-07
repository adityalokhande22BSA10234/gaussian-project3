# Real-image preprocessing & inference

`python3 main.py infer --image <path> --model-path checkpoints/<model>_best.pt`
runs a trained model on a real (or synthetic) beam-camera image.

## Pipeline (`inference/preprocess.py` + `inference/infer.py`)

1. **Acquisition** -- `load_grayscale(path)` reads any image file
   (`cv2.imread` with `IMREAD_GRAYSCALE`), converting color images to
   grayscale automatically.

2. **Background subtraction** -- `subtract_background(image, background=None)`:
   - If you pass a real *measured* background/dark frame (captured with
     the beam blocked), it's subtracted directly via `cv2.subtract`.
   - If you don't, the function does **not** try to estimate one by
     blurring the image itself. Early testing showed this fails: this
     project's beams can have sigma up to 90px, comparable to or larger
     than a typical blur kernel, so self-blur-subtraction erases most of
     the beam along with the ambient light it's meant to remove. Instead,
     a light median filter (kernel size 3) is applied, which still
     removes isolated hot-pixel noise without touching smooth Gaussian
     blobs of any size.

3. **ROI (region-of-interest) cropping** -- `threshold_roi(image, thresh_ratio=0.15, padding=30)`:
   - Blurs slightly, thresholds at `thresh_ratio * max_intensity`.
   - Finds the largest connected bright contour via `cv2.findContours`.
   - Crops to its bounding box plus 30px padding on each side.
   - Falls back to the full frame if no contour is found (e.g. an empty
     frame).

4. **Inference** -- `run_inference(image_path, model_path)`:
   - Resizes the ROI crop to the model's training resolution (128x128 by
     default) and runs it through the model.
   - Rescales the normalized outputs back to real pixel coordinates:
     - `x`, `y` are fractions of the input frame, so they scale directly
       by the ROI's actual width/height.
     - `sigma_x`, `sigma_y` were normalized against a full
       `IMG_WIDTH`/`IMG_HEIGHT`-sized canvas during training, so they're
       rescaled by `(roi_width / IMG_WIDTH)` and `(roi_height /
       IMG_HEIGHT)` respectively to account for the ROI crop being
       smaller (or larger) than that canvas.
   - Returns `{has_beam, x, y, sigma_x, sigma_y, roi_bbox}`, all in the
     original image's pixel coordinates.

## Tips for real camera captures

- **Grayscale, 8-bit**: convert color images beforehand if needed; 16-bit
  raw sensor data should be rescaled to 8-bit first.
- **Avoid saturation**: use neutral-density (ND) filters -- laser beams
  are usually far too bright for a bare sensor, and a clipped/blown-out
  peak breaks the Gaussian assumption.
- **Fixed exposure/gain**: disable auto-exposure; it distorts the beam's
  true intensity profile between frames.
- **Capture a real dark/background frame** if possible (camera output
  with the beam blocked) and pass it to `subtract_background()` -- this
  is more reliable than the no-background fallback for cluttered or
  non-uniformly-lit scenes.
- Resolution doesn't need to match 640x480 -- the ROI-cropping step
  handles arbitrary real image sizes and rescales correctly.

## Known limitation

A trained model's accuracy is bounded by what it saw in training: the
synthetic dataset covers sigma in `[15, 90]` px, centered away from the
frame edges by a margin, on a clean (or moderately noisy) background. A
real capture far outside that regime (e.g. a beam much larger than 90px,
or one clipped at the frame edge) is out-of-distribution and will likely
be poorly estimated. If real captures consistently fall outside the
synthetic ranges, adjust `config.MIN_SIGMA`/`MAX_SIGMA`/`CENTER_MARGIN`
and regenerate the dataset before retraining.
