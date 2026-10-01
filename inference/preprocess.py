"""Inference-time preprocessing for the beam-parameter CNN.

The model is trained on whole frames resized to `config.TRAIN_RESOLUTION`
(see data/dataset.py), so inference must feed it the *whole frame* the same
way. Cropping to a bright blob before resizing would shrink the beam and
stretch it to fill the input, which is a different distribution from anything
the network saw during training and makes its predictions meaningless.

`find_blob_bbox` is kept purely for drawing an overlay on the result; it never
feeds the network.
"""
import cv2
import numpy as np
from PIL import Image


def load_grayscale(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def subtract_background(image, background=None):
    """Remove a *measured* static baseline (a dark/ambient frame captured
    with the beam blocked) if one is supplied.

    Without a real background frame there's nothing reliable to subtract:
    approximating one by blurring the image itself is only safe if the blur
    kernel is smaller than the beam, which doesn't hold for wide beams (this
    project's sigma range goes up to ~120px) -- it would subtract away most
    of the beam along with the ambient light. So we just apply a light
    median filter to kill isolated hot pixels, which leaves smooth Gaussian
    blobs of any size untouched.
    """
    if background is not None:
        return cv2.subtract(image, background)
    return cv2.medianBlur(image, 3)


def find_blob_bbox(image, thresh_ratio=0.15, padding=30):
    """Bounding box of the brightest connected blob, for display only.

    Returns (x0, y0, x1, y1), or the full frame when nothing stands out.
    """
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    thresh_val = int(blurred.max() * thresh_ratio)
    if thresh_val <= 0:
        return (0, 0, image.shape[1], image.shape[0])

    _, mask = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return (0, 0, image.shape[1], image.shape[0])

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    return (
        max(x - padding, 0),
        max(y - padding, 0),
        min(x + w + padding, image.shape[1]),
        min(y + h + padding, image.shape[0]),
    )


def to_model_input(frame, resolution):
    """Resize a grayscale frame exactly the way data/dataset.py does.

    Full frame, bilinear, scaled to [0, 1], shaped (1, 1, R, R) for the CNN.
    """
    resized = Image.fromarray(frame).resize(
        (resolution, resolution), Image.BILINEAR
    )
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    return arr.reshape(1, 1, resolution, resolution)


def preprocess_for_inference(path, background=None):
    """Load -> optional background subtraction -> model-ready full-frame array.

    Returns (raw_image, background_subtracted_image, model_input, blob_bbox).
    The model input is always the *full* frame, matching training.
    """
    import config

    img = load_grayscale(path)
    subtracted = subtract_background(img, background)
    model_input = to_model_input(subtracted, config.TRAIN_RESOLUTION)
    bbox = find_blob_bbox(subtracted)
    return img, subtracted, model_input, bbox