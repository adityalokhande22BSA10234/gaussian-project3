"""Real-image preprocessing: background subtraction and ROI extraction."""
import cv2
import numpy as np


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
    project's sigma range goes up to ~90px) -- it would subtract away most
    of the beam along with the ambient light. So we just apply a light
    median filter to kill isolated hot pixels, which leaves smooth Gaussian
    blobs of any size untouched.
    """
    if background is not None:
        return cv2.subtract(image, background)
    return cv2.medianBlur(image, 3)


def threshold_roi(image, thresh_ratio=0.15, padding=30):
    """Find the brightest connected blob and return a padded crop around it."""
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    thresh_val = int(blurred.max() * thresh_ratio)
    if thresh_val <= 0:
        return None, (0, 0, image.shape[1], image.shape[0])

    _, mask = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, (0, 0, image.shape[1], image.shape[0])

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    x0 = max(x - padding, 0)
    y0 = max(y - padding, 0)
    x1 = min(x + w + padding, image.shape[1])
    y1 = min(y + h + padding, image.shape[0])
    return image[y0:y1, x0:x1], (x0, y0, x1, y1)


def preprocess_for_inference(path, background=None):
    """Full pipeline: load -> background-subtract -> ROI crop.

    Returns (raw_image, background_subtracted_image, roi_crop, roi_bbox).
    Falls back to the full subtracted frame if no beam-like blob is found.
    """
    img = load_grayscale(path)
    subtracted = subtract_background(img, background)
    roi, bbox = threshold_roi(subtracted)
    if roi is None or roi.size == 0:
        roi, bbox = subtracted, (0, 0, img.shape[1], img.shape[0])
    return img, subtracted, roi, bbox
