"""Synthetic 2D Gaussian beam image generation."""
from dataclasses import dataclass

import numpy as np

import config


@dataclass
class BeamSample:
    image: np.ndarray  # uint8, shape (H, W)
    x0: float
    y0: float
    sigma_x: float
    sigma_y: float
    theta: float  # degrees
    has_beam: int
    category: str


def gaussian_2d(width, height, x0, y0, sigma_x, sigma_y, theta_deg=0.0,
                 amplitude=1.0, offset=0.0):
    """Render a (possibly rotated) 2D Gaussian intensity profile."""
    y_idx, x_idx = np.mgrid[0:height, 0:width].astype(np.float32)
    theta = np.deg2rad(theta_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    a = (cos_t ** 2) / (2 * sigma_x ** 2) + (sin_t ** 2) / (2 * sigma_y ** 2)
    b = (-np.sin(2 * theta)) / (4 * sigma_x ** 2) + (np.sin(2 * theta)) / (4 * sigma_y ** 2)
    c = (sin_t ** 2) / (2 * sigma_x ** 2) + (cos_t ** 2) / (2 * sigma_y ** 2)

    dx = x_idx - x0
    dy = y_idx - y0
    exponent = a * dx ** 2 + 2 * b * dx * dy + c * dy ** 2
    return (offset + amplitude * np.exp(-exponent)).astype(np.float32)


def add_gaussian_noise(image, sigma=0.02, rng=None):
    rng = rng or np.random.default_rng()
    noise = rng.normal(0.0, sigma, image.shape).astype(np.float32)
    return image + noise


def add_hot_pixels(image, amount=0.001, rng=None):
    rng = rng or np.random.default_rng()
    mask = rng.random(image.shape) < amount
    image = image.copy()
    image[mask] = 1.0
    return image


def add_background_gradient(image, strength=0.1, rng=None):
    rng = rng or np.random.default_rng()
    h, w = image.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    a, b = rng.uniform(-1, 1, size=2)
    gradient = a * (xx / w) + b * (yy / h)
    gradient = (gradient - gradient.min()) / (gradient.max() - gradient.min() + 1e-8)
    return image + strength * gradient


def to_uint8(image):
    image = np.clip(image, 0.0, 1.0)
    return (image * 255).astype(np.uint8)


def _random_center(rng, width, height, margin=config.CENTER_MARGIN):
    x0 = rng.uniform(margin, width - margin)
    y0 = rng.uniform(margin, height - margin)
    return x0, y0


def make_normal_sample(rng, width=config.IMG_WIDTH, height=config.IMG_HEIGHT):
    x0, y0 = _random_center(rng, width, height)
    sigma = rng.uniform(config.MIN_SIGMA, config.MAX_SIGMA * 0.6)
    amplitude = rng.uniform(0.7, 1.0)
    offset = rng.uniform(0.0, 0.05)

    image = gaussian_2d(width, height, x0, y0, sigma, sigma, 0.0, amplitude, offset)
    image = add_gaussian_noise(image, sigma=rng.uniform(0.005, 0.02), rng=rng)

    return BeamSample(to_uint8(image), x0, y0, sigma, sigma, 0.0, 1, "normal")


def _axis_aligned_projection(sigma_major, sigma_minor, theta_deg):
    """Project a rotated ellipse's principal widths onto the image x/y axes.

    The CNN has no rotation output, so it can only ever learn the width it
    actually sees along x and y -- not the pre-rotation principal sigmas,
    which are unrecoverable without theta. This matches what a real
    rotation-unaware second-moment beam-width measurement would report.
    """
    theta = np.deg2rad(theta_deg)
    sigma_x = np.sqrt((sigma_major * np.cos(theta)) ** 2 + (sigma_minor * np.sin(theta)) ** 2)
    sigma_y = np.sqrt((sigma_major * np.sin(theta)) ** 2 + (sigma_minor * np.cos(theta)) ** 2)
    return float(sigma_x), float(sigma_y)


def make_tilted_sample(rng, width=config.IMG_WIDTH, height=config.IMG_HEIGHT):
    x0, y0 = _random_center(rng, width, height)
    sigma_major = rng.uniform(config.MIN_SIGMA, config.MAX_SIGMA)
    sigma_minor = rng.uniform(config.MIN_SIGMA, config.MAX_SIGMA)
    while abs(sigma_major - sigma_minor) < 10:
        sigma_minor = rng.uniform(config.MIN_SIGMA, config.MAX_SIGMA)
    theta = rng.uniform(10, 170)
    amplitude = rng.uniform(0.7, 1.0)
    offset = rng.uniform(0.0, 0.05)

    image = gaussian_2d(width, height, x0, y0, sigma_major, sigma_minor, theta, amplitude, offset)
    image = add_gaussian_noise(image, sigma=rng.uniform(0.005, 0.02), rng=rng)

    sigma_x, sigma_y = _axis_aligned_projection(sigma_major, sigma_minor, theta)
    return BeamSample(to_uint8(image), x0, y0, sigma_x, sigma_y, theta, 1, "tilted")


def make_noisy_sample(rng, width=config.IMG_WIDTH, height=config.IMG_HEIGHT):
    x0, y0 = _random_center(rng, width, height)
    sigma_x = rng.uniform(config.MIN_SIGMA, config.MAX_SIGMA * 0.8)
    sigma_y = sigma_x if rng.random() < 0.5 else rng.uniform(config.MIN_SIGMA, config.MAX_SIGMA * 0.8)
    theta = rng.uniform(-15, 15)
    amplitude = rng.uniform(0.6, 1.0)
    offset = rng.uniform(0.0, 0.05)

    image = gaussian_2d(width, height, x0, y0, sigma_x, sigma_y, theta, amplitude, offset)
    image = add_background_gradient(image, strength=rng.uniform(0.05, 0.2), rng=rng)
    image = add_gaussian_noise(image, sigma=rng.uniform(0.04, 0.09), rng=rng)
    image = add_hot_pixels(image, amount=rng.uniform(0.0005, 0.002), rng=rng)

    return BeamSample(to_uint8(image), x0, y0, sigma_x, sigma_y, theta, 1, "noisy")


def make_background_sample(rng, width=config.IMG_WIDTH, height=config.IMG_HEIGHT):
    # Deliberately no hot pixels here: a saturated pixel would look like a
    # spurious beam-like feature, contradicting the has_beam=0 label these
    # ambient/empty frames are meant to teach the model.
    image = np.zeros((height, width), dtype=np.float32)
    image = add_background_gradient(image, strength=rng.uniform(0.05, 0.25), rng=rng)
    image = add_gaussian_noise(image, sigma=rng.uniform(0.01, 0.04), rng=rng)

    return BeamSample(
        to_uint8(image),
        width / 2.0, height / 2.0,
        config.MIN_SIGMA, config.MIN_SIGMA,
        0.0, 0, "background",
    )
