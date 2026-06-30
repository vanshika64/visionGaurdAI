"""
Generic, reusable helpers shared across the project.
No screen-detection-specific logic lives here.
"""

import logging
import time
import functools
import sys

import cv2
import numpy as np

import config


def setup_logger(name: str = "screen_detector") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        formatter = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


logger = setup_logger()


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Avoid divide-by-zero in ratio-based features."""
    if denominator == 0 or np.isnan(denominator):
        return default
    result = numerator / denominator
    if np.isnan(result) or np.isinf(result):
        return default
    return float(result)


def load_image(path: str) -> np.ndarray:
    """
    Load an image from disk as a BGR uint8 array, honoring EXIF orientation.
    Raises ValueError if the file cannot be read or decoded.
    """
    # cv2.imread does not honor EXIF orientation by default; use the
    # IMREAD_COLOR flag combined with a manual EXIF check via PIL fallback
    # only when necessary would add a heavy dependency, so we rely on
    # OpenCV's own EXIF-aware flag introduced in modern builds.
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def preprocess_image(image: np.ndarray, size: int = None) -> np.ndarray:
    """
    Resize to a fixed square resolution (config.RESIZE_DIM by default).
    Uses INTER_AREA for downscaling and INTER_LINEAR for upscaling to
    preserve frequency content as faithfully as possible.
    """
    if size is None:
        size = config.RESIZE_DIM
    h, w = image.shape[:2]
    interp = cv2.INTER_AREA if (h > size or w > size) else cv2.INTER_LINEAR
    resized = cv2.resize(image, (size, size), interpolation=interp)
    return resized


def to_gray(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def timer(func):
    """Decorator to log execution time of a function -- used for profiling
    in Step 9."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        logger.debug(f"{func.__name__} took {elapsed_ms:.2f} ms")
        return result, elapsed_ms

    return wrapper


def list_image_files(directory: str):
    """Return sorted list of image file paths in a directory."""
    import os

    valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    files = [
        os.path.join(directory, f)
        for f in sorted(os.listdir(directory))
        if f.lower().endswith(valid_ext)
    ]
    return files
