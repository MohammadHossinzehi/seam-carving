"""Image I/O helpers built on Pillow.

Kept in a thin separate module so the algorithm code (energy.py,
carver.py) has no dependency beyond numpy and can be unit tested with
plain arrays; only this module and the CLI touch actual image files.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from PIL import Image


def load_image(path: str) -> np.ndarray:
    """Load an image from disk as an (H, W, 3) uint8 RGB array."""
    with Image.open(path) as im:
        return np.array(im.convert("RGB"), dtype=np.uint8)


def save_image(image: np.ndarray, path: str) -> None:
    """Save an (H, W, 3) array to disk as an image."""
    Image.fromarray(image.astype(np.uint8), mode="RGB").save(path)


def load_mask(path: str, expected_shape: Tuple[int, int]) -> np.ndarray:
    """Load a mask image as an (H, W) boolean array.

    Any pixel that isn't close to pure black counts as part of the
    mask, so it's easy to produce one: paint the object to remove in
    any bright color over a black background in an image editor.
    """
    with Image.open(path) as im:
        arr = np.array(im.convert("L"), dtype=np.uint8)
    if arr.shape != expected_shape:
        raise ValueError(f"mask shape {arr.shape} does not match image shape {expected_shape}")
    return arr > 10
