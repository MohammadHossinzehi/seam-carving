"""Generate a synthetic demo image and run seam carving on it end to end.

Run with `python examples/generate_demo.py` from the repo root. No
external photo is required -- this builds its own test image (a sky
gradient with a bright circular "sun") so the whole pipeline is
runnable with nothing but this repo.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from seam_carving.carver import remove_object, resize_width
from seam_carving.energy import dual_gradient_energy
from seam_carving.io_utils import save_image

OUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def make_demo_image(h: int = 200, w: int = 300):
    y, x = np.mgrid[0:h, 0:w]

    top = np.array([30, 60, 120])
    bottom = np.array([180, 210, 240])
    t = (y / (h - 1))[:, :, None]
    img = (top * (1 - t) + bottom * t).astype(np.uint8)

    # A bright "sun": a filled circle near the top-right.
    cy, cx, r = h * 0.25, w * 0.75, min(h, w) * 0.12
    sun = (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2
    img[sun] = [255, 235, 140]

    # A dark "hill" silhouette across the bottom for a strong
    # horizontal edge too.
    hill = y > (h * 0.7 + 20 * np.sin(x / 20))
    img[hill] = [40, 90, 50]

    return img.astype(np.uint8), sun


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    img, sun_mask = make_demo_image()

    save_image(img, os.path.join(OUT_DIR, "original.png"))

    energy = dual_gradient_energy(img)
    peak = energy.max()
    normalized = (255 * energy / peak).astype(np.uint8)
    save_image(np.stack([normalized] * 3, axis=-1), os.path.join(OUT_DIR, "energy.png"))

    narrower = resize_width(img, int(img.shape[1] * 0.7))
    save_image(narrower, os.path.join(OUT_DIR, "resized_narrower.png"))

    no_sun = remove_object(img, sun_mask, restore_size=True)
    save_image(no_sun, os.path.join(OUT_DIR, "sun_removed.png"))

    print(f"Wrote demo images to {OUT_DIR}/")
    print("  original.png          - the synthetic input")
    print("  energy.png            - its dual-gradient energy map")
    print("  resized_narrower.png  - width reduced 30% via seam removal")
    print("  sun_removed.png       - the sun content-aware removed, size restored")


if __name__ == "__main__":
    main()
