"""Energy computation for seam carving.

Implements the dual-gradient energy function from Avidan & Shamir's
"Seam Carving for Content-Aware Image Resizing" (SIGGRAPH 2007).

For a pixel p = (x, y), the energy is the magnitude of the color
gradient in the x and y directions, summed over the R, G, B channels:

    e(x, y) = sqrt( sum_c [ (dI_c/dx)^2 + (dI_c/dy)^2 ] )

where dI_c/dx and dI_c/dy are central differences of channel c between
the pixel's neighbors. Border pixels reuse their nearest interior
neighbor (edge padding) so the formula stays well defined at the edge
of the image without wrapping to the opposite side.
"""

from __future__ import annotations

import numpy as np


def dual_gradient_energy(image: np.ndarray) -> np.ndarray:
    """Compute the dual-gradient energy map of an RGB image.

    Args:
        image: (H, W, 3) array, any numeric dtype. Cast to float64
            internally so squared differences don't wrap/clip for
            uint8 input.

    Returns:
        (H, W) float64 array of per-pixel energy. Higher values mean
        the pixel sits on a stronger edge, i.e. it is more "important"
        to preserve and less likely to be chosen for removal.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected an (H, W, 3) RGB array, got shape {image.shape}")

    img = image.astype(np.float64)

    # Pad with edge replication so border pixels get a one-sided
    # difference instead of wrapping around the image.
    padded = np.pad(img, ((1, 1), (1, 1), (0, 0)), mode="edge")

    dx = padded[1:-1, 2:, :] - padded[1:-1, :-2, :]
    dy = padded[2:, 1:-1, :] - padded[:-2, 1:-1, :]

    energy = np.sum(dx ** 2 + dy ** 2, axis=2)
    return np.sqrt(energy)
