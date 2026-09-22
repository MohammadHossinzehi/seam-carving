"""Seam finding, seam removal/insertion, resizing, and object removal.

The core idea (Avidan & Shamir, 2007): define a per-pixel "energy"
(how important a pixel looks, e.g. how strong an edge it sits on),
then find the connected top-to-bottom path of pixels ("seam") with
the lowest total energy via dynamic programming. Removing that path
shrinks the image by one column while deleting mostly "boring"
pixels; inserting a duplicated version of it grows the image by one
column in the same content-aware way.
"""

from __future__ import annotations

from typing import Callable, List

import numpy as np

from .energy import dual_gradient_energy

EnergyFn = Callable[[np.ndarray], np.ndarray]


def compute_cumulative_energy(energy: np.ndarray) -> np.ndarray:
    """Dynamic-programming table of minimum cumulative energy to each pixel.

        M[0, j]   = energy[0, j]
        M[i, j]   = energy[i, j] + min(M[i-1, j-1], M[i-1, j], M[i-1, j+1])

    Out-of-bounds neighbors at the left/right border are treated as
    +inf so they are never selected by `min`.
    """
    h, w = energy.shape
    M = np.empty_like(energy)
    M[0] = energy[0]

    for i in range(1, h):
        left = np.concatenate(([np.inf], M[i - 1, :-1]))
        center = M[i - 1]
        right = np.concatenate((M[i - 1, 1:], [np.inf]))
        M[i] = energy[i] + np.minimum(np.minimum(left, center), right)

    return M


def find_vertical_seam(energy: np.ndarray) -> np.ndarray:
    """Find the lowest-total-energy top-to-bottom 8-connected seam.

    Returns:
        1D int64 array of length H: the column index of the seam at
        each row.
    """
    h, w = energy.shape
    M = compute_cumulative_energy(energy)

    seam = np.empty(h, dtype=np.int64)
    seam[-1] = int(np.argmin(M[-1]))

    for i in range(h - 2, -1, -1):
        j = int(seam[i + 1])
        lo = max(j - 1, 0)
        hi = min(j + 1, w - 1)
        window = M[i, lo:hi + 1]
        seam[i] = lo + int(np.argmin(window))

    return seam


def seam_cost(energy: np.ndarray, seam: np.ndarray) -> float:
    """Total energy along a given seam. Handy for tests/diagnostics."""
    return float(energy[np.arange(len(seam)), seam].sum())


def remove_vertical_seam(image: np.ndarray, seam: np.ndarray) -> np.ndarray:
    """Delete one pixel per row (given by `seam`) -> (H, W-1, C) image."""
    h, w, c = image.shape
    mask = np.ones((h, w), dtype=bool)
    mask[np.arange(h), seam] = False
    return image[mask].reshape(h, w - 1, c)


def insert_vertical_seam(image: np.ndarray, seam: np.ndarray) -> np.ndarray:
    """Duplicate one pixel per row (given by `seam`) -> (H, W+1, C) image.

    The inserted pixel is the average of the seam pixel and its right
    neighbor rather than an exact duplicate, which avoids a visible
    hard-edged double line down the enlarged image.
    """
    h, w, c = image.shape
    out = np.empty((h, w + 1, c), dtype=image.dtype)
    for i in range(h):
        j = int(seam[i])
        neighbor = image[i, min(j + 1, w - 1)]
        new_pixel = ((image[i, j].astype(np.int32) + neighbor.astype(np.int32)) // 2).astype(image.dtype)
        out[i, :j] = image[i, :j]
        out[i, j] = image[i, j]
        out[i, j + 1] = new_pixel
        out[i, j + 2:] = image[i, j + 1:]
    return out


def resize_width(image: np.ndarray, target_width: int, energy_fn: EnergyFn = dual_gradient_energy) -> np.ndarray:
    """Resize `image` to `target_width` columns via seam removal/insertion."""
    h, w, _ = image.shape
    if target_width <= 0:
        raise ValueError("target_width must be positive")
    if target_width == w:
        return image.copy()
    if target_width > w:
        return _enlarge_width(image, target_width, energy_fn)
    return _shrink_width(image, target_width, energy_fn)


def resize_height(image: np.ndarray, target_height: int, energy_fn: EnergyFn = dual_gradient_energy) -> np.ndarray:
    """Resize `image` to `target_height` rows.

    Implemented by transposing the image (rows <-> columns), reusing
    the width-resize machinery, and transposing back.
    """
    transposed = np.transpose(image, (1, 0, 2))
    resized = resize_width(transposed, target_height, energy_fn)
    return np.transpose(resized, (1, 0, 2))


def _shrink_width(image: np.ndarray, target_width: int, energy_fn: EnergyFn) -> np.ndarray:
    current = image
    n_remove = image.shape[1] - target_width
    for _ in range(n_remove):
        energy = energy_fn(current)
        seam = find_vertical_seam(energy)
        current = remove_vertical_seam(current, seam)
    return current


def _enlarge_width(image: np.ndarray, target_width: int, energy_fn: EnergyFn) -> np.ndarray:
    """Grow the image by inserting the `n_add` lowest-energy seams.

    A naive "find one seam, insert it, repeat" would keep re-selecting
    the same (now-duplicated) seam. Instead we find all `n_add` seams
    up front on a shrinking *copy* of the image, tracking where each
    one falls in the *original* image via an index map, then insert
    them all into the original image in one pass.
    """
    n_add = target_width - image.shape[1]
    h, w, _ = image.shape

    temp = image.copy()
    index_map = np.tile(np.arange(w), (h, 1))
    seams_per_row: List[List[int]] = [[] for _ in range(h)]

    for _ in range(n_add):
        energy = energy_fn(temp)
        seam = find_vertical_seam(energy)
        for i in range(h):
            seams_per_row[i].append(int(index_map[i, seam[i]]))
        temp = remove_vertical_seam(temp, seam)
        index_map = remove_vertical_seam(index_map[:, :, None], seam)[:, :, 0]

    rows = []
    for i in range(h):
        cols = sorted(seams_per_row[i])
        row = image[i]
        new_row = []
        col_iter = iter(cols)
        next_dup = next(col_iter, None)
        for j in range(w):
            new_row.append(row[j])
            while next_dup == j:
                neighbor = row[min(j + 1, w - 1)]
                avg = ((row[j].astype(np.int32) + neighbor.astype(np.int32)) // 2).astype(row.dtype)
                new_row.append(avg)
                next_dup = next(col_iter, None)
        rows.append(np.stack(new_row, axis=0))

    return np.stack(rows, axis=0)


def remove_object(
    image: np.ndarray,
    mask: np.ndarray,
    energy_fn: EnergyFn = dual_gradient_energy,
    restore_size: bool = True,
) -> np.ndarray:
    """Remove the region marked True in `mask` from the image.

    Works by giving masked pixels an artificially huge negative
    energy so the minimum-energy seam is guaranteed to cut through the
    object, then removing that seam. Repeated until no masked pixels
    remain. This is the classic "seam carving object removal" trick.

    Args:
        image: (H, W, 3) image.
        mask: (H, W) boolean array, True where the object to delete is.
        energy_fn: energy function used for the *unmasked* pixels.
        restore_size: if True, re-widen the result back to the
            original width via seam insertion once the object is
            gone, "healing" the hole with content-aware seams instead
            of leaving the image permanently narrower.

    Returns:
        The processed image.
    """
    if mask.shape != image.shape[:2]:
        raise ValueError(f"mask shape {mask.shape} must match image shape {image.shape[:2]}")

    current = image.copy()
    current_mask = mask.copy()
    original_width = image.shape[1]

    penalty = -1e6  # far below any real energy value: always on the minimum seam

    while current_mask.any():
        energy = energy_fn(current)
        energy = np.where(current_mask, penalty, energy)
        seam = find_vertical_seam(energy)
        current = remove_vertical_seam(current, seam)
        current_mask = remove_vertical_seam(
            current_mask[:, :, None].astype(np.uint8), seam
        )[:, :, 0].astype(bool)

    if restore_size and current.shape[1] < original_width:
        current = resize_width(current, original_width, energy_fn)

    return current
