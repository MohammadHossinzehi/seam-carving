"""Seam carving: content-aware image resizing and object removal.

Implements the algorithm from Avidan & Shamir's "Seam Carving for
Content-Aware Image Resizing" (SIGGRAPH 2007): repeatedly find and
remove (or insert) the lowest-energy connected path of pixels running
top-to-bottom or left-to-right through an image, so that resizing
shrinks or grows the "boring" parts of a photo instead of uniformly
squashing everything or cropping content off the edges.
"""

from .energy import dual_gradient_energy
from .carver import (
    compute_cumulative_energy,
    find_vertical_seam,
    remove_vertical_seam,
    insert_vertical_seam,
    resize_width,
    resize_height,
    remove_object,
    seam_cost,
)

__all__ = [
    "dual_gradient_energy",
    "compute_cumulative_energy",
    "find_vertical_seam",
    "remove_vertical_seam",
    "insert_vertical_seam",
    "resize_width",
    "resize_height",
    "remove_object",
    "seam_cost",
]

__version__ = "1.0.0"
