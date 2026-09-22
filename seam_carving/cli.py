"""Command-line interface for seam carving.

Examples:
    python -m seam_carving resize photo.jpg out.jpg --width 800
    python -m seam_carving resize photo.jpg out.jpg --width 800 --height 500
    python -m seam_carving remove-object photo.jpg mask.png out.jpg
    python -m seam_carving energy photo.jpg energy.png
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from .carver import remove_object, resize_height, resize_width
from .energy import dual_gradient_energy
from .io_utils import load_image, load_mask, save_image


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="seam_carving",
        description="Content-aware image resizing and object removal via seam carving.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    resize_p = sub.add_parser("resize", help="Resize an image by removing/inserting seams.")
    resize_p.add_argument("input", help="Path to the input image.")
    resize_p.add_argument("output", help="Path to write the resized image.")
    resize_p.add_argument("--width", type=int, default=None, help="Target width in pixels.")
    resize_p.add_argument("--height", type=int, default=None, help="Target height in pixels.")

    remove_p = sub.add_parser("remove-object", help="Remove a masked region from an image.")
    remove_p.add_argument("input", help="Path to the input image.")
    remove_p.add_argument("mask", help="Path to a mask image (non-black pixels = object to remove).")
    remove_p.add_argument("output", help="Path to write the result.")
    remove_p.add_argument(
        "--no-restore-size",
        action="store_true",
        help="Keep the image narrower instead of re-inserting seams to restore the original width.",
    )

    energy_p = sub.add_parser("energy", help="Save a grayscale visualization of an image's energy map.")
    energy_p.add_argument("input", help="Path to the input image.")
    energy_p.add_argument("output", help="Path to write the energy map visualization.")

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "resize":
        if args.width is None and args.height is None:
            parser.error("resize requires --width and/or --height")
        image = load_image(args.input)
        if args.width is not None:
            image = resize_width(image, args.width)
        if args.height is not None:
            image = resize_height(image, args.height)
        save_image(image, args.output)
        print(f"Wrote {args.output} ({image.shape[1]}x{image.shape[0]})")

    elif args.command == "remove-object":
        image = load_image(args.input)
        mask = load_mask(args.mask, image.shape[:2])
        if not mask.any():
            parser.error("mask is empty (no non-black pixels found)")
        result = remove_object(image, mask, restore_size=not args.no_restore_size)
        save_image(result, args.output)
        print(f"Wrote {args.output} ({result.shape[1]}x{result.shape[0]})")

    elif args.command == "energy":
        image = load_image(args.input)
        energy = dual_gradient_energy(image)
        peak = energy.max()
        normalized = (255 * energy / peak).astype(np.uint8) if peak > 0 else energy.astype(np.uint8)
        save_image(np.stack([normalized] * 3, axis=-1), args.output)
        print(f"Wrote {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
