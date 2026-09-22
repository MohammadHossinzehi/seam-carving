# Seam Carving

Content-aware image resizing and object removal, implemented from scratch in Python (NumPy only for the algorithm; Pillow is used solely for reading/writing image files).

## What it does, and why it's useful

Ordinary resizing either scales everything uniformly (squashing faces and objects) or crops the edges (losing content). Seam carving, introduced by Avidan & Shamir in ["Seam Carving for Content-Aware Image Resizing"](https://dl.acm.org/doi/10.1145/1276377.1276390) (SIGGRAPH 2007), instead finds the connected, top-to-bottom path of least-important pixels in an image -- a "seam" -- and removes (or duplicates) it. Repeating this one column at a time shrinks or grows the image while leaving visually important content essentially untouched.

This repo implements:

- **Dual-gradient energy** -- a per-pixel "importance" score based on local color gradients.
- **Seam finding** -- dynamic programming over the energy map to find the minimum-cost 8-connected vertical (or, via transpose, horizontal) seam.
- **Resizing** -- shrink by repeatedly removing the cheapest seam; grow by finding several cheap seams at once (via an index map back to the original image, so the same seam isn't just cloned k times) and inserting them together.
- **Object removal** -- paint a mask over anything in the photo, and the tool forces seams through the masked pixels until they're gone entirely, then (optionally) re-inserts seams to "heal" the image back to its original size.

## How to run it

```bash
pip install -r requirements.txt

# No photo on hand? Generate a synthetic demo image and see all three
# operations (resize, energy visualization, object removal) run end to end:
python examples/generate_demo.py
# -> writes examples/output/{original,energy,resized_narrower,sun_removed}.png
```

Command-line interface, once you have your own image:

```bash
# Shrink width to 800px (height unchanged), content-aware:
python -m seam_carving resize photo.jpg out.jpg --width 800

# Resize both dimensions:
python -m seam_carving resize photo.jpg out.jpg --width 800 --height 500

# Visualize what the algorithm thinks is "important" (bright = high energy):
python -m seam_carving energy photo.jpg energy.png

# Remove an object: paint it white on a black PNG the same size as the
# photo, then:
python -m seam_carving remove-object photo.jpg mask.png out.jpg
```

Or use it as a library:

```python
from seam_carving import resize_width, remove_object
from seam_carving.io_utils import load_image, save_image

img = load_image("photo.jpg")
narrower = resize_width(img, 600)
save_image(narrower, "narrower.jpg")
```

## Design decisions

- **Dual-gradient energy** (sum of squared RGB derivatives in x and y, summed over channels, then square-rooted) was chosen over plain grayscale Sobel because it's what the original paper uses, is cheap to compute, and responds to color edges a grayscale conversion would wash out. Border pixels use edge-replicated padding (`np.pad(..., mode="edge")`) instead of wrapping around the image, which would invent a fake seam between the left and right edges.
- **Backward energy only.** The DP table and energy function only look at energy already in the image ("backward energy"), not at the energy a removal would *create* (Rubinstein et al.'s 2008 "forward energy" extension). This keeps the implementation close to the original, well-known algorithm; the tradeoff is documented below.
- **Enlarging via an index map, not repeated single-seam insertion.** A naive loop of "find cheapest seam, duplicate it, repeat" keeps re-finding the same seam, producing an obvious stretched stripe. Instead, the k cheapest seams are found on a shrinking *copy* of the image while an index map tracks each seam's column position in the *original* image; all k are inserted into the original in a single pass.
- **Object removal reuses the same machinery.** Masked pixels get an artificial energy of `-1e6`, which guarantees the minimum-energy seam always cuts through the mask. Removing seams until the mask is empty deletes the object; re-inserting seams (the same enlarge codepath) restores the original canvas size. No separate "inpainting" logic was needed.
- **Height resize via transpose.** Rather than writing a second, nearly-identical horizontal-seam implementation, `resize_height` transposes the image, calls the vertical-seam code, and transposes back.

## Testing

`pytest` unit tests (24, all passing) cover:

- The energy function on hand-checkable cases (flat image -> zero energy, a synthetic hard edge -> higher energy near the edge, invalid non-RGB input rejected).
- The DP seam finder against small, hand-constructed energy grids where the correct minimum path is known exactly (a single cheap column, and a diagonal cheap path), so the test asserts the *exact* seam rather than just "some seam was found."
- Seam removal/insertion shape and content correctness (removing a known column matches `np.delete`).
- End-to-end `resize_width`/`resize_height` shrink and grow paths, including the no-op case and input validation.
- Object removal: the output size is restored when requested, is strictly narrower when it isn't, and mismatched mask shapes raise clearly.
- Image I/O round-trips using `tmp_path`, so no binary fixture files are committed to the repo.

Run them with:

```bash
pip install -r requirements-dev.txt
pytest
```

## Limitations / possible extensions

- No forward-energy term, so on some images (especially those with fine repeated texture) a removed seam can occasionally leave a faint visible artifact -- visible as a thin seam-shaped line in `sun_removed.png` from the demo, right where a seam had to cut across the hill silhouette.
- Pure NumPy, single-threaded; the DP pass is O(H x W) per seam, so removing/inserting many seams on a large photo is the main cost. There's no GPU or multiprocessing path.
- No interactive mask painting tool is included -- masks are plain PNGs, paintable in any image editor.
