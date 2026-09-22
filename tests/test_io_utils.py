import numpy as np
import pytest
from PIL import Image

from seam_carving.io_utils import load_image, load_mask, save_image


def test_save_and_load_round_trip(tmp_path):
    img = np.random.randint(0, 256, (8, 10, 3), dtype=np.uint8)
    path = tmp_path / "test.png"
    save_image(img, str(path))
    loaded = load_image(str(path))
    assert np.array_equal(loaded, img)


def test_load_mask_thresholds_non_black_pixels(tmp_path):
    mask_img = np.zeros((5, 5), dtype=np.uint8)
    mask_img[2, 2] = 255
    path = tmp_path / "mask.png"
    Image.fromarray(mask_img, mode="L").save(path)

    mask = load_mask(str(path), (5, 5))
    assert mask[2, 2]
    assert not mask[0, 0]


def test_load_mask_rejects_shape_mismatch(tmp_path):
    mask_img = np.zeros((5, 5), dtype=np.uint8)
    path = tmp_path / "mask.png"
    Image.fromarray(mask_img, mode="L").save(path)

    with pytest.raises(ValueError):
        load_mask(str(path), (6, 6))
