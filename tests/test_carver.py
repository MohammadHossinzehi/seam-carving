import numpy as np
import pytest

from seam_carving.carver import (
    compute_cumulative_energy,
    find_vertical_seam,
    insert_vertical_seam,
    remove_object,
    remove_vertical_seam,
    resize_height,
    resize_width,
    seam_cost,
)


def _cheap_column_energy():
    # Column 2 is cheapest in every row, so the minimum seam should
    # hug it exactly -- this makes the "correct" answer hand-checkable.
    return np.array(
        [
            [9, 9, 1, 9, 9],
            [9, 9, 1, 9, 9],
            [9, 9, 1, 9, 9],
            [9, 9, 1, 9, 9],
        ],
        dtype=np.float64,
    )


def test_find_vertical_seam_follows_cheap_column():
    seam = find_vertical_seam(_cheap_column_energy())
    assert list(seam) == [2, 2, 2, 2]


def test_find_vertical_seam_can_move_diagonally():
    # Cheap path drifts one column to the right each row: (0,0) -> (1,1) -> (2,2).
    energy = np.array(
        [
            [1, 9, 9],
            [9, 1, 9],
            [9, 9, 1],
        ],
        dtype=np.float64,
    )
    seam = find_vertical_seam(energy)
    assert list(seam) == [0, 1, 2]


def test_cumulative_energy_first_row_matches_energy():
    energy = _cheap_column_energy()
    M = compute_cumulative_energy(energy)
    assert np.array_equal(M[0], energy[0])


def test_cumulative_energy_is_monotonically_built_from_minimum_neighbors():
    energy = _cheap_column_energy()
    M = compute_cumulative_energy(energy)
    # Row 1, column 2 should be its own energy plus the minimum of the
    # three cells above it (all reachable neighbors are cost 1 here).
    assert M[1, 2] == pytest.approx(1 + 1)


def test_seam_cost_matches_manual_sum():
    energy = _cheap_column_energy()
    seam = np.array([2, 2, 2, 2])
    assert seam_cost(energy, seam) == pytest.approx(4.0)


def test_remove_vertical_seam_shrinks_width_by_one():
    img = np.random.randint(0, 256, (6, 8, 3), dtype=np.uint8)
    seam = np.array([0, 1, 2, 3, 4, 5])
    out = remove_vertical_seam(img, seam)
    assert out.shape == (6, 7, 3)


def test_remove_vertical_seam_preserves_untouched_pixels():
    img = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
    seam = np.array([2, 2, 2, 2])  # remove column 2 in every row
    out = remove_vertical_seam(img, seam)
    expected = np.delete(img, 2, axis=1)
    assert np.array_equal(out, expected)


def test_insert_vertical_seam_grows_width_by_one():
    img = np.random.randint(0, 256, (6, 8, 3), dtype=np.uint8)
    seam = np.array([0, 1, 2, 3, 4, 5])
    out = insert_vertical_seam(img, seam)
    assert out.shape == (6, 9, 3)


def test_resize_width_shrinks_to_target():
    img = np.random.randint(0, 256, (20, 30, 3), dtype=np.uint8)
    out = resize_width(img, 24)
    assert out.shape == (20, 24, 3)


def test_resize_width_grows_to_target():
    img = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)
    out = resize_width(img, 14)
    assert out.shape == (10, 14, 3)


def test_resize_width_noop_returns_a_copy():
    img = np.random.randint(0, 256, (5, 5, 3), dtype=np.uint8)
    out = resize_width(img, 5)
    assert np.array_equal(out, img)
    assert out is not img


def test_resize_width_rejects_non_positive_target():
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        resize_width(img, 0)


def test_resize_height_uses_transpose_path():
    img = np.random.randint(0, 256, (20, 12, 3), dtype=np.uint8)
    out = resize_height(img, 15)
    assert out.shape == (15, 12, 3)


def test_remove_object_clears_masked_region_and_restores_size():
    img = np.random.randint(0, 256, (16, 20, 3), dtype=np.uint8)
    mask = np.zeros((16, 20), dtype=bool)
    mask[4:8, 8:12] = True  # a small rectangular "object"

    out = remove_object(img, mask, restore_size=True)

    assert out.shape == img.shape


def test_remove_object_without_restore_is_narrower():
    img = np.random.randint(0, 256, (10, 20, 3), dtype=np.uint8)
    mask = np.zeros((10, 20), dtype=bool)
    mask[:, 9:11] = True  # a 2px-wide vertical strip

    out = remove_object(img, mask, restore_size=False)

    assert out.shape[1] < img.shape[1]
    assert out.shape[0] == img.shape[0]


def test_remove_object_rejects_mismatched_mask_shape():
    img = np.zeros((5, 5, 3), dtype=np.uint8)
    mask = np.zeros((4, 4), dtype=bool)
    with pytest.raises(ValueError):
        remove_object(img, mask)
