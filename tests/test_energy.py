import numpy as np
import pytest

from seam_carving.energy import dual_gradient_energy


def test_energy_shape_matches_input():
    img = np.random.randint(0, 256, size=(10, 12, 3), dtype=np.uint8)
    e = dual_gradient_energy(img)
    assert e.shape == (10, 12)


def test_flat_image_has_zero_energy():
    img = np.full((5, 5, 3), 128, dtype=np.uint8)
    e = dual_gradient_energy(img)
    assert np.allclose(e, 0)


def test_vertical_edge_has_higher_energy_than_flat_region():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    img[:, 5:, :] = 255  # sharp vertical edge down the middle

    e = dual_gradient_energy(img)

    # Columns straddling the edge should read much hotter than
    # columns safely inside either flat half.
    assert e[:, 4].mean() > e[:, 1].mean()
    assert e[:, 5].mean() > e[:, 8].mean()


def test_rejects_non_rgb_input():
    with pytest.raises(ValueError):
        dual_gradient_energy(np.zeros((4, 4)))


def test_energy_is_never_negative():
    img = np.random.randint(0, 256, size=(6, 6, 3), dtype=np.uint8)
    e = dual_gradient_energy(img)
    assert (e >= 0).all()
