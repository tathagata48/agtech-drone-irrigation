"""
tests/conftest.py
==================
Shared pytest fixtures for the precision irrigation test suite.
"""

import pytest
import numpy as np
import cv2


# ── Image size ───────────────────────────────────────────────────────────────
IMG_H = 200
IMG_W = 200


@pytest.fixture(scope="session")
def synthetic_pair():
    """
    A minimal synthetic RGB + thermal pair generated in-memory.
    Brown soil background with one green circle (hydrated, ~26°C) and
    one orange-tinted circle (stressed, ~34°C) drawn on it.

    Returns (rgb_bgr, thermal_grey) both as numpy arrays.
    """
    from src.data.synthetic import generate_synthetic_arrays
    return generate_synthetic_arrays(image_size=500, seed=42)


@pytest.fixture(scope="session")
def rgb_bgr(synthetic_pair):
    return synthetic_pair[0]


@pytest.fixture(scope="session")
def thermal_grey(synthetic_pair):
    return synthetic_pair[1]


@pytest.fixture()
def blank_green_rgb():
    """A pure solid-green BGR image — all pixels should segment as canopy."""
    img = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
    img[:] = (60, 200, 60)  # BGR green
    return img


@pytest.fixture()
def blank_brown_rgb():
    """A pure soil-brown BGR image — no pixels should segment as canopy."""
    img = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
    img[:] = (60, 80, 130)  # BGR brown
    return img


@pytest.fixture()
def uniform_thermal(tmp_path):
    """A 500×500 uniform 8-bit thermal image at pixel 100 (~39°C)."""
    arr = np.full((500, 500), 100, dtype=np.uint8)
    path = tmp_path / "thermal_uniform.jpg"
    cv2.imwrite(str(path), arr)
    return str(path), arr
