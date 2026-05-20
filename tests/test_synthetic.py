"""tests/test_synthetic.py"""

import numpy as np
import pytest

from src.data.synthetic import (
    generate_synthetic_farm_data,
    generate_synthetic_arrays,
    _temp_to_pixel,
    TEMP_MIN_C,
    TEMP_MAX_C,
)


class TestTempToPixel:
    def test_min_temp_maps_to_zero(self):
        assert _temp_to_pixel(TEMP_MIN_C) == 0

    def test_max_temp_maps_to_255(self):
        assert _temp_to_pixel(TEMP_MAX_C) == 255

    def test_midpoint(self):
        mid_temp = (TEMP_MIN_C + TEMP_MAX_C) / 2
        expected = 127  # ≈ 255/2
        assert abs(_temp_to_pixel(mid_temp) - expected) <= 1

    def test_clamps_below_min(self):
        assert _temp_to_pixel(TEMP_MIN_C - 10) == 0

    def test_clamps_above_max(self):
        assert _temp_to_pixel(TEMP_MAX_C + 10) == 255


class TestGenerateSyntheticArrays:
    def test_rgb_shape(self):
        rgb, _ = generate_synthetic_arrays(image_size=500)
        assert rgb.shape == (500, 500, 3)

    def test_thermal_shape(self):
        _, thermal = generate_synthetic_arrays(image_size=500)
        assert thermal.shape == (500, 500)

    def test_rgb_dtype_uint8(self):
        rgb, _ = generate_synthetic_arrays()
        assert rgb.dtype == np.uint8

    def test_thermal_dtype_uint8(self):
        _, thermal = generate_synthetic_arrays()
        assert thermal.dtype == np.uint8

    def test_deterministic_with_same_seed(self):
        rgb1, th1 = generate_synthetic_arrays(seed=7)
        rgb2, th2 = generate_synthetic_arrays(seed=7)
        np.testing.assert_array_equal(rgb1, rgb2)
        np.testing.assert_array_equal(th1, th2)

    def test_different_seeds_differ(self):
        rgb1, _ = generate_synthetic_arrays(seed=1)
        rgb2, _ = generate_synthetic_arrays(seed=2)
        assert not np.array_equal(rgb1, rgb2)

    def test_row1_cooler_than_row2_thermal(self):
        """Row 1 (hydrated) pixels must be cooler than Row 2 (dehydrated)."""
        _, thermal = generate_synthetic_arrays(image_size=500)
        # Row 1 is at y ≈ 500/3 ≈ 166; Row 2 at y ≈ 333
        row1_strip = thermal[140:190, 60:440]
        row2_strip = thermal[315:365, 60:440]
        assert row1_strip.mean() < row2_strip.mean(), (
            f"Row 1 ({row1_strip.mean():.1f}) not cooler than "
            f"Row 2 ({row2_strip.mean():.1f})"
        )


class TestGenerateSyntheticFarmData:
    def test_files_created(self, tmp_path):
        rgb_path = str(tmp_path / "rgb.jpg")
        thr_path = str(tmp_path / "thermal.jpg")
        out_rgb, out_thr = generate_synthetic_farm_data(
            out_rgb=rgb_path, out_thermal=thr_path
        )
        import pathlib
        assert pathlib.Path(out_rgb).exists()
        assert pathlib.Path(out_thr).exists()

    def test_returned_paths_match_args(self, tmp_path):
        rgb_path = str(tmp_path / "r.jpg")
        thr_path = str(tmp_path / "t.jpg")
        a, b = generate_synthetic_farm_data(rgb_path, thr_path)
        assert a == rgb_path
        assert b == thr_path
