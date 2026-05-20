"""tests/test_thermal.py"""

import numpy as np
import pytest

from src.pipeline.thermal import (
    translate_to_celsius,
    apply_canopy_mask_to_thermal,
    analyze_contours,
    build_canopy_heatmap,
    set_temp_range,
    _TEMP_MIN_C,
    _TEMP_MAX_C,
)


class TestTranslateToCelsius:
    def test_pixel_0_maps_to_temp_min(self):
        arr = np.zeros((10, 10), dtype=np.uint8)
        result = translate_to_celsius(arr)
        np.testing.assert_allclose(result, _TEMP_MIN_C, atol=0.1)

    def test_pixel_255_maps_to_temp_max(self):
        arr = np.full((10, 10), 255, dtype=np.uint8)
        result = translate_to_celsius(arr)
        np.testing.assert_allclose(result, _TEMP_MAX_C, atol=0.1)

    def test_pixel_128_maps_to_midpoint(self):
        arr = np.full((10, 10), 128, dtype=np.uint8)
        result = translate_to_celsius(arr)
        expected = _TEMP_MIN_C + (_TEMP_MAX_C - _TEMP_MIN_C) * 128 / 255
        np.testing.assert_allclose(result, expected, atol=0.2)

    def test_output_dtype_is_float32(self):
        arr = np.zeros((5, 5), dtype=np.uint8)
        result = translate_to_celsius(arr)
        assert result.dtype == np.float32

    def test_output_shape_preserved(self):
        arr = np.zeros((100, 80), dtype=np.uint8)
        result = translate_to_celsius(arr)
        assert result.shape == (100, 80)


class TestApplyCanopyMask:
    def test_soil_pixels_zeroed(self):
        temp = np.full((10, 10), 30.0, dtype=np.float32)
        mask = np.zeros((10, 10), dtype=np.uint8)   # all soil
        result = apply_canopy_mask_to_thermal(temp, mask)
        np.testing.assert_array_equal(result, 0.0)

    def test_leaf_pixels_preserved(self):
        temp = np.full((10, 10), 28.5, dtype=np.float32)
        mask = np.full((10, 10), 255, dtype=np.uint8)  # all canopy
        result = apply_canopy_mask_to_thermal(temp, mask)
        np.testing.assert_allclose(result, 28.5)

    def test_partial_mask(self):
        temp = np.full((10, 10), 32.0, dtype=np.float32)
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[:5, :] = 255  # top half is leaf
        result = apply_canopy_mask_to_thermal(temp, mask)
        assert (result[:5, :] == 32.0).all()
        assert (result[5:, :] == 0.0).all()


class TestAnalyzeContours:
    def _make_simple_scene(self, plant_temp: float, soil_temp: float = 40.0):
        """One green circle on a brown background with matching thermal."""
        H, W = 200, 200
        # RGB: brown bg + green circle
        rgb = np.full((H, W, 3), [60, 80, 130], dtype=np.uint8)
        import cv2
        cv2.circle(rgb, (100, 100), 30, (50, 180, 70), thickness=-1)

        # Canopy mask
        import cv2 as cv
        from src.pipeline.segmentation import segment_canopy
        mask = segment_canopy(rgb)

        # Temperature field
        temp = np.full((H, W), soil_temp, dtype=np.float32)
        temp[mask > 0] = plant_temp

        return rgb, temp, mask

    def test_stressed_plant_flagged(self):
        rgb, temp, mask = self._make_simple_scene(plant_temp=35.0)
        _, n_stress, n_ok = analyze_contours(rgb, temp, mask, temp_threshold=30.0)
        assert n_stress >= 1

    def test_healthy_plant_flagged(self):
        rgb, temp, mask = self._make_simple_scene(plant_temp=25.0)
        _, n_stress, n_ok = analyze_contours(rgb, temp, mask, temp_threshold=30.0)
        assert n_ok >= 1

    def test_empty_mask_returns_zero_counts(self):
        rgb = np.zeros((100, 100, 3), dtype=np.uint8)
        temp = np.full((100, 100), 30.0, dtype=np.float32)
        mask = np.zeros((100, 100), dtype=np.uint8)
        _, n_stress, n_ok = analyze_contours(rgb, temp, mask, 30.0)
        assert n_stress == 0
        assert n_ok == 0

    def test_overlay_is_copy_not_in_place(self):
        rgb, temp, mask = self._make_simple_scene(plant_temp=25.0)
        original = rgb.copy()
        overlay, _, _ = analyze_contours(rgb, temp, mask, 30.0)
        np.testing.assert_array_equal(rgb, original)


class TestBuildCanopyHeatmap:
    def test_soil_is_black(self):
        canopy_temp = np.full((50, 50), 0.0, dtype=np.float32)
        mask = np.zeros((50, 50), dtype=np.uint8)
        heatmap = build_canopy_heatmap(canopy_temp, mask)
        np.testing.assert_array_equal(heatmap, 0)

    def test_output_shape_and_dtype(self):
        canopy_temp = np.full((100, 100), 30.0, dtype=np.float32)
        mask = np.full((100, 100), 255, dtype=np.uint8)
        heatmap = build_canopy_heatmap(canopy_temp, mask)
        assert heatmap.shape == (100, 100, 3)
        assert heatmap.dtype == np.uint8


class TestSetTempRange:
    def test_valid_range_accepted(self):
        set_temp_range(15.0, 50.0)

    def test_invalid_range_raises(self):
        with pytest.raises(ValueError):
            set_temp_range(40.0, 20.0)  # min >= max

    def teardown_method(self, method):
        # Restore defaults after each test
        set_temp_range(20.0, 45.0)
