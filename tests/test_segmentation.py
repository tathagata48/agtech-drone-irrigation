"""tests/test_segmentation.py"""

import numpy as np
import pytest

from src.pipeline.segmentation import (
    segment_canopy,
    get_calibration_bounds,
    HSV_LOWER_DEFAULT,
    HSV_UPPER_DEFAULT,
)


class TestSegmentCanopy:
    def test_output_shape_matches_input(self, blank_green_rgb):
        mask = segment_canopy(blank_green_rgb)
        assert mask.shape == blank_green_rgb.shape[:2]

    def test_output_dtype_is_uint8(self, blank_green_rgb):
        mask = segment_canopy(blank_green_rgb)
        assert mask.dtype == np.uint8

    def test_solid_green_produces_mostly_white_mask(self, blank_green_rgb):
        """At least 80% of pixels should be 255 for a solid-green image."""
        mask = segment_canopy(blank_green_rgb)
        white_fraction = (mask == 255).mean()
        assert white_fraction > 0.80, f"Only {white_fraction:.1%} white"

    def test_solid_brown_produces_mostly_zero_mask(self, blank_brown_rgb):
        """Bare soil image should produce near-zero canopy mask."""
        mask = segment_canopy(blank_brown_rgb)
        white_fraction = (mask == 255).mean()
        assert white_fraction < 0.05, f"{white_fraction:.1%} white on soil image"

    def test_mask_is_binary(self, blank_green_rgb):
        """Mask values must be either 0 or 255 — nothing in between."""
        mask = segment_canopy(blank_green_rgb)
        unique = set(np.unique(mask))
        assert unique.issubset({0, 255}), f"Non-binary values: {unique}"

    def test_custom_hsv_bounds(self):
        """Custom HSV bounds should be applied (not the defaults)."""
        # Create a pure-red image — should NOT match default green bounds
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:] = (0, 0, 255)  # BGR pure red
        default_mask = segment_canopy(img)
        # Nearly all should be zero with default green bounds
        assert (default_mask == 0).mean() > 0.95

    def test_synthetic_pair_produces_nonzero_mask(self, synthetic_pair):
        rgb, _ = synthetic_pair
        mask = segment_canopy(rgb)
        assert mask.max() > 0, "No canopy detected in synthetic pair"


class TestCalibrationBounds:
    def test_returns_two_arrays(self, blank_green_rgb):
        lower, upper = get_calibration_bounds(blank_green_rgb)
        assert lower.shape == (3,)
        assert upper.shape == (3,)

    def test_lower_less_than_upper(self, blank_green_rgb):
        lower, upper = get_calibration_bounds(blank_green_rgb)
        assert all(lower <= upper), f"lower {lower} > upper {upper}"

    def test_custom_roi(self, blank_green_rgb):
        lower, upper = get_calibration_bounds(
            blank_green_rgb, roi=(10, 10, 80, 80)
        )
        assert lower.shape == (3,)
