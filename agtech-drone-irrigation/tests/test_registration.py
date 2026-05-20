"""tests/test_registration.py"""

import numpy as np
import pytest

from src.pipeline.registration import (
    align_thermal_to_rgb,
    make_correction_matrix,
    estimate_matrix_from_keypoints,
    DEFAULT_MATRIX,
)


class TestMakeCorrectionMatrix:
    def test_shape(self):
        M = make_correction_matrix(3.0, -2.0)
        assert M.shape == (2, 3)

    def test_dtype(self):
        M = make_correction_matrix(0, 0)
        assert M.dtype == np.float32

    def test_identity_shift(self):
        M = make_correction_matrix(0.0, 0.0)
        expected = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        np.testing.assert_array_almost_equal(M, expected)

    def test_values(self):
        M = make_correction_matrix(-3.0, -2.0)
        assert M[0, 2] == -3.0
        assert M[1, 2] == -2.0


class TestAlignThermalToRgb:
    def test_output_shape_matches_target(self):
        thermal = np.random.randint(0, 255, (300, 400), dtype=np.uint8)
        aligned = align_thermal_to_rgb(thermal, (400, 300))
        assert aligned.shape == (300, 400)

    def test_identity_matrix_preserves_image(self):
        thermal = np.arange(0, 100, dtype=np.uint8).reshape(10, 10)
        M = make_correction_matrix(0.0, 0.0)
        aligned = align_thermal_to_rgb(thermal, (10, 10), matrix=M)
        np.testing.assert_array_equal(thermal, aligned)

    def test_invalid_matrix_shape_raises(self):
        thermal = np.zeros((100, 100), dtype=np.uint8)
        bad_matrix = np.eye(3, dtype=np.float32)  # 3x3, not 2x3
        with pytest.raises(ValueError, match="shape"):
            align_thermal_to_rgb(thermal, (100, 100), matrix=bad_matrix)

    def test_border_replicate_no_black_corners(self):
        """A large shift should not produce pure-zero corners (BORDER_REPLICATE)."""
        thermal = np.full((100, 100), 200, dtype=np.uint8)
        M = make_correction_matrix(-30.0, -20.0)
        aligned = align_thermal_to_rgb(thermal, (100, 100), matrix=M)
        # Top-left corner should NOT be zero (would be if BORDER_CONSTANT)
        assert aligned[0, 0] > 0


class TestEstimateMatrixFromKeypoints:
    def test_returns_default_on_featureless_images(self):
        """Uniform images have no keypoints — should fall back to default."""
        rgb = np.full((200, 200, 3), 128, dtype=np.uint8)
        thermal = np.full((200, 200), 100, dtype=np.uint8)
        M = estimate_matrix_from_keypoints(rgb, thermal)
        assert M.shape == (2, 3)
