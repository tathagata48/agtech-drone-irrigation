"""
tests/test_fusion.py
=====================
Integration tests for the end-to-end process_precision_irrigation pipeline.
"""

import numpy as np
import pytest

from src.pipeline.fusion import process_precision_irrigation


class TestReturnShapes:
    def test_overlay_matches_rgb_shape(self, synthetic_pair):
        rgb, thermal = synthetic_pair
        overlay, _, _, _ = process_precision_irrigation(rgb, thermal, 30.0)
        assert overlay.shape == rgb.shape

    def test_heatmap_matches_rgb_shape(self, synthetic_pair):
        rgb, thermal = synthetic_pair
        _, heatmap, _, _ = process_precision_irrigation(rgb, thermal, 30.0)
        assert heatmap.shape == rgb.shape

    def test_counts_are_non_negative(self, synthetic_pair):
        rgb, thermal = synthetic_pair
        _, _, n_stress, n_ok = process_precision_irrigation(rgb, thermal, 30.0)
        assert n_stress >= 0
        assert n_ok >= 0


class TestStressClassification:
    def test_threshold_30_classifies_5_stressed_5_ok(self, synthetic_pair):
        rgb, thermal = synthetic_pair
        _, _, n_stress, n_ok = process_precision_irrigation(rgb, thermal, 30.0)
        assert n_stress == 5, f"Expected 5 stressed, got {n_stress}"
        assert n_ok == 5,     f"Expected 5 OK, got {n_ok}"

    def test_high_threshold_all_healthy(self, synthetic_pair):
        """At 45°C threshold, no plant should be STRESSED."""
        rgb, thermal = synthetic_pair
        _, _, n_stress, n_ok = process_precision_irrigation(rgb, thermal, 45.0)
        assert n_stress == 0

    def test_low_threshold_all_stressed(self, synthetic_pair):
        """At 20°C threshold, all plants should be STRESSED."""
        rgb, thermal = synthetic_pair
        _, _, n_stress, n_ok = process_precision_irrigation(rgb, thermal, 20.0)
        assert n_ok == 0

    def test_stress_count_decreases_as_threshold_rises(self, synthetic_pair):
        rgb, thermal = synthetic_pair
        thresholds = [22.0, 27.0, 31.0, 38.0]
        stress_counts = [
            process_precision_irrigation(rgb, thermal, t)[2]
            for t in thresholds
        ]
        # Stress count must be non-increasing as threshold rises
        for i in range(len(stress_counts) - 1):
            assert stress_counts[i] >= stress_counts[i + 1], (
                f"Stress count increased as threshold rose: {stress_counts}"
            )


class TestInputFormats:
    def test_accepts_file_paths(self, synthetic_pair, tmp_path):
        import cv2
        rgb, thermal = synthetic_pair
        rgb_path = str(tmp_path / "rgb.jpg")
        thr_path = str(tmp_path / "thermal.jpg")
        cv2.imwrite(rgb_path, rgb)
        cv2.imwrite(thr_path, thermal)
        overlay, heatmap, n_stress, n_ok = process_precision_irrigation(
            rgb_path, thr_path, 30.0
        )
        assert overlay.shape == rgb.shape

    def test_accepts_numpy_arrays(self, synthetic_pair):
        rgb, thermal = synthetic_pair
        overlay, _, _, _ = process_precision_irrigation(rgb, thermal, 30.0)
        assert overlay is not None

    def test_thermal_resolution_mismatch_auto_resizes(self, synthetic_pair):
        """Thermal at half resolution should still produce valid output."""
        rgb, thermal = synthetic_pair
        import cv2
        small_thermal = cv2.resize(thermal, (thermal.shape[1] // 2,
                                             thermal.shape[0] // 2))
        overlay, _, _, _ = process_precision_irrigation(rgb, small_thermal, 30.0)
        assert overlay.shape == rgb.shape


class TestEdgeCases:
    def test_invalid_rgb_path_raises(self):
        with pytest.raises((ValueError, FileNotFoundError)):
            process_precision_irrigation(
                "nonexistent_rgb.jpg", "nonexistent_thermal.jpg", 30.0
            )

    def test_solid_brown_rgb_no_plants(self, blank_brown_rgb, thermal_grey):
        """All-soil image should detect zero plants."""
        import cv2
        thermal_resized = cv2.resize(
            thermal_grey,
            (blank_brown_rgb.shape[1], blank_brown_rgb.shape[0])
        )
        _, _, n_stress, n_ok = process_precision_irrigation(
            blank_brown_rgb, thermal_resized, 30.0
        )
        assert n_stress + n_ok == 0

    def test_output_images_are_bgr_uint8(self, synthetic_pair):
        rgb, thermal = synthetic_pair
        overlay, heatmap, _, _ = process_precision_irrigation(rgb, thermal, 30.0)
        assert overlay.dtype == np.uint8
        assert heatmap.dtype == np.uint8
        assert overlay.ndim == 3 and overlay.shape[2] == 3
        assert heatmap.ndim == 3 and heatmap.shape[2] == 3
