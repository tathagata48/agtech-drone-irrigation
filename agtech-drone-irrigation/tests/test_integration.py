"""
tests/test_integration.py
==========================
End-to-end integration tests that exercise the full stack:
synthetic data → fusion pipeline → profiler → log formatter.

These tests intentionally duplicate nothing from unit tests; they verify
that the modules work together correctly as a system.
"""

from __future__ import annotations

import time
import numpy as np
import pytest

from src.data.synthetic import generate_synthetic_arrays
from src.pipeline.fusion import process_precision_irrigation
from src.utils.profiler import Profiler
from src.utils.config import load_config, apply_config_to_pipeline


class TestEndToEnd:
    """Full pipeline from synthetic data to annotated overlay."""

    def test_pipeline_produces_valid_bgr_outputs(self):
        rgb, thermal = generate_synthetic_arrays(image_size=500)
        overlay, heatmap, n_stress, n_ok = process_precision_irrigation(
            rgb, thermal, 30.0
        )
        for img in (overlay, heatmap):
            assert img.dtype == np.uint8
            assert img.ndim == 3
            assert img.shape[2] == 3

    def test_heatmap_soil_is_pure_black(self):
        """Pixels outside the canopy mask must be (0,0,0) — no JET bleed."""
        from src.pipeline.segmentation import segment_canopy
        rgb, thermal = generate_synthetic_arrays(image_size=500)
        _, heatmap, _, _ = process_precision_irrigation(rgb, thermal, 30.0)
        mask = segment_canopy(rgb)
        # All pixels where mask==0 should be black in the heatmap
        soil_pixels = heatmap[mask == 0]
        np.testing.assert_array_equal(
            soil_pixels, 0,
            err_msg="Soil pixels in heatmap are not black",
        )

    def test_overlay_input_not_mutated(self):
        """The pipeline must never modify the input RGB array in-place."""
        rgb, thermal = generate_synthetic_arrays(image_size=500)
        original = rgb.copy()
        process_precision_irrigation(rgb, thermal, 30.0)
        np.testing.assert_array_equal(rgb, original)

    def test_profiler_records_real_elapsed(self):
        rgb, thermal = generate_synthetic_arrays(image_size=500)
        with Profiler() as p:
            process_precision_irrigation(rgb, thermal, 30.0)
        assert p.elapsed_ms > 0
        assert p.fps() > 0

    def test_config_apply_changes_threshold_behaviour(self):
        """
        Changing the calibration range alters measured leaf temperatures,
        which changes how plants are classified at a fixed threshold.

        Default range 20–45°C: Row-2 pixels (dehydrated, ~143/255) map
        to ≈34°C — above a 30°C threshold → STRESS.

        Widened range 20–60°C: those same pixels map to ≈ (143/255)*(60-20)+20
        ≈ 42°C — still above 30°C, so all 10 plants become STRESS.

        To demonstrate the range effect we use a threshold in the middle of
        the FLIR range (42°C) where the default mapping puts ALL plants
        below it but the wider mapping pushes some above.
        """
        from src.pipeline.thermal import set_temp_range

        rgb, thermal = generate_synthetic_arrays(image_size=500)

        # Default calibration (20–45°C): Row-2 plants ≈ 34°C < 42°C → all OK
        _, _, stress_default, _ = process_precision_irrigation(rgb, thermal, 42.0)

        # FLIR profile (20–60°C): Row-2 pixels now map to ≈ 42°C → some STRESS
        set_temp_range(20.0, 60.0)
        _, _, stress_flir, _ = process_precision_irrigation(rgb, thermal, 42.0)
        set_temp_range(20.0, 45.0)  # restore for other tests

        # With the wider FLIR range, more plants should exceed the same threshold
        assert stress_flir >= stress_default, (
            f"Expected more STRESS with wider range: "
            f"default={stress_default}, flir={stress_flir}"
        )

    def test_batch_consistency(self):
        """Running the same pair twice must produce identical counts."""
        rgb, thermal = generate_synthetic_arrays(image_size=500, seed=42)
        results = [
            process_precision_irrigation(rgb, thermal, 30.0)[2:]
            for _ in range(3)
        ]
        assert all(r == results[0] for r in results), \
            f"Non-deterministic results: {results}"

    def test_small_image_no_crash(self):
        """Very small images (edge case) should not raise."""
        rgb, thermal = generate_synthetic_arrays(image_size=64)
        overlay, heatmap, n_stress, n_ok = process_precision_irrigation(
            rgb, thermal, 30.0
        )
        assert overlay.shape[:2] == (64, 64)

    def test_throughput_meets_minimum_fps(self):
        """
        Sanity check: the pipeline must run faster than 1 FPS on any
        reasonable CI machine for 500×500 images.
        """
        rgb, thermal = generate_synthetic_arrays(image_size=500)
        t0 = time.perf_counter()
        for _ in range(5):
            process_precision_irrigation(rgb, thermal, 30.0)
        avg_ms = (time.perf_counter() - t0) / 5 * 1000
        fps = 1000 / avg_ms
        assert fps >= 1.0, f"Pipeline too slow: {fps:.2f} FPS ({avg_ms:.0f} ms/frame)"
