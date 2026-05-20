"""tests/test_colormap.py"""

import numpy as np
import pytest

from src.utils.colormap import apply_jet_heatmap, draw_temperature_legend


class TestApplyJetHeatmap:
    def test_output_shape_3ch(self):
        grey = np.zeros((100, 100), dtype=np.uint8)
        result = apply_jet_heatmap(grey)
        assert result.shape == (100, 100, 3)

    def test_mask_zeros_out_soil(self):
        grey = np.full((50, 50), 200, dtype=np.uint8)
        mask = np.zeros((50, 50), dtype=np.uint8)   # all soil
        result = apply_jet_heatmap(grey, mask=mask)
        np.testing.assert_array_equal(result, 0)

    def test_blend_with_base(self):
        grey = np.full((50, 50), 128, dtype=np.uint8)
        base = np.full((50, 50, 3), 100, dtype=np.uint8)
        result = apply_jet_heatmap(grey, alpha=0.5, base_bgr=base)
        assert result.shape == (50, 50, 3)
        # Blended result should differ from pure heatmap
        pure = apply_jet_heatmap(grey)
        assert not np.array_equal(result, pure)

    def test_output_dtype_uint8(self):
        grey = np.zeros((20, 20), dtype=np.uint8)
        result = apply_jet_heatmap(grey)
        assert result.dtype == np.uint8


class TestDrawTemperatureLegend:
    def test_returns_image(self):
        img = np.zeros((300, 400, 3), dtype=np.uint8)
        result = draw_temperature_legend(img, 20.0, 45.0)
        assert result is img  # modifies in-place and returns same object

    def test_legend_draws_non_zero_pixels(self):
        img = np.zeros((300, 400, 3), dtype=np.uint8)
        draw_temperature_legend(img, 20.0, 45.0, pos=(10, 10))
        # The legend strip should contain coloured pixels
        strip = img[10:160, 10:30]
        assert strip.max() > 0
