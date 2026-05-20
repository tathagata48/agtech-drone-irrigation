"""tests/test_loaders.py"""

import numpy as np
import pytest
import cv2

from src.data.loaders import load_rgb, load_thermal, validate_pair, _normalise_to_uint8


class TestLoadRgb:
    def test_from_file_path(self, tmp_path, rgb_bgr):
        path = str(tmp_path / "test_rgb.jpg")
        cv2.imwrite(path, rgb_bgr)
        loaded = load_rgb(path)
        assert loaded.shape == rgb_bgr.shape
        assert loaded.dtype == np.uint8

    def test_from_bgr_array(self, rgb_bgr):
        loaded = load_rgb(rgb_bgr, colour_order="bgr")
        np.testing.assert_array_equal(loaded, rgb_bgr)

    def test_from_rgb_array_converts_to_bgr(self, rgb_bgr):
        rgb_rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        loaded  = load_rgb(rgb_rgb, colour_order="rgb")
        # After round-trip BGR→RGB→BGR the arrays should match
        np.testing.assert_array_equal(loaded, rgb_bgr)

    def test_bad_path_raises(self):
        with pytest.raises((ValueError, FileNotFoundError)):
            load_rgb("does_not_exist.jpg")

    def test_wrong_ndim_raises(self):
        with pytest.raises(ValueError):
            load_rgb(np.zeros((100, 100), dtype=np.uint8))

    def test_unsupported_extension_raises(self, tmp_path):
        bad = tmp_path / "img.xyz"
        bad.write_bytes(b"fake")
        with pytest.raises(ValueError, match="extension"):
            load_rgb(str(bad))


class TestLoadThermal:
    def test_from_file_path(self, tmp_path, thermal_grey):
        path = str(tmp_path / "test_thermal.jpg")
        cv2.imwrite(path, thermal_grey)
        loaded = load_thermal(path)
        assert loaded.ndim == 2
        assert loaded.dtype == np.uint8

    def test_from_greyscale_array(self, thermal_grey):
        loaded = load_thermal(thermal_grey)
        np.testing.assert_array_equal(loaded, thermal_grey)

    def test_3ch_array_collapsed_to_2d(self, thermal_grey):
        # Simulate Gradio wrapping greyscale in 3 channels
        fake_3ch = cv2.cvtColor(thermal_grey, cv2.COLOR_GRAY2RGB)
        loaded   = load_thermal(fake_3ch)
        assert loaded.ndim == 2

    def test_16bit_normalised_to_uint8(self):
        arr16 = np.arange(0, 65536, dtype=np.uint16).reshape(256, 256)
        loaded = load_thermal(arr16, force_8bit=True)
        assert loaded.dtype == np.uint8
        assert loaded.max() == 255
        assert loaded.min() == 0


class TestValidatePair:
    def test_valid_pair_does_not_raise(self, rgb_bgr, thermal_grey):
        validate_pair(rgb_bgr, thermal_grey)  # should not raise

    def test_wrong_rgb_ndim_raises(self, thermal_grey):
        bad = np.zeros((100, 100), dtype=np.uint8)
        with pytest.raises(ValueError, match="HxWx3"):
            validate_pair(bad, thermal_grey)

    def test_wrong_thermal_ndim_raises(self, rgb_bgr):
        bad = np.zeros((100, 100, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="HxW"):
            validate_pair(rgb_bgr, bad)

    def test_empty_rgb_raises(self, thermal_grey):
        empty = np.zeros((0, 0, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="empty"):
            validate_pair(empty, thermal_grey)


class TestNormaliseToUint8:
    def test_zero_range_returns_zeros(self):
        arr = np.full((10, 10), 42.0, dtype=np.float32)
        result = _normalise_to_uint8(arr)
        np.testing.assert_array_equal(result, 0)

    def test_full_range_maps_to_0_255(self):
        arr = np.array([0.0, 127.5, 255.0], dtype=np.float32)
        result = _normalise_to_uint8(arr)
        assert result[0]  == 0
        assert result[-1] == 255
