"""tests/test_config.py"""

import pathlib
import pytest
import numpy as np

from src.utils.config import load_config, _deep_merge


class TestDeepMerge:
    def test_base_keys_preserved(self):
        base     = {"a": 1, "b": {"c": 2, "d": 3}}
        override = {"b": {"c": 99}}
        result   = _deep_merge(base, override)
        assert result["a"] == 1
        assert result["b"]["d"] == 3   # not overridden

    def test_override_wins(self):
        base     = {"x": 10}
        override = {"x": 20}
        assert _deep_merge(base, override)["x"] == 20

    def test_nested_override(self):
        base     = {"thermal": {"temp_min_c": 20.0, "temp_max_c": 45.0}}
        override = {"thermal": {"temp_max_c": 60.0}}
        result   = _deep_merge(base, override)
        assert result["thermal"]["temp_min_c"] == 20.0
        assert result["thermal"]["temp_max_c"] == 60.0


class TestLoadConfig:
    def test_default_config_loads(self):
        cfg = load_config()
        assert "pipeline" in cfg
        assert "thermal" in cfg
        assert "segmentation" in cfg

    def test_default_thermal_range(self):
        cfg = load_config()
        assert cfg["thermal"]["temp_min_c"] == 20.0
        assert cfg["thermal"]["temp_max_c"] == 45.0

    def test_override_config_merges(self):
        cfg = load_config("configs/flir_duo_pro.yaml")
        # FLIR profile overrides temp_max_c to 60
        assert cfg["thermal"]["temp_max_c"] == 60.0
        # But pipeline section comes from default
        assert "default_temp_threshold" in cfg["pipeline"]

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("configs/nonexistent_sensor.yaml")
