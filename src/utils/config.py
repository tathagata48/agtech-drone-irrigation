"""
src/utils/config.py
====================
YAML configuration loader with deep-merge support.

Usage
-----
    from src.utils.config import load_config

    cfg = load_config()                          # loads configs/default.yaml
    cfg = load_config("configs/flir_duo_pro.yaml")   # deep-merges over default
    threshold = cfg["pipeline"]["default_temp_threshold"]
"""

from __future__ import annotations

import pathlib
from typing import Any

import yaml

from src.utils.logger import get_logger

logger = get_logger(__name__)

_REPO_ROOT = pathlib.Path(__file__).parent.parent.parent
_DEFAULT_CONFIG = _REPO_ROOT / "configs" / "default.yaml"


def load_config(
    override_path: str | pathlib.Path | None = None,
) -> dict[str, Any]:
    """
    Load the pipeline configuration.

    Parameters
    ----------
    override_path : str | Path | None
        Path to a YAML file that will be deep-merged on top of
        `configs/default.yaml`.  Keys absent in the override are
        inherited from the default.

    Returns
    -------
    dict
        Merged configuration dictionary.
    """
    base = _load_yaml(_DEFAULT_CONFIG)

    if override_path is not None:
        override = _load_yaml(pathlib.Path(override_path))
        base = _deep_merge(base, override)
        logger.info("Config loaded: default + %s", override_path)
    else:
        logger.info("Config loaded: default only")

    return base


def _load_yaml(path: pathlib.Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data or {}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base (override wins on conflicts)."""
    result = base.copy()
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def apply_config_to_pipeline(cfg: dict[str, Any]) -> None:
    """
    Push loaded config values into the pipeline modules.

    Call this once at startup after load_config(), before processing
    any images.
    """
    from src.pipeline.thermal import set_temp_range
    from src.pipeline import registration, segmentation, thermal
    import numpy as np

    # Thermal range
    t = cfg.get("thermal", {})
    set_temp_range(
        float(t.get("temp_min_c", 20.0)),
        float(t.get("temp_max_c", 45.0)),
    )

    # Registration correction matrix
    r = cfg.get("registration", {})
    registration.DEFAULT_MATRIX = registration.make_correction_matrix(
        float(r.get("correction_dx", -3.0)),
        float(r.get("correction_dy", -2.0)),
    )

    # Segmentation HSV bounds
    s = cfg.get("segmentation", {})
    lo = s.get("hsv_lower", [35, 40, 40])
    hi = s.get("hsv_upper", [85, 255, 255])
    segmentation.HSV_LOWER_DEFAULT = np.array(lo, dtype=np.uint8)
    segmentation.HSV_UPPER_DEFAULT = np.array(hi, dtype=np.uint8)
    segmentation.MORPH_KERNEL_SIZE = int(s.get("morph_kernel_size", 5))

    # Contour filter
    p = cfg.get("pipeline", {})
    thermal.MIN_PLANT_AREA_PX = int(p.get("min_plant_area_px", 200))

    logger.info("Pipeline configured from YAML.")
