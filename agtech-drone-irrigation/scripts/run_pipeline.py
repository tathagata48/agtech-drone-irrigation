#!/usr/bin/env python3
"""
scripts/run_pipeline.py
========================
CLI entry point for the precision irrigation pipeline.

Usage examples
--------------
    # Launch Gradio dashboard with synthetic demo data
    python scripts/run_pipeline.py --demo

    # Process a specific RGB + Thermal pair
    python scripts/run_pipeline.py \\
        --rgb path/to/rgb.jpg \\
        --thermal path/to/thermal.jpg \\
        --threshold 30.0 \\
        --out-dir results/

    # Launch Gradio dashboard (upload your own images)
    python scripts/run_pipeline.py --dashboard
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import os

# Allow running from repo root without installing the package
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import cv2

from src.data.synthetic import generate_synthetic_farm_data
from src.pipeline.fusion import process_precision_irrigation
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_pipeline",
        description="Precision irrigation sensor fusion pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--demo",      action="store_true",
                      help="Generate synthetic data and run the pipeline once.")
    mode.add_argument("--dashboard", action="store_true",
                      help="Launch the Gradio interactive dashboard.")
    mode.add_argument("--rgb",       metavar="PATH",
                      help="Path to the RGB drone image.")

    parser.add_argument("--thermal",   metavar="PATH",
                        help="Path to the thermal image (required with --rgb).")
    parser.add_argument("--threshold", type=float, default=30.0,
                        help="Stress temperature threshold in °C (default: 30.0).")
    parser.add_argument("--out-dir",   metavar="DIR", default="results",
                        help="Output directory for saved images (default: results/).")
    return parser.parse_args()


def run_demo(threshold: float, out_dir: pathlib.Path) -> None:
    logger.info("Generating synthetic farm data…")
    rgb_path, thr_path = generate_synthetic_farm_data(
        out_rgb=str(out_dir / "synthetic_rgb.jpg"),
        out_thermal=str(out_dir / "synthetic_thermal.jpg"),
    )
    logger.info("Running pipeline (threshold=%.1f°C)…", threshold)
    overlay, heatmap, n_stress, n_ok = process_precision_irrigation(
        rgb_path, thr_path, threshold
    )
    overlay_path  = out_dir / "overlay.jpg"
    heatmap_path  = out_dir / "heatmap.jpg"
    cv2.imwrite(str(overlay_path), overlay)
    cv2.imwrite(str(heatmap_path), heatmap)

    print(f"\n{'═'*50}")
    print(f"  Stressed zones : {n_stress}")
    print(f"  Healthy zones  : {n_ok}")
    print(f"  Overlay saved  : {overlay_path}")
    print(f"  Heatmap saved  : {heatmap_path}")
    print(f"{'═'*50}\n")


def run_pair(
    rgb_path: str,
    thermal_path: str,
    threshold: float,
    out_dir: pathlib.Path,
) -> None:
    logger.info("Processing %s + %s", rgb_path, thermal_path)
    overlay, heatmap, n_stress, n_ok = process_precision_irrigation(
        rgb_path, thermal_path, threshold
    )
    overlay_path = out_dir / "overlay.jpg"
    heatmap_path = out_dir / "heatmap.jpg"
    cv2.imwrite(str(overlay_path), overlay)
    cv2.imwrite(str(heatmap_path), heatmap)
    print(f"Stress: {n_stress}  OK: {n_ok}")
    print(f"Results saved to {out_dir}/")


def main() -> None:
    args = _parse_args()
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.demo:
        run_demo(args.threshold, out_dir)

    elif args.dashboard:
        from src.dashboard.app import main as launch_dashboard
        launch_dashboard()

    elif args.rgb:
        if not args.thermal:
            print("ERROR: --thermal is required when using --rgb", file=sys.stderr)
            sys.exit(1)
        run_pair(args.rgb, args.thermal, args.threshold, out_dir)


if __name__ == "__main__":
    main()
