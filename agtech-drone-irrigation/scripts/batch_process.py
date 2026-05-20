#!/usr/bin/env python3
"""
scripts/batch_process.py
=========================
Batch-process a directory of paired RGB + Thermal drone captures.

Naming convention: the script expects matching filenames in two sub-folders:
    <input_dir>/rgb/<name>.jpg
    <input_dir>/thermal/<name>.jpg

Usage
-----
    python scripts/batch_process.py \\
        --input-dir /data/field_20240615 \\
        --threshold 30.0 \\
        --out-dir   /data/results_20240615 \\
        --workers   4
"""

from __future__ import annotations

import argparse
import concurrent.futures
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import cv2

from src.pipeline.fusion import process_precision_irrigation
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch precision irrigation pipeline")
    p.add_argument("--input-dir",  required=True, metavar="DIR")
    p.add_argument("--threshold",  type=float,    default=30.0)
    p.add_argument("--out-dir",    required=True, metavar="DIR")
    p.add_argument("--workers",    type=int,      default=1,
                   help="Number of parallel worker threads (default: 1).")
    p.add_argument("--ext",        default=".jpg",
                   help="Image file extension (default: .jpg).")
    return p.parse_args()


def _process_one(
    rgb_path: pathlib.Path,
    thermal_path: pathlib.Path,
    threshold: float,
    out_dir: pathlib.Path,
) -> dict:
    t0 = time.perf_counter()
    try:
        overlay, heatmap, n_stress, n_ok = process_precision_irrigation(
            str(rgb_path), str(thermal_path), threshold
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        stem = rgb_path.stem
        cv2.imwrite(str(out_dir / f"{stem}_overlay.jpg"),  overlay)
        cv2.imwrite(str(out_dir / f"{stem}_heatmap.jpg"),  heatmap)

        return {
            "file": stem,
            "stress": n_stress,
            "ok": n_ok,
            "ms": elapsed_ms,
            "error": None,
        }
    except Exception as exc:
        return {"file": rgb_path.stem, "error": str(exc)}


def main() -> None:
    args = _parse_args()
    in_dir   = pathlib.Path(args.input_dir)
    out_dir  = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rgb_dir     = in_dir / "rgb"
    thermal_dir = in_dir / "thermal"

    if not rgb_dir.exists() or not thermal_dir.exists():
        print(f"ERROR: Expected {rgb_dir}/ and {thermal_dir}/ sub-folders.",
              file=sys.stderr)
        sys.exit(1)

    rgb_files = sorted(rgb_dir.glob(f"*{args.ext}"))
    if not rgb_files:
        print(f"No {args.ext} files found in {rgb_dir}", file=sys.stderr)
        sys.exit(1)

    pairs = []
    for rgb_path in rgb_files:
        thr_path = thermal_dir / rgb_path.name
        if thr_path.exists():
            pairs.append((rgb_path, thr_path))
        else:
            logger.warning("No thermal match for %s — skipping.", rgb_path.name)

    logger.info("Processing %d pairs with %d worker(s)…",
                len(pairs), args.workers)

    results = []
    if args.workers > 1:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.workers
        ) as pool:
            futures = {
                pool.submit(
                    _process_one, rgb, thr, args.threshold, out_dir
                ): rgb.stem
                for rgb, thr in pairs
            }
            for fut in concurrent.futures.as_completed(futures):
                results.append(fut.result())
    else:
        for rgb, thr in pairs:
            results.append(_process_one(rgb, thr, args.threshold, out_dir))

    # ── Summary ────────────────────────────────────────────────────────────
    ok_results = [r for r in results if r.get("error") is None]
    err_results = [r for r in results if r.get("error") is not None]

    total_stress  = sum(r["stress"] for r in ok_results)
    total_healthy = sum(r["ok"]     for r in ok_results)
    avg_ms        = (sum(r["ms"] for r in ok_results) / len(ok_results)
                     if ok_results else 0)

    print(f"\n{'═'*55}")
    print(f"  Frames processed : {len(ok_results)} / {len(pairs)}")
    print(f"  Errors           : {len(err_results)}")
    print(f"  Total stressed   : {total_stress}")
    print(f"  Total healthy    : {total_healthy}")
    print(f"  Avg latency      : {avg_ms:.1f} ms")
    print(f"  Results dir      : {out_dir}/")
    print(f"{'═'*55}\n")

    if err_results:
        print("Failed files:")
        for r in err_results:
            print(f"  {r['file']}: {r['error']}")


if __name__ == "__main__":
    main()
