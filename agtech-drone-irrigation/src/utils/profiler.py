"""
src/utils/profiler.py
======================
Lightweight wall-clock profiler for pipeline latency measurement.
"""

from __future__ import annotations

import time


class Profiler:
    """
    Simple context-manager and manual start/stop wall-clock profiler.

    Usage (manual)::

        p = Profiler()
        p.start()
        do_work()
        elapsed_ms = p.stop()

    Usage (context manager)::

        with Profiler() as p:
            do_work()
        print(p.elapsed_ms)
    """

    def __init__(self) -> None:
        self._t0: float | None = None
        self.elapsed_ms: float = 0.0

    def start(self) -> "Profiler":
        self._t0 = time.perf_counter()
        return self

    def stop(self) -> float:
        """Stop timing and return elapsed milliseconds."""
        if self._t0 is None:
            raise RuntimeError("Profiler.start() was not called.")
        self.elapsed_ms = (time.perf_counter() - self._t0) * 1000.0
        self._t0 = None
        return self.elapsed_ms

    def fps(self) -> float:
        """Return frames-per-second from the last recorded elapsed time."""
        if self.elapsed_ms <= 0:
            return float("inf")
        return 1000.0 / self.elapsed_ms

    # ── Context manager support ──────────────────────────────────────────

    def __enter__(self) -> "Profiler":
        return self.start()

    def __exit__(self, *args: object) -> None:
        self.stop()
