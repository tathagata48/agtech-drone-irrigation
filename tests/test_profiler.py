"""tests/test_profiler.py"""

import time
import pytest

from src.utils.profiler import Profiler


class TestProfilerManual:
    def test_elapsed_ms_is_positive(self):
        p = Profiler()
        p.start()
        time.sleep(0.01)
        ms = p.stop()
        assert ms > 0

    def test_elapsed_ms_ballpark(self):
        p = Profiler()
        p.start()
        time.sleep(0.05)
        ms = p.stop()
        # Allow generous tolerance for CI runners
        assert 30 < ms < 500, f"Unexpected elapsed: {ms:.1f} ms"

    def test_stop_without_start_raises(self):
        p = Profiler()
        with pytest.raises(RuntimeError, match="start"):
            p.stop()

    def test_fps_after_stop(self):
        p = Profiler()
        p.start()
        time.sleep(0.01)
        p.stop()
        fps = p.fps()
        assert fps > 0

    def test_fps_on_fresh_profiler_returns_inf(self):
        p = Profiler()
        assert p.fps() == float("inf")


class TestProfilerContextManager:
    def test_context_manager_records_elapsed(self):
        with Profiler() as p:
            time.sleep(0.01)
        assert p.elapsed_ms > 0

    def test_context_manager_no_exception_on_exit(self):
        # Should not raise
        with Profiler():
            pass
