"""
src/utils/logger.py
====================
Centralised structured logger factory.

All modules call `get_logger(__name__)` for consistent formatting.
"""

from __future__ import annotations

import logging
import os
import sys


_LOG_LEVEL = os.environ.get("AGTECH_LOG_LEVEL", "INFO").upper()
_FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATEFMT = "%H:%M:%S"


def _configure_root() -> None:
    """Configure the root logger once at import time."""
    root = logging.getLogger()
    if root.handlers:
        return  # Already configured (e.g. by Gradio or pytest)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_FMT, datefmt=_DATEFMT))
    root.addHandler(handler)
    root.setLevel(getattr(logging, _LOG_LEVEL, logging.INFO))


_configure_root()


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger with the project namespace prefix."""
    return logging.getLogger(name)
