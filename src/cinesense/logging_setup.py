"""Console logging that is readable by a human and greppable by a script."""

from __future__ import annotations

import logging
import sys

__all__ = ["configure_logging", "step"]

_FORMAT = "%(asctime)s  %(levelname)-7s %(message)s"
_DATE_FORMAT = "%H:%M:%S"

#: Printed once per pipeline phase by cli.py.
_STEP_TEMPLATE = "\n== {n}/{total}  {title} {pad}"


def configure_logging(level: int = logging.INFO) -> None:
    """Idempotent root logger setup (safe to call more than once)."""
    root = logging.getLogger("cinesense")
    if getattr(root, "_cinesense_configured", False):
        root.setLevel(level)
        return

    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT))

    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
    root.propagate = False
    root._cinesense_configured = True  # type: ignore[attr-defined]


def step(n: int, total: int, title: str, width: int = 58) -> str:
    """Format a phase header, e.g. ``== 2/6  Ingesting CSVs ====``."""
    pad = "=" * max(0, width - len(title))
    return _STEP_TEMPLATE.format(n=n, total=total, title=title, pad=pad)
