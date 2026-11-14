"""CineSense - an explainable hybrid movie recommender.

Package layout:

    cinesense.config           typed settings loaded from configs/base.toml
    cinesense.data             download, ingest, preprocess (offline only)
    cinesense.cli              command line entry point
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
