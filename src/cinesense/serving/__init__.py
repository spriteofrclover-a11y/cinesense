"""Serving layer: FastAPI application and static frontend.

Runtime dependencies are numpy + stdlib + fastapi/pydantic. No pandas, no
scikit-learn, no scipy - those live in ``cinesense.features`` and run offline.
"""

from __future__ import annotations

from cinesense.serving.app import create_app

__all__ = ["create_app"]
