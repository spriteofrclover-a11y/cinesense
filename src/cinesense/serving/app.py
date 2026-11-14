"""The FastAPI application.

Models load once in the lifespan handler, not per request. A cold start pays
~1 s for that; every request after it is a few milliseconds.

The frontend is plain HTML/CSS/ES modules served from ``static``. There is no
build step, so nothing about the deployment depends on Node.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from cinesense import __version__
from cinesense.logging_setup import configure_logging
from cinesense.serving.routes import router
from cinesense.serving.state import get_bundle

__all__ = ["create_app"]

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    try:
        get_bundle()
        log.info("models loaded; ready to serve")
    except Exception:
        # Serve anyway: /api/health reports the failure and the static shell still
        # loads, so the user sees a real error message instead of a blank page.
        log.exception("model loading failed; the API will report degraded status")
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="CineSense",
        version=__version__,
        description="Explainable hybrid movie recommendations.",
        lifespan=lifespan,
    )
    app.include_router(router)

    @app.middleware("http")
    async def revalidate_static(request, call_next):
        """Make browsers revalidate CSS/JS instead of guessing.

        Starlette sends only ETag and Last-Modified. With no Cache-Control a
        browser applies heuristic freshness and may serve a cached stylesheet
        without asking - so an edit never reaches the user, and after a deploy
        they can run old JavaScript against a new API. `no-cache` still permits
        a cheap 304 when nothing changed.
        """
        response = await call_next(request)
        if request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-cache, must-revalidate"
        return response

    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/favicon.svg", include_in_schema=False)
    def favicon() -> FileResponse:
        return FileResponse(STATIC_DIR / "favicon.svg", media_type="image/svg+xml")

    @app.exception_handler(500)
    async def internal_error(request, exc):  # pragma: no cover - defensive
        log.exception("unhandled error on %s", request.url.path)
        return JSONResponse(
            status_code=500,
            content={"detail": "internal error", "path": request.url.path},
        )

    return app


app = create_app()
