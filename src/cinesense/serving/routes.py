"""HTTP endpoints.

Every route is stateless: ratings arrive in the request body and are scored on
the fly. There is no session, no database and no per-user server state, which is
what lets this deploy to a serverless platform unchanged.

All three routers live in one module on purpose - twelve endpoints with shared
helpers read better in one file than split across a package.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Query

from cinesense import __version__
from cinesense.recommend.hybrid import PreferenceProfile
from cinesense.serving.schemas import (
    ExplanationOut,
    FacetsOut,
    GenreOut,
    HealthOut,
    MovieDetailOut,
    MovieIdsIn,
    MovieOut,
    RecommendationOut,
    RecommendIn,
    StatsOut,
)
from cinesense.serving.state import ModelBundle, get_bundle

__all__ = ["router"]

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _movie(bundle: ModelBundle, movie_id: int) -> MovieOut:
    meta = bundle.content.meta(movie_id)
    if meta is None:
        raise HTTPException(status_code=404, detail=f"movie {movie_id} not found")
    return MovieOut(
        id=meta.movie_id,
        title=meta.title,
        year=meta.year,
        genres=list(meta.genres),
        poster=bundle.poster_url(meta.movie_id),
        avg_rating=round(meta.avg_rating, 2),
        n_ratings=meta.n_ratings,
        language=bundle.language(meta.movie_id),
    )


def _recommendation(bundle: ModelBundle, rec) -> RecommendationOut:
    meta = bundle.content.meta(rec.movie_id)
    return RecommendationOut(
        id=rec.movie_id,
        title=rec.title,
        year=rec.year,
        genres=list(rec.genres),
        poster=bundle.poster_url(rec.movie_id),
        avg_rating=round(meta.avg_rating, 2) if meta else 0.0,
        n_ratings=meta.n_ratings if meta else 0,
        language=bundle.language(rec.movie_id),
        score=round(rec.score, 4),
        explanation=ExplanationOut(
            content=round(rec.explanation.content, 4),
            collaborative=round(rec.explanation.collaborative, 4),
            popularity=round(rec.explanation.popularity, 4),
            reasons=list(rec.explanation.reasons),
            matched_genres=list(rec.explanation.matched_genres),
            matched_terms=list(rec.explanation.matched_terms),
        ),
    )


# --------------------------------------------------------------------------- #
# Meta
# --------------------------------------------------------------------------- #


@router.get("/health", response_model=HealthOut)
def health() -> HealthOut:
    bundle = get_bundle()
    return HealthOut(
        status="ok",
        version=__version__,
        models={"content": True, "collaborative": bundle.collab is not None},
        movies=len(bundle.content),
    )


@router.get("/stats", response_model=StatsOut)
def stats() -> StatsOut:
    bundle = get_bundle()
    data = bundle.stats
    return StatsOut(
        dataset=data["dataset"],
        content=data["content"],
        collaborative=data["collaborative"],
        posters=data["posters"],
        ranking=data.get("ranking"),
    )


@router.get("/facets", response_model=FacetsOut)
def facets() -> FacetsOut:
    """Language and country filters, counts computed at startup."""
    data = get_bundle().stats.get("facets") or {"languages": [], "regions": []}
    return FacetsOut(**data)


@router.get("/genres", response_model=list[GenreOut])
def genres() -> list[GenreOut]:
    """Genre facets, ordered by how many films carry them."""
    bundle = get_bundle()
    counts: dict[str, int] = {}
    for movie_id in bundle.content.movie_ids:
        meta = bundle.content.meta(int(movie_id))
        if meta is None:
            continue
        for genre in meta.genres:
            counts[genre] = counts.get(genre, 0) + 1
    return [
        GenreOut(name=name, count=count)
        for name, count in sorted(counts.items(), key=lambda item: -item[1])
    ]


# --------------------------------------------------------------------------- #
# Movies
# --------------------------------------------------------------------------- #


@router.get("/search", response_model=list[MovieOut])
def search(
    q: str = Query("", min_length=0, max_length=120),
    limit: int = Query(8, ge=1, le=30),
) -> list[MovieOut]:
    bundle = get_bundle()
    return [_movie(bundle, meta.movie_id) for meta in bundle.content.search(q, limit=limit)]


@router.get("/movies/{movie_id}", response_model=MovieDetailOut)
def movie(movie_id: int) -> MovieDetailOut:
    bundle = get_bundle()
    base = _movie(bundle, movie_id)
    return MovieDetailOut(
        **base.model_dump(),
        terms=[term.split(":", 1)[-1] for term in bundle.content.terms(movie_id)],
    )


@router.post("/movies", response_model=list[MovieOut])
def movies_by_ids(body: MovieIdsIn) -> list[MovieOut]:
    """Metadata for many films at once. Unknown ids are skipped, not an error."""
    bundle = get_bundle()
    return [_movie(bundle, movie_id) for movie_id in body.ids if bundle.content.meta(movie_id)]


@router.get("/movies/{movie_id}/similar", response_model=list[RecommendationOut])
def similar(movie_id: int, k: int = Query(12, ge=1, le=40)) -> list[RecommendationOut]:
    """Content neighbours - "more like this"."""
    bundle = get_bundle()
    if bundle.content.meta(movie_id) is None:
        raise HTTPException(status_code=404, detail=f"movie {movie_id} not found")
    return [_recommendation(bundle, rec) for rec in bundle.content.similar_to(movie_id, k=k)]


@router.get("/movies/{movie_id}/related", response_model=list[MovieOut])
def related(movie_id: int, k: int = Query(12, ge=1, le=40)) -> list[MovieOut]:
    """Collaborative neighbours - what people with similar taste rated highly."""
    bundle = get_bundle()
    if bundle.collab is None:
        return []
    if bundle.content.meta(movie_id) is None:
        raise HTTPException(status_code=404, detail=f"movie {movie_id} not found")

    results: list[MovieOut] = []
    for rec in bundle.collab.similar_items(movie_id, k=k * 2):
        meta = bundle.content.meta(rec.movie_id)
        if meta is None or meta.n_ratings < 20:
            continue
        results.append(_movie(bundle, rec.movie_id))
        if len(results) >= k:
            break
    return results


# --------------------------------------------------------------------------- #
# Recommendations
# --------------------------------------------------------------------------- #


@router.post("/recommendations", response_model=list[RecommendationOut])
def recommendations(body: RecommendIn) -> list[RecommendationOut]:
    bundle = get_bundle()

    prefs = None
    if body.prefs is not None:
        prefs = PreferenceProfile(
            favorite_movie_ids=tuple(body.prefs.favorite_movie_ids),
            genres=tuple(body.prefs.genres),
            min_year=body.prefs.min_year,
            max_year=body.prefs.max_year,
            languages=tuple(body.prefs.languages),
            regions=tuple(body.prefs.regions),
        )

    results = bundle.ranker.recommend(
        body.ratings, k=body.k, prefs=prefs, exclude=body.exclude
    )
    return [_recommendation(bundle, rec) for rec in results]
