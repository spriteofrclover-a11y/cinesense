"""Request and response models.

Ratings arrive from the browser on every request, so they are validated hard:
a 9-star rating or a 100k-entry payload should be a 422, not a slow 500.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field, field_validator

__all__ = [
    "ExplanationOut",
    "GenreOut",
    "HealthOut",
    "MovieOut",
    "PreferencesIn",
    "RecommendIn",
    "RecommendationOut",
    "StatsOut",
]

#: Ratings outside this range are meaningless to a model trained on 0.5-5.
MIN_RATING = 0.5
MAX_RATING = 5.0

#: Bounds the work one request can ask for.
MAX_RATINGS = 2000
MAX_EXCLUDE = 2000


class PreferencesIn(BaseModel):
    """Cold-start onboarding answers."""

    favorite_movie_ids: list[int] = Field(default_factory=list, max_length=50)
    genres: list[str] = Field(default_factory=list, max_length=12)
    min_year: int | None = Field(default=None, ge=1880, le=2100)
    max_year: int | None = Field(default=None, ge=1880, le=2100)
    #: ISO 639-1 language codes ("hi", "ta").
    languages: list[str] = Field(default_factory=list, max_length=20)
    #: ISO 3166-1 country codes ("IN") - broader than a language.
    regions: list[str] = Field(default_factory=list, max_length=20)


class RecommendIn(BaseModel):
    ratings: dict[int, float] = Field(default_factory=dict)
    prefs: PreferencesIn | None = None
    k: int = Field(default=20, ge=1, le=60)
    exclude: list[int] = Field(default_factory=list, max_length=MAX_EXCLUDE)

    @field_validator("ratings")
    @classmethod
    def _validate_ratings(cls, value: dict[int, float]) -> dict[int, float]:
        if len(value) > MAX_RATINGS:
            raise ValueError(f"at most {MAX_RATINGS} ratings per request")
        for movie_id, rating in value.items():
            if movie_id <= 0:
                raise ValueError(f"invalid movie id {movie_id}")
            if not MIN_RATING <= rating <= MAX_RATING:
                raise ValueError(
                    f"rating {rating} for movie {movie_id} outside "
                    f"{MIN_RATING}-{MAX_RATING}"
                )
        return value

    @field_validator("prefs")
    @classmethod
    def _validate_years(cls, value: PreferencesIn | None) -> PreferencesIn | None:
        if value and value.min_year and value.max_year and value.min_year > value.max_year:
            raise ValueError("min_year must not exceed max_year")
        return value


class MovieOut(BaseModel):
    id: int
    title: str
    year: int | None = None
    genres: list[str] = Field(default_factory=list)
    poster: str | None = None
    avg_rating: float = 0.0
    n_ratings: int = 0
    language: str | None = None


class ExplanationOut(BaseModel):
    content: float = 0.0
    collaborative: float = 0.0
    popularity: float = 0.0
    reasons: list[str] = Field(default_factory=list)
    matched_genres: list[str] = Field(default_factory=list)
    matched_terms: list[str] = Field(default_factory=list)


class RecommendationOut(MovieOut):
    score: float = 0.0
    explanation: ExplanationOut = Field(default_factory=ExplanationOut)


class MovieDetailOut(MovieOut):
    terms: list[str] = Field(default_factory=list)


class GenreOut(BaseModel):
    name: str
    count: int


class FacetOut(BaseModel):
    """A filterable facet: language code or country code, with a display name."""

    code: str
    name: str
    count: int


class FacetsOut(BaseModel):
    languages: list[FacetOut] = Field(default_factory=list)
    regions: list[FacetOut] = Field(default_factory=list)


class MovieIdsIn(BaseModel):
    """Bulk metadata lookup, used by the taste profile in the browser."""

    ids: list[int] = Field(default_factory=list, max_length=300)


class HealthOut(BaseModel):
    status: str
    version: str
    models: dict[str, bool]
    movies: int


class StatsOut(BaseModel):
    dataset: dict
    content: dict
    collaborative: dict
    posters: dict
    ranking: dict | None = None


Score = Annotated[float, Field(ge=-1.0, le=10.0)]
