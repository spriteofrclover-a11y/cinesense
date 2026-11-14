"""Human-readable justifications.

Kept separate from scoring so the language of an explanation can change without
touching the maths, and so it can be unit-tested with plain strings.
"""

from __future__ import annotations

from typing import Sequence

from cinesense.recommend.contracts import Explanation

__all__ = [
    "content_explanation",
    "profile_explanation",
    "shared_terms",
    "shared_values",
]

#: How many chips to surface. More than this and the UI looks like noise.
MAX_CHIPS = 4


def shared_values(left: Sequence[str], right: Sequence[str], limit: int = MAX_CHIPS) -> tuple[str, ...]:
    """Ordered, de-duplicated intersection of two short string sequences."""
    right_set = {value.lower() for value in right}
    seen: set[str] = set()
    out: list[str] = []
    for value in left:
        key = value.lower()
        if key in right_set and key not in seen:
            seen.add(key)
            out.append(value)
            if len(out) >= limit:
                break
    return tuple(out)


def shared_terms(
    left: Sequence[str], right: Sequence[str], limit: int = MAX_CHIPS
) -> tuple[str, ...]:
    """Same as :func:`shared_values` but strips the ``field:`` prefix.

    Feature names carry their origin (``tags:space``) so the build can tell two
    identically-spelled tokens apart. That prefix is noise to a reader.

    Genre tokens are excluded entirely: they are already reported as
    ``matched_genres``, and keeping them produced "Shares Sci-Fi with Arrival"
    followed by "Both described as scifi" - the same fact twice.
    """
    def bare(term: str) -> str | None:
        field, _, value = term.partition(":")
        if field == "genres":
            return None
        return value or field

    cleaned_left = [value for value in map(bare, left) if value]
    cleaned_right = [value for value in map(bare, right) if value]
    return shared_values(tuple(cleaned_left), tuple(cleaned_right), limit)


def _join(values: Sequence[str]) -> str:
    if len(values) == 1:
        return values[0]
    return ", ".join(values[:-1]) + " and " + values[-1]


def content_explanation(
    *,
    score: float,
    query_title: str,
    query_genres: Sequence[str] = (),
    query_terms: Sequence[str] = (),
    candidate_genres: Sequence[str] = (),
    candidate_terms: Sequence[str] = (),
    query_year: int | None = None,
    candidate_year: int | None = None,
) -> Explanation:
    """Explain a 'more like this' result."""
    genres = shared_values(query_genres, candidate_genres, limit=3)
    terms = shared_terms(query_terms, candidate_terms, limit=MAX_CHIPS)

    reasons: list[str] = []
    if genres:
        reasons.append(f"Shares {_join(list(genres))} with {query_title}")
    if terms:
        reasons.append(f"Both described as {_join(list(terms))}")
    if (
        query_year is not None
        and candidate_year is not None
        and abs(query_year - candidate_year) <= 3
    ):
        reasons.append(f"Released around the same time ({candidate_year})")
    if not reasons:
        reasons.append(f"Close overall match to {query_title}")

    return Explanation(
        content=1.0,
        collaborative=0.0,
        popularity=0.0,
        reasons=tuple(reasons),
        matched_genres=genres,
        matched_terms=terms,
    )


def profile_explanation(
    *,
    score: float,
    anchor_title: str,
    anchor_rating: float | None = None,
    anchor_genres: Sequence[str] = (),
    anchor_terms: Sequence[str] = (),
    candidate_genres: Sequence[str] = (),
    candidate_terms: Sequence[str] = (),
    content_share: float = 1.0,
) -> Explanation:
    """Explain a recommendation derived from a user's ratings."""
    genres = shared_values(anchor_genres, candidate_genres, limit=3)
    terms = shared_terms(anchor_terms, candidate_terms, limit=MAX_CHIPS)

    stars = f" {anchor_rating:g}★" if anchor_rating is not None else ""
    reasons = [f"Because you rated {anchor_title}{stars}"]
    if genres:
        reasons.append(f"Same ground: {_join(list(genres))}")
    if terms:
        reasons.append(f"Similar themes: {_join(list(terms))}")

    return Explanation(
        content=max(0.0, min(1.0, content_share)),
        collaborative=0.0,
        popularity=0.0,
        reasons=tuple(reasons),
        matched_genres=genres,
        matched_terms=terms,
    )
