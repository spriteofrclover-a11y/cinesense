"""The data contract every recommender speaks.

A recommendation is never a bare movie id. It carries its own justification,
because "why did you suggest this" is a product requirement, not a debug log.
Designing that in now means the hybrid ranker in Phase 3 produces explanations
for free instead of having them bolted on at the API layer.

This module is deliberately dependency-free (no numpy, no pandas): it is
imported by the serving process, which must stay lightweight.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

__all__ = ["Explanation", "Recommendation", "ScoredMovie"]


@dataclass(frozen=True, slots=True)
class Explanation:
    """Why a movie appeared, in both numbers and words.

    The numeric fields are the normalised contributions of each signal and
    should sum to roughly 1.0 for a blended result. ``reasons`` is what a human
    reads; ``matched_*`` is what the UI renders as chips.
    """

    content: float = 0.0
    collaborative: float = 0.0
    popularity: float = 0.0
    reasons: tuple[str, ...] = ()
    matched_genres: tuple[str, ...] = ()
    matched_terms: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("content", "collaborative", "popularity"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} contribution must be in [0, 1], got {value}")

    @property
    def dominant(self) -> str:
        """Which signal drove this recommendation."""
        scores = {
            "content": self.content,
            "collaborative": self.collaborative,
            "popularity": self.popularity,
        }
        return max(scores, key=lambda k: scores[k])

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Recommendation:
    """One ranked suggestion, ready to serialise to JSON."""

    movie_id: int
    title: str
    year: int | None
    genres: tuple[str, ...]
    score: float
    explanation: Explanation = field(default_factory=Explanation)

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["genres"] = list(self.genres)
        return payload


@dataclass(frozen=True, slots=True)
class ScoredMovie:
    """An intermediate: a movie id and a raw similarity, no metadata yet.

    Rankers return these so they can work purely in numpy over id/index arrays;
    the catalogue lookup and explanation happen once, at the boundary.
    """

    movie_id: int
    score: float

    def __post_init__(self) -> None:
        if self.score != self.score:  # NaN check without importing math
            raise ValueError("score must not be NaN")


def contribution_weights(
    raw: Mapping[str, float], total: float | None = None
) -> dict[str, float]:
    """Normalise raw signal strengths into [0, 1] contributions.

    Guards the ``Explanation`` invariant: a signal that contributed nothing must
    report 0.0, not a division artefact.
    """
    cleaned = {key: max(0.0, float(value)) for key, value in raw.items()}
    denominator = total if total is not None else sum(cleaned.values())
    if denominator <= 0:
        return {key: 0.0 for key in cleaned}
    return {key: min(1.0, value / denominator) for key, value in cleaned.items()}
