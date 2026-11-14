"""The hybrid ranker: three signals, one list, with diversification.

Design notes:

* **Signals are z-scored before blending.** Content similarity lives in [-1, 1],
  a collaborative prediction lives in [0.5, 5], and popularity is a count.
  Adding them raw would mean whichever has the biggest numbers wins regardless
  of the configured weights.
* **MMR runs after relevance, not instead of it.** Relevance picks a candidate
  pool of 200, then maximal marginal relevance chooses the final k, penalising
  picks that duplicate something already chosen. Phase 2 measured intra-list
  diversity at 6.4% without it.
* **Cold start is not a special path.** A new user supplies favourite movies,
  which become pseudo-ratings, and everything downstream runs unchanged. When
  there are no ratings at all, the collaborative signal is simply absent and the
  weights renormalise.

Dependencies: numpy and the standard library only.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

from cinesense.config import HybridSettings
from cinesense.recommend.collab import CollabModel
from cinesense.recommend.content import ContentIndex
from cinesense.recommend.contracts import Explanation, Recommendation, contribution_weights
from cinesense.recommend.explain import shared_terms, shared_values

__all__ = ["HybridRecommender", "PreferenceProfile"]

log = logging.getLogger(__name__)

#: A 5-star given to an onboarding pick. High enough to dominate, not so high it
#: implies a real viewing history.
PSEUDO_RATING = 5.0


@dataclass(frozen=True, slots=True)
class PreferenceProfile:
    """What we know about a user before they have rated anything."""

    favorite_movie_ids: tuple[int, ...] = ()
    genres: tuple[str, ...] = ()
    min_year: int | None = None
    max_year: int | None = None
    #: ISO 639-1 language codes, e.g. ("hi", "ta"). Matched against TMDB data.
    languages: tuple[str, ...] = ()
    #: ISO 3166-1 country codes, e.g. ("IN",). Matched against TMDB data.
    regions: tuple[str, ...] = ()

    def pseudo_ratings(self) -> dict[int, float]:
        return {int(mid): PSEUDO_RATING for mid in self.favorite_movie_ids}

    @property
    def is_empty(self) -> bool:
        return not (
            self.favorite_movie_ids
            or self.genres
            or self.min_year
            or self.max_year
            or self.languages
            or self.regions
        )


class HybridRecommender:
    """Blends content, collaborative and popularity signals, then diversifies."""

    def __init__(
        self,
        content: ContentIndex,
        collab: CollabModel | None,
        settings: HybridSettings,
        *,
        language_of: Callable[[int], str | None] | None = None,
        countries_of: Callable[[int], list[str]] | None = None,
    ) -> None:
        if len(content) != (len(collab) if collab is not None else len(content)):
            raise ValueError("content and collaborative models disagree on item count")
        self.content = content
        self.collab = collab
        self.settings = settings

        movie_ids = content.movie_ids

        # Cached once: an 87k Python loop per request would be wasteful.
        self._n_ratings = np.array(
            [
                (content.meta(int(mid)).n_ratings if content.meta(int(mid)) else 0)
                for mid in movie_ids
            ],
            dtype=np.int32,
        )
        self._languages = (
            np.array([language_of(int(mid)) or "" for mid in movie_ids], dtype=object)
            if language_of is not None
            else None
        )
        self._countries = (
            np.array([tuple(countries_of(int(mid))) for mid in movie_ids], dtype=object)
            if countries_of is not None
            else None
        )

    # -- public ------------------------------------------------------------- #

    def recommend(
        self,
        ratings: Mapping[int, float] | None = None,
        k: int = 10,
        *,
        prefs: PreferenceProfile | None = None,
        exclude: Iterable[int] = (),
    ) -> list[Recommendation]:
        ratings = dict(ratings or {})
        if prefs is not None and not ratings:
            ratings = prefs.pseudo_ratings()

        eligible = self._eligible_mask(prefs)
        seen = set(ratings)
        seen.update(exclude)
        for movie_id in seen:
            row = self.content.row_of(movie_id)
            if row is not None:
                eligible[row] = False

        if not eligible.any():
            return []

        signals: dict[str, np.ndarray] = {}
        weights: dict[str, float] = {}

        content_scores = self._content_scores(ratings, prefs)
        if content_scores is not None:
            signals["content"] = content_scores
            weights["content"] = self.settings.content_weight

        collab_scores = None
        if ratings and self.collab is not None:
            collab_scores = self.collab.scores(ratings)
            if collab_scores is not None:
                signals["collaborative"] = collab_scores
                weights["collaborative"] = self.settings.collab_weight

        popularity = self._popularity_scores()
        if popularity is not None and self.settings.popularity_weight > 0:
            signals["popularity"] = popularity
            weights["popularity"] = self.settings.popularity_weight

        if not signals:
            return []

        blended = self._blend(signals, weights, eligible)
        candidates = self._candidate_pool(blended, eligible)
        chosen = self._mmr(candidates, blended, k)

        return [
            self._to_recommendation(row, blended, signals, ratings, prefs)
            for row in chosen
        ]

    # -- signals ------------------------------------------------------------ #

    def _eligible_mask(self, prefs: PreferenceProfile | None) -> np.ndarray:
        """Films that may be recommended at all."""
        mask = np.ones(len(self.content), dtype=bool)

        # One rating is enough to be recommendable. This used to use the
        # collaborative *training* counts, which only include films with 30+
        # ratings - so every film below that threshold was excluded outright
        # rather than merely ranked low. Most non-Western cinema sat there:
        # 636 India-tagged films existed, 158 were reachable.
        mask &= self._n_ratings > 0

        if prefs is None:
            return mask

        movie_ids = self.content.movie_ids
        metadata = [self.content.meta(int(mid)) for mid in movie_ids]

        if prefs.genres:
            wanted = {genre.lower() for genre in prefs.genres}
            for row, meta in enumerate(metadata):
                if not meta or not wanted.intersection(g.lower() for g in meta.genres):
                    mask[row] = False

        if prefs.min_year is not None or prefs.max_year is not None:
            for row, meta in enumerate(metadata):
                year = meta.year if meta else None
                if year is None:
                    mask[row] = False
                    continue
                if prefs.min_year is not None and year < prefs.min_year:
                    mask[row] = False
                if prefs.max_year is not None and year > prefs.max_year:
                    mask[row] = False

        if prefs.languages or prefs.regions:
            mask &= self._matches_language(prefs)

        return mask

    def _matches_language(self, prefs: PreferenceProfile) -> np.ndarray:
        """Films in a wanted language, or produced in a wanted country."""
        wanted_languages = {code.lower() for code in prefs.languages}
        wanted_regions = {code.upper() for code in prefs.regions}
        hits = np.zeros(len(self.content), dtype=bool)

        for row in range(len(self.content)):
            if self._languages is not None and self._languages[row] in wanted_languages:
                hits[row] = True
            elif self._countries is not None and wanted_regions.intersection(self._countries[row]):
                hits[row] = True
        return hits

    def _content_scores(
        self, ratings: Mapping[int, float], prefs: PreferenceProfile | None
    ) -> np.ndarray | None:
        profile = self.content.profile(ratings) if ratings else None
        if profile is None:
            return None
        return (self.content.vectors @ profile).astype(np.float32)

    def _popularity_scores(self) -> np.ndarray | None:
        """How widely rated a film is, across every rating rather than the
        training subset - sparse films deserve a small weight, not zero."""
        if self._n_ratings.size == 0:
            return None
        return np.log1p(self._n_ratings.astype(np.float32))

    @staticmethod
    def _zscore(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Standardise over the eligible set so weights are comparable."""
        selected = values[mask]
        if selected.size == 0:
            return np.zeros_like(values)

        mean = float(selected.mean())
        std = float(selected.std())
        if std < 1e-6:
            return np.zeros_like(values)
        return ((values - mean) / std).astype(np.float32)

    def _blend(
        self,
        signals: dict[str, np.ndarray],
        weights: dict[str, float],
        eligible: np.ndarray,
    ) -> np.ndarray:
        total = np.zeros(len(self.content), dtype=np.float32)
        for name, values in signals.items():
            total += weights.get(name, 1.0) * self._zscore(values, eligible)

        total[~eligible] = -np.inf
        return total

    # -- selection ---------------------------------------------------------- #

    def _candidate_pool(self, blended: np.ndarray, eligible: np.ndarray) -> np.ndarray:
        size = min(int(self.settings.candidate_pool), int(eligible.sum()))
        if size <= 0:
            return np.empty(0, dtype=np.int64)

        threshold = np.partition(blended, len(blended) - size)[len(blended) - size]
        pool = np.flatnonzero(blended >= threshold)
        return pool[np.argsort(-blended[pool])][:size]

    def _mmr(self, pool: np.ndarray, relevance: np.ndarray, k: int) -> list[int]:
        """Maximal marginal relevance.

        Pure relevance repeats itself: Phase 2's top results had a mean pairwise
        cosine of 0.94, so ten slots showed one film ten times. MMR subtracts a
        penalty for being close to something already picked.
        """
        if pool.size == 0:
            return []

        lam = float(np.clip(self.settings.mmr_lambda, 0.0, 1.0))
        limited = min(k, pool.size)

        if lam >= 1.0:
            return [int(row) for row in pool[:limited]]

        # Rescale relevance to [0, 1] so lambda has a stable meaning.
        low, high = float(relevance[pool].min()), float(relevance[pool].max())
        span = high - low
        normalised = (relevance - low) / span if span > 1e-9 else np.ones_like(relevance)

        chosen: list[int] = []
        remaining = pool.tolist()
        vectors = self.content.vectors

        while len(chosen) < limited and remaining:
            if not chosen:
                scores = normalised[remaining]
            else:
                similarity = vectors[remaining] @ vectors[chosen].T
                redundancy = similarity.max(axis=1)
                scores = lam * normalised[remaining] - (1.0 - lam) * redundancy

            best = int(np.argmax(scores))
            chosen.append(remaining.pop(best))

        return chosen

    # -- output ------------------------------------------------------------- #

    def _to_recommendation(
        self,
        row: int,
        blended: np.ndarray,
        signals: dict[str, np.ndarray],
        ratings: Mapping[int, float],
        prefs: PreferenceProfile | None,
    ) -> Recommendation:
        movie_id = int(self.content.movie_ids[row])
        meta = self.content.meta(movie_id)

        if meta is None:
            raise ValueError(f"no metadata for item row {row}")

        raw = {
            name: max(0.0, float(weights_value))
            for name, weights_value in self._contributions(signals, row).items()
        }
        parts = contribution_weights(raw)

        reasons, genres, terms = self._explain(meta, ratings, prefs, parts)

        return Recommendation(
            movie_id=movie_id,
            title=meta.title,
            year=meta.year,
            genres=meta.genres,
            score=float(blended[row]),
            explanation=Explanation(
                content=parts.get("content", 0.0),
                collaborative=parts.get("collaborative", 0.0),
                popularity=parts.get("popularity", 0.0),
                reasons=reasons,
                matched_genres=genres,
                matched_terms=terms,
            ),
        )

    def _contributions(self, signals: dict[str, np.ndarray], row: int) -> dict[str, float]:
        """Signed signal strength for one item, in the blend's own units."""
        values: dict[str, float] = {}
        for name, signal in signals.items():
            weight = {
                "content": self.settings.content_weight,
                "collaborative": self.settings.collab_weight,
                "popularity": self.settings.popularity_weight,
            }.get(name, 1.0)
            values[name] = float(weight * signal[row])
        return values

    def _explain(
        self,
        meta,
        ratings: Mapping[int, float],
        prefs: PreferenceProfile | None,
        parts: Mapping[str, float],
    ) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
        reasons: list[str] = []
        matched_genres: tuple[str, ...] = ()
        matched_terms: tuple[str, ...] = ()

        anchor = self._best_anchor(meta.movie_id, ratings)
        if anchor is not None:
            anchor_id, anchor_rating = anchor
            anchor_meta = self.content.meta(anchor_id)
            if anchor_meta is not None:
                stars = f" {anchor_rating:g}★" if anchor_rating else ""
                reasons.append(f"Because you rated {anchor_meta.title}{stars}")
                matched_genres = shared_values(
                    anchor_meta.genres, meta.genres, limit=3
                )
                matched_terms = shared_terms(
                    self.content.terms(anchor_id), self.content.terms(meta.movie_id)
                )
                if matched_genres:
                    reasons.append("Same ground: " + ", ".join(matched_genres))

        if parts.get("collaborative", 0.0) >= 0.45 and anchor is None:
            reasons.append("Viewers with similar taste rated this highly")

        if prefs is not None and not ratings:
            if prefs.genres:
                overlap = shared_values(prefs.genres, meta.genres, limit=3)
                if overlap:
                    matched_genres = overlap
                    reasons.append("Matches your preferred genres: " + ", ".join(overlap))
            if prefs.min_year or prefs.max_year:
                reasons.append("Fits the era you asked for")

        if parts.get("popularity", 0.0) >= 0.5:
            reasons.append("Widely rated and well liked")

        if not reasons:
            reasons.append("Strong all-round match")

        return tuple(reasons), matched_genres, matched_terms

    def _best_anchor(
        self, movie_id: int, ratings: Mapping[int, float]
    ) -> tuple[int, float] | None:
        """The rated film most similar to this one, weighted by how it was rated.

        Only genuinely liked films qualify: citing a 1-star rating as a
        recommendation reason is nonsense.
        """
        from cinesense.recommend.content import ANCHOR_MIN_RATING

        liked = [
            (mid, float(rating))
            for mid, rating in ratings.items()
            if float(rating) >= ANCHOR_MIN_RATING and mid in self.content
        ]
        if not liked:
            return None

        target = self.content.vector(movie_id)
        if target is None:
            return None

        best: tuple[int, float] | None = None
        best_score = -np.inf
        for anchor_id, rating in liked:
            vector = self.content.vector(anchor_id)
            if vector is None:
                continue
            score = float(vector @ target)
            if score > best_score:
                best_score = score
                best = (anchor_id, rating)
        return best
