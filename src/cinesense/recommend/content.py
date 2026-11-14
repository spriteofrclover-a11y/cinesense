"""Content-based recommendations at runtime.

Loads one float16 latent matrix plus a compact catalogue, then answers two
questions with plain numpy:

* "more like this" - cosine between a movie's vector and every other movie.
* "more like my taste" - fold a user's ratings into a vector, then cosine.

There is deliberately **no precomputed neighbour table**. Scoring 87,585 movies
against a 96-dimension vector is ~8M multiply-adds, a few milliseconds, and it
costs 17 MB of artifact instead of ~35 MB of stored neighbours. Storing
neighbours would also go stale the moment the feature matrix is rebuilt.

Dependencies: numpy and the standard library only.
"""

from __future__ import annotations

import gzip
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from cinesense.recommend.contracts import Recommendation
from cinesense.recommend.explain import content_explanation, profile_explanation

__all__ = ["ContentIndex", "MovieMeta"]

log = logging.getLogger(__name__)

VECTORS_FILE = "vectors.npy"
IDS_FILE = "movie_ids.npy"
CATALOG_FILE = "catalog.json.gz"

#: Ratings at or above this are treated as genuine likes when choosing which
#: movie to cite as the reason for a recommendation.
ANCHOR_MIN_RATING = 4.0

#: Ratings at or below this carry no positive weight in the taste profile.
#:
#: An earlier version mean-centred the ratings, so a 1-star rating *subtracted*
#: that film's vector. Disliking "The Matrix" therefore pushed the profile away
#: from Sci-Fi and Action - which is almost the entire catalogue - and surfaced
#: obscure Crime/Drama instead. A dislike means "do not show me this", not
#: "recommend the opposite", so it is expressed as an exclusion.
NEUTRAL_RATING = 2.5

_NORMALISE_RE = re.compile(r"[^0-9a-z]+")

#: MovieLens inverts titles with a leading article: "Godfather, The (1972)".
#: Must run on the raw title - normalising first strips the comma the pattern
#: depends on.
_ARTICLE_RE = re.compile(r"^(.*),\s*(the|a|an)\s*$", re.IGNORECASE)


def _normalise(text: str) -> str:
    return " ".join(_NORMALISE_RE.sub(" ", text.lower()).split())


def _title_key(text: str) -> str:
    raw = text.strip()
    match = _ARTICLE_RE.match(raw)
    if match:
        raw = f"{match.group(2)} {match.group(1)}"
    return _normalise(raw)


_LEADING_ARTICLE_RE = re.compile(r"^(the|a|an)\s+")

#: Everything that is not a letter or digit, for the "no separator" form.
_COMPACT_RE = re.compile(r"[^a-z0-9]+")


def _compact_key(key: str) -> str:
    """Title with every separator removed: "k.g.f: chapter 1" -> "kgfchapter1"."""
    return _COMPACT_RE.sub("", key)


def _core_key(key: str) -> str:
    """Title with any leading article removed.

    Without this, searching "godfather" ranks the 1991 film "GodFather" above
    "The Godfather" (1972): the latter is stored as "Godfather, The", becomes
    "the godfather", and only matched as a substring.
    """
    stripped = _LEADING_ARTICLE_RE.sub("", key)
    return stripped or key


def _l2(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else vector


@dataclass(frozen=True, slots=True)
class MovieMeta:
    movie_id: int
    title: str
    year: int | None
    genres: tuple[str, ...]
    #: How many ratings the film has. Used only to break exact ties and to pick
    #: the better-known match in search - never to make a film more similar.
    n_ratings: int = 0
    #: Mean rating across all users, 0.0 when unrated.
    avg_rating: float = 0.0

    @property
    def display(self) -> str:
        return f"{self.title} ({self.year})" if self.year else self.title


class ContentIndex:
    """An immutable, in-memory content model."""

    def __init__(
        self,
        movie_ids: np.ndarray,
        vectors: np.ndarray,
        metadata: Mapping[int, MovieMeta],
        terms: Mapping[int, tuple[str, ...]],
    ) -> None:
        if len(movie_ids) != vectors.shape[0]:
            raise ValueError(
                f"{len(movie_ids)} ids but {vectors.shape[0]} vectors"
            )
        self._ids = np.asarray(movie_ids, dtype=np.int32)
        self._vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self._meta = dict(metadata)
        self._terms = dict(terms)
        self._row_of = {int(mid): row for row, mid in enumerate(self._ids)}
        self._popularity = np.array(
            [self._meta.get(int(mid), MovieMeta(0, "", None, ())).n_ratings for mid in self._ids],
            dtype=np.int32,
        )

    # -- construction ------------------------------------------------------ #

    @classmethod
    def load(cls, directory: Path | str) -> ContentIndex:
        directory = Path(directory)
        vectors = np.load(directory / VECTORS_FILE).astype(np.float32)
        ids = np.load(directory / IDS_FILE)

        with gzip.open(directory / CATALOG_FILE, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)

        metadata: dict[int, MovieMeta] = {}
        terms: dict[int, tuple[str, ...]] = {}
        ratings = payload.get("ratings") or [0] * len(payload["ids"])
        averages = payload.get("avg") or [0.0] * len(payload["ids"])
        for mid, title, year, genres, movie_terms, n_ratings, avg in zip(
            payload["ids"],
            payload["titles"],
            payload["years"],
            payload["genres"],
            payload["terms"],
            ratings,
            averages,
            strict=True,
        ):
            metadata[mid] = MovieMeta(
                movie_id=mid,
                title=title,
                year=year,
                genres=tuple(genres),
                n_ratings=int(n_ratings),
                avg_rating=float(avg or 0.0),
            )
            terms[mid] = tuple(movie_terms)

        log.info(
            "content index loaded: %d movies, %d dims",
            len(ids),
            vectors.shape[1],
        )
        return cls(ids, vectors, metadata, terms)

    @classmethod
    def save(
        cls,
        directory: Path | str,
        movie_ids: np.ndarray,
        vectors: np.ndarray,
        metadata: Sequence[MovieMeta],
        terms: Sequence[Sequence[str]],
    ) -> None:
        """Persist artifacts. Vectors are stored float16 to halve the size."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        np.save(directory / VECTORS_FILE, np.asarray(vectors, dtype=np.float16))
        np.save(directory / IDS_FILE, np.asarray(movie_ids, dtype=np.int32))

        payload = {
            "ids": [meta.movie_id for meta in metadata],
            "titles": [meta.title for meta in metadata],
            "years": [meta.year for meta in metadata],
            "genres": [list(meta.genres) for meta in metadata],
            "terms": [list(term) for term in terms],
            "ratings": [meta.n_ratings for meta in metadata],
            "avg": [round(meta.avg_rating, 3) for meta in metadata],
        }
        with gzip.open(directory / CATALOG_FILE, "wt", encoding="utf-8", compresslevel=6) as handle:
            json.dump(payload, handle, separators=(",", ":"), ensure_ascii=False)

    # -- introspection ------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self._ids)

    def __contains__(self, movie_id: object) -> bool:
        return movie_id in self._row_of

    @property
    def dimensions(self) -> int:
        return int(self._vectors.shape[1])

    @property
    def movie_ids(self) -> np.ndarray:
        return self._ids

    @property
    def vectors(self) -> np.ndarray:
        """Row-aligned latent matrix. Read-only by convention."""
        return self._vectors

    def row_of(self, movie_id: int) -> int | None:
        """Catalogue row for a movie id, or ``None`` if it is not indexed."""
        return self._row_of.get(movie_id)

    def meta(self, movie_id: int) -> MovieMeta | None:
        return self._meta.get(movie_id)

    def terms(self, movie_id: int) -> tuple[str, ...]:
        return self._terms.get(movie_id, ())

    def vector(self, movie_id: int) -> np.ndarray | None:
        row = self._row_of.get(movie_id)
        return None if row is None else self._vectors[row]

    def _search_index(self) -> tuple[list, list[str], list[str], list[str]]:
        """Title keys built once, on first search.

        87k titles: rebuilding these per keystroke would be wasteful, and the
        catalogue never changes after load.
        """
        cached = getattr(self, "_search_cache", None)
        if cached is not None:
            return cached

        metas = list(self._meta.values())
        keys = [_title_key(meta.title) for meta in metas]
        cores = [_core_key(key) for key in keys]
        # Separators removed entirely, so "kgf" matches "k.g.f: chapter 1".
        compacts = [_compact_key(key) for key in keys]

        self._search_cache = (metas, keys, cores, compacts)
        return self._search_cache

    def search(self, query: str, limit: int = 10) -> list[MovieMeta]:
        """Substring search over canonicalised titles, best matches first."""
        needle = _title_key(query)
        if not needle:
            return []

        needle_core = _core_key(needle)
        needle_compact = _compact_key(needle)

        metas, keys, cores, compacts = self._search_index()

        exact: list[MovieMeta] = []
        prefix: list[MovieMeta] = []
        contains: list[MovieMeta] = []

        for meta, haystack, hay_core, hay_compact in zip(metas, keys, cores, compacts):
            # Literal and article-insensitive matches share one tier on purpose.
            # Separating them fixed "Master" (2021) but broke "The Godfather"
            # (1972), because the article form is sometimes the famous film and
            # sometimes an obscure one. Within the tier, the better-known wins.
            if haystack == needle or hay_core == needle_core:
                exact.append(meta)
            elif (
                haystack.startswith(needle)
                or hay_core.startswith(needle_core)
                or (needle_compact and hay_compact.startswith(needle_compact))
            ):
                prefix.append(meta)
            elif needle in haystack:
                contains.append(meta)

        key = lambda meta: (-meta.n_ratings, _recency(meta))  # noqa: E731
        ordered = (
            sorted(exact, key=key)
            + sorted(prefix, key=key)
            + sorted(contains, key=key)
        )
        return ordered[:limit]

    # -- scoring ------------------------------------------------------------ #

    def _top_rows(self, scores: np.ndarray, k: int) -> np.ndarray:
        """Indices of the k highest scores, descending.

        ``argpartition`` keeps this O(n) per query; a full sort of 87k items per
        request would be wasted work.

        Exact ties are broken by rating count. Without this, films that share a
        single genre and no tags - documentaries especially - all score 1.000
        and win their place by array position, which looks arbitrary.
        """
        n = scores.size
        if k >= n:
            pool = np.arange(n)
        else:
            threshold = np.partition(scores, n - k)[n - k]
            # Keep every item tied at the cut-off, so the tie-break decides
            # among all equals rather than a sample of them.
            pool = np.flatnonzero(scores >= threshold)

        order = np.lexsort((-self._popularity[pool], -scores[pool]))
        return pool[order][:k]

    def similar_to(
        self,
        movie_id: int,
        k: int = 10,
        *,
        exclude: Iterable[int] | None = None,
    ) -> list[Recommendation]:
        """Movies closest to ``movie_id`` by content similarity."""
        row = self._row_of.get(movie_id)
        if row is None:
            return []

        scores = self._vectors @ self._vectors[row]
        scores[row] = -np.inf
        for other in exclude or ():
            other_row = self._row_of.get(other)
            if other_row is not None:
                scores[other_row] = -np.inf

        query = self._meta.get(movie_id)
        if query is None:
            return []

        results: list[Recommendation] = []
        for index in self._top_rows(scores, k):
            if not np.isfinite(scores[index]):
                continue
            candidate = self._meta[int(self._ids[index])]
            results.append(
                Recommendation(
                    movie_id=candidate.movie_id,
                    title=candidate.title,
                    year=candidate.year,
                    genres=candidate.genres,
                    score=float(scores[index]),
                    explanation=content_explanation(
                        score=float(scores[index]),
                        query_title=query.title,
                        query_genres=query.genres,
                        query_terms=self.terms(movie_id),
                        candidate_genres=candidate.genres,
                        candidate_terms=self.terms(candidate.movie_id),
                        query_year=query.year,
                        candidate_year=candidate.year,
                    ),
                )
            )
        return results

    def batch_similar(
        self, movie_ids: Sequence[int], k: int
    ) -> list[list[tuple[int, float]]]:
        """Top-k content neighbours for many queries at once.

        One matrix product beats N separate ones: 500 queries score against
        87,585 movies in a single sgemm instead of 500 passes over the matrix.
        """
        rows = [self._row_of[mid] for mid in movie_ids if mid in self._row_of]
        if not rows:
            return []

        scores = self._vectors[rows] @ self._vectors.T
        for position, row in enumerate(rows):
            scores[position, row] = -np.inf

        output: list[list[tuple[int, float]]] = []
        for position in range(len(rows)):
            indices = self._top_rows(scores[position], k)
            output.append(
                [
                    (int(self._ids[index]), float(scores[position, index]))
                    for index in indices
                    if np.isfinite(scores[position, index])
                ]
            )
        return output

    def profile(self, ratings: Mapping[int, float]) -> np.ndarray | None:
        """Fold a rating history into a single content vector.

        Likes are weighted by how far above neutral they sit; ratings at or
        below ``NEUTRAL_RATING`` contribute nothing. See the comment on
        ``NEUTRAL_RATING`` for why dislikes are not subtracted.
        """
        known = {
            mid: float(rating)
            for mid, rating in ratings.items()
            if mid in self._row_of and np.isfinite(rating)
        }
        if not known:
            return None

        weights = {mid: max(0.0, rating - NEUTRAL_RATING) for mid, rating in known.items()}

        if sum(weights.values()) <= 0.0:
            # Everything was disliked or neutral. Fall back to the highest-rated
            # films rather than returning nothing.
            best = max(known.values())
            weights = {mid: 1.0 for mid, rating in known.items() if rating == best}

        accumulator = np.zeros(self.dimensions, dtype=np.float32)
        for mid, weight in weights.items():
            if weight > 0:
                accumulator += weight * self._vectors[self._row_of[mid]]

        return _l2(accumulator)

    def recommend(
        self,
        ratings: Mapping[int, float],
        k: int = 10,
        *,
        exclude: Iterable[int] | None = None,
    ) -> list[Recommendation]:
        """Recommendations from a rating history (cold start included)."""
        vector = self.profile(ratings)
        if vector is None:
            return []

        scores = self._vectors @ vector

        seen = set(ratings)
        seen.update(exclude or ())
        for mid in seen:
            row = self._row_of.get(mid)
            if row is not None:
                scores[row] = -np.inf

        anchors = self._anchor_rows(ratings)

        results: list[Recommendation] = []
        for index in self._top_rows(scores, k):
            if not np.isfinite(scores[index]):
                continue
            candidate = self._meta[int(self._ids[index])]
            explanation = self._explain_against_anchors(
                candidate, scores[index], anchors
            )
            results.append(
                Recommendation(
                    movie_id=candidate.movie_id,
                    title=candidate.title,
                    year=candidate.year,
                    genres=candidate.genres,
                    score=float(scores[index]),
                    explanation=explanation,
                )
            )
        return results

    # -- explanation helpers ------------------------------------------------ #

    def _anchor_rows(self, ratings: Mapping[int, float]) -> list[tuple[int, float, int]]:
        """(movie_id, rating, row) for movies worth citing as a reason."""
        liked = [
            (mid, float(rating))
            for mid, rating in ratings.items()
            if mid in self._row_of
            and np.isfinite(rating)
            and float(rating) >= ANCHOR_MIN_RATING
        ]
        if not liked:
            liked = [
                (mid, float(rating))
                for mid, rating in ratings.items()
                if mid in self._row_of
            ]
        return [(mid, rating, self._row_of[mid]) for mid, rating in liked]

    def _explain_against_anchors(
        self,
        candidate: MovieMeta,
        score: float,
        anchors: list[tuple[int, float, int]],
    ):
        from cinesense.recommend.contracts import Explanation

        if not anchors:
            return Explanation(content=1.0, reasons=("Popular overall",))

        candidate_row = self._row_of[candidate.movie_id]
        anchor_matrix = self._vectors[[row for _, _, row in anchors]]
        similarities = anchor_matrix @ self._vectors[candidate_row]
        best = int(np.argmax(similarities))

        anchor_id, anchor_rating, _ = anchors[best]
        anchor = self._meta.get(anchor_id)
        if anchor is None:
            return Explanation(content=1.0)

        magnitude = float(np.clip(score, 0.0, 1.0))
        return profile_explanation(
            score=score,
            anchor_title=anchor.title,
            anchor_rating=anchor_rating,
            anchor_genres=anchor.genres,
            anchor_terms=self.terms(anchor_id),
            candidate_genres=candidate.genres,
            candidate_terms=self.terms(candidate.movie_id),
            content_share=magnitude,
        )


def _recency(meta: MovieMeta) -> tuple[int, str]:
    return (-(meta.year or 0), meta.title)
