"""Collaborative inference at request time.

No user factors are stored. A user's taste vector is folded in from whatever
ratings we have for them:

    residual_i = r_i - mu - b_i
    b_u        = mean(residual_i)
    x_u        = sum (residual_i - b_u) * p_i

That is the same algebra the SVD was trained with, applied to one user instead
of 200,948, and it costs a handful of vector additions. It also means a brand
new user with three ratings gets a real collaborative estimate, which a stored
user matrix could never provide.

Dependencies: numpy and the standard library only.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from cinesense.recommend.contracts import Recommendation

__all__ = ["CollabModel", "FoldedUser"]

log = logging.getLogger(__name__)

FACTORS_FILE = "item_factors.npy"
BIAS_FILE = "item_bias.npy"
POPULARITY_FILE = "item_popularity.npy"
META_FILE = "meta.json"


@dataclass(frozen=True, slots=True)
class FoldedUser:
    """A user's taste vector and bias, derived from their ratings."""

    factors: np.ndarray
    bias: float
    rated: tuple[int, ...]

    @property
    def is_empty(self) -> bool:
        return not self.rated


class CollabModel:
    """Item factors, item biases and popularity, aligned to the catalogue axis."""

    def __init__(
        self,
        item_factors: np.ndarray,
        item_bias: np.ndarray,
        popularity: np.ndarray,
        item_ids: np.ndarray,
        *,
        global_mean: float = 3.5,
    ) -> None:
        if not (len(item_ids) == item_factors.shape[0] == len(item_bias) == len(popularity)):
            raise ValueError(
                "collaborative artifacts are misaligned: "
                f"{len(item_ids)} ids, {item_factors.shape[0]} factor rows, "
                f"{len(item_bias)} biases, {len(popularity)} popularity values"
            )
        self._ids = np.asarray(item_ids, dtype=np.int32)
        self._factors = np.ascontiguousarray(item_factors, dtype=np.float32)
        self._bias = np.asarray(item_bias, dtype=np.float32)
        self._popularity = np.asarray(popularity, dtype=np.int32)
        self._mean = float(global_mean)
        self._row_of = {int(mid): row for row, mid in enumerate(self._ids)}

    # -- construction ------------------------------------------------------ #

    @classmethod
    def load(cls, directory: Path | str) -> CollabModel:
        directory = Path(directory)
        meta = json.loads((directory / META_FILE).read_text(encoding="utf-8"))

        # The item axis must come from the same array the factors were built
        # against; the content artifacts are the single source of truth.
        ids_path = directory.parent / "content" / "movie_ids.npy"
        if not ids_path.exists():
            raise FileNotFoundError(
                f"{ids_path} not found. Content and collaborative models must "
                "share one item axis."
            )
        item_ids = np.load(ids_path)

        model = cls(
            np.load(directory / FACTORS_FILE).astype(np.float32),
            np.load(directory / BIAS_FILE),
            np.load(directory / POPULARITY_FILE),
            item_ids,
            global_mean=meta.get("global_mean", 3.5),
        )
        log.info(
            "collab model loaded: %d items, %d factors, rmse %.3f",
            len(item_ids),
            model.factors_dim,
            meta.get("rmse", float("nan")),
        )
        return model

    # -- introspection ------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self._ids)

    @property
    def factors_dim(self) -> int:
        return int(self._factors.shape[1])

    @property
    def global_mean(self) -> float:
        return self._mean

    @property
    def popularity(self) -> np.ndarray:
        return self._popularity

    def row(self, movie_id: int) -> int | None:
        return self._row_of.get(movie_id)

    # -- inference ---------------------------------------------------------- #

    def fold_in(self, ratings: Mapping[int, float]) -> FoldedUser | None:
        """Derive a taste vector from a rating history."""
        rows: list[int] = []
        values: list[float] = []
        for movie_id, rating in ratings.items():
            row = self._row_of.get(movie_id)
            if row is None or not np.isfinite(rating):
                continue
            rows.append(row)
            values.append(float(rating))

        if not rows:
            return None

        rows_array = np.asarray(rows, dtype=np.int64)
        residual = np.asarray(values, dtype=np.float32) - self._mean - self._bias[rows_array]

        # The user's own generosity: someone who rates everything 5 is not
        # evidence that every film is good.
        bias = float(residual.mean())
        centred = residual - bias

        factors = np.zeros(self.factors_dim, dtype=np.float32)
        factors += centred @ self._factors[rows_array]

        return FoldedUser(factors=factors, bias=bias, rated=tuple(ratings))

    def scores(self, ratings: Mapping[int, float]) -> np.ndarray | None:
        """Predicted preference for every item, or None when there is no history."""
        folded = self.fold_in(ratings)
        if folded is None:
            return None

        scores = (
            self._mean
            + self._bias
            + folded.bias
            + (self._factors @ folded.factors)
        )
        return np.clip(scores, 0.5, 5.0).astype(np.float32)

    def similar_items(self, movie_id: int, k: int = 10) -> list[Recommendation]:
        """Item-item similarity straight from the factors.

        Two items are similar when their latent factors point the same way, which
        is exactly item-based collaborative filtering - without materialising an
        87,585 x 87,585 matrix.
        """
        row = self._row_of.get(movie_id)
        if row is None:
            return []

        norms = np.linalg.norm(self._factors, axis=1)
        denominator = norms * norms[row]
        safe = np.where(denominator > 0, denominator, 1.0)
        similarity = (self._factors @ self._factors[row]) / safe
        similarity[row] = -np.inf

        top = np.argpartition(-similarity, min(k, len(similarity) - 1))[:k]
        top = top[np.argsort(-similarity[top])]

        return [
            Recommendation(
                movie_id=int(self._ids[index]),
                title="",
                year=None,
                genres=(),
                score=float(similarity[index]),
            )
            for index in top
            # A zero or negative cosine means "not similar at all". Returning
            # those would pad the list with unrelated films that merely happen
            # to sort first when a factor row is all zeros.
            if np.isfinite(similarity[index]) and similarity[index] > 0
        ]
