"""Offline evaluation of the content model.

There is no ground truth for "similar" beyond human judgement, so we measure a
proxy that is honest about what it captures: do the returned neighbours share
genres with the query? It cannot detect "this is the better film", only "this is
the same kind of film" - which is exactly what a content-based model claims to
provide, and it is compared against a random baseline so the number means
something.

The same metric is used in ``scripts/measure_genome.py``, so genome and
non-genome variants stay directly comparable.

Dependencies: numpy only.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np

from cinesense.recommend.content import ContentIndex

__all__ = ["EvalReport", "evaluate", "genre_jaccard_at_k"]

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class EvalReport:
    """Measured quality of a content model."""

    queries: int
    k: int
    genre_agreement: float
    random_baseline: float
    lift: float
    catalogue_coverage: float
    intra_list_diversity: float

    def as_dict(self) -> dict:
        return asdict(self)


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def genre_jaccard_at_k(
    index: ContentIndex,
    query_ids: Iterable[int],
    *,
    k: int = 10,
) -> float:
    """Mean Jaccard overlap between each query's genres and its neighbours'."""
    queries = [
        mid for mid in query_ids if index.meta(mid) and index.meta(mid).genres
    ]
    if not queries:
        return 0.0

    genre_sets = {mid: set(index.meta(mid).genres) for mid in queries}
    neighbours = index.batch_similar(queries, k)

    scores: list[float] = []
    for mid, rows in zip(queries, neighbours, strict=True):
        base = genre_sets[mid]
        for neighbour_id, _ in rows:
            meta = index.meta(neighbour_id)
            if meta is None:
                continue
            scores.append(_jaccard(base, set(meta.genres)))

    return float(np.mean(scores)) if scores else 0.0


def _random_baseline(
    index: ContentIndex,
    genre_sets: dict[int, set[str]],
    *,
    k: int,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    all_ids = index.movie_ids
    scores: list[float] = []
    for genres in genre_sets.values():
        picks = rng.choice(all_ids, size=min(k, len(all_ids)), replace=False)
        for movie_id in picks:
            meta = index.meta(int(movie_id))
            if meta is not None:
                scores.append(_jaccard(genres, set(meta.genres)))
    return float(np.mean(scores)) if scores else 0.0


def _diversity(index: ContentIndex, neighbours: list[list[tuple[int, float]]]) -> float:
    """1 - mean cosine between distinct pairs inside each result list."""
    values: list[float] = []
    for rows in neighbours:
        vectors = [index.vector(mid) for mid, _ in rows]
        vectors = [v for v in vectors if v is not None]
        if len(vectors) < 2:
            continue
        stacked = np.vstack(vectors)
        similarities = stacked @ stacked.T
        upper = similarities[np.triu_indices_from(similarities, k=1)]
        if upper.size:
            values.append(float(1.0 - upper.mean()))
    return float(np.mean(values)) if values else 0.0


def evaluate(
    index: ContentIndex,
    *,
    k: int = 10,
    sample: int = 500,
    seed: int = 42,
) -> EvalReport:
    """Score the content model against a genre-agreement baseline."""
    eligible = [mid for mid in index.movie_ids if index.meta(int(mid)) and index.meta(int(mid)).genres]
    if not eligible:
        raise ValueError("no movies with genres; cannot evaluate")

    rng = np.random.default_rng(seed)
    size = min(sample, len(eligible))
    query_ids = [int(mid) for mid in rng.choice(eligible, size=size, replace=False)]

    genre_sets = {mid: set(index.meta(mid).genres) for mid in query_ids}

    neighbours = index.batch_similar(query_ids, k)

    scores: list[float] = []
    recommended: set[int] = set()
    for mid, rows in zip(query_ids, neighbours, strict=True):
        base = genre_sets[mid]
        for neighbour_id, _ in rows:
            recommended.add(neighbour_id)
            meta = index.meta(neighbour_id)
            if meta is not None:
                scores.append(_jaccard(base, set(meta.genres)))

    agreement = float(np.mean(scores)) if scores else 0.0
    baseline = _random_baseline(index, genre_sets, k=k, seed=seed)
    coverage = len(recommended) / len(index) if len(index) else 0.0

    return EvalReport(
        queries=len(query_ids),
        k=k,
        genre_agreement=round(agreement, 4),
        random_baseline=round(baseline, 4),
        lift=round(agreement / baseline, 2) if baseline > 0 else 0.0,
        catalogue_coverage=round(coverage, 4),
        intra_list_diversity=round(_diversity(index, neighbours), 4),
    )


def render_eval_report(report: EvalReport) -> str:
    width = 62

    def row(label: str, value: object, note: str = "") -> str:
        suffix = f"   {note}" if note else ""
        return f"  {label:<26}{value!s:>{width - 28}}{suffix}"

    return "\n".join(
        [
            "",
            "-" * width,
            f"  Content model evaluation  (k={report.k}, n={report.queries})",
            "-" * width,
            row("genre agreement@k", f"{report.genre_agreement:.4f}", "higher is better"),
            row("random baseline", f"{report.random_baseline:.4f}"),
            row("lift over random", f"{report.lift:.2f}x"),
            "-" * width,
            row("catalogue coverage", f"{report.catalogue_coverage:.1%}", "distinct items returned"),
            row("intra-list diversity", f"{report.intra_list_diversity:.1%}", "higher = less repetitive"),
            "-" * width,
            "",
        ]
    )
