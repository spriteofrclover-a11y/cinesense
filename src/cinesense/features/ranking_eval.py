"""Ranking evaluation: does the hybrid beat its parts?

Accuracy (RMSE) says how well the model predicts a number. It says nothing about
whether the right films reach the top of a list, which is what a user sees. This
module measures NDCG@k and precision@k against a held-out set the model never
trained on.

All four variants run through the *same* recommender with different weights, so
the comparison holds everything else constant - candidate pool, eligibility,
tie-breaking.

MMR is disabled here (lambda = 1.0). Diversification deliberately trades
relevance for variety, so scoring a diversified list against a relevance metric
would understate it.

Dependencies: pandas, numpy. Offline only.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, replace
from typing import Mapping

import numpy as np
import pandas as pd

from cinesense.config import ARTIFACTS_DIR, HybridSettings, Settings
from cinesense.features.collab import prepare_ratings
from cinesense.recommend.collab import CollabModel
from cinesense.recommend.content import ContentIndex
from cinesense.recommend.hybrid import HybridRecommender

__all__ = ["RankingReport", "evaluate_ranking", "render_ranking_report"]

log = logging.getLogger(__name__)

CONTENT_DIR = ARTIFACTS_DIR / "content"
COLLAB_DIR = ARTIFACTS_DIR / "collab"

#: A held-out rating at or above this counts as a hit.
RELEVANT_RATING = 4.0


@dataclass(frozen=True, slots=True)
class RankingReport:
    k: int
    users: int
    ndcg: dict[str, float]
    precision: dict[str, float]

    def as_dict(self) -> dict:
        return asdict(self)


def _variants(settings: HybridSettings) -> dict[str, HybridSettings]:
    """The same ranker with one signal enabled at a time, plus the real blend."""
    return {
        "popularity": replace(
            settings, content_weight=0.0, collab_weight=0.0, popularity_weight=1.0, mmr_lambda=1.0
        ),
        "content": replace(
            settings, content_weight=1.0, collab_weight=0.0, popularity_weight=0.0, mmr_lambda=1.0
        ),
        "collaborative": replace(
            settings, content_weight=0.0, collab_weight=1.0, popularity_weight=0.0, mmr_lambda=1.0
        ),
        "hybrid": replace(settings, mmr_lambda=1.0),
    }


def _ndcg_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    hits = 0
    dcg = 0.0
    for position, movie_id in enumerate(recommended[:k], start=1):
        if movie_id in relevant:
            hits += 1
            dcg += 1.0 / np.log2(position + 1)

    ideal_hits = min(len(relevant), k)
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / np.log2(position + 1) for position in range(1, ideal_hits + 1))
    return dcg / idcg


def evaluate_ranking(
    settings: Settings,
    *,
    k: int = 10,
    users: int = 200,
    mmr_lambda: float = 1.0,
) -> RankingReport:
    prepared = prepare_ratings(settings, verbose=False)

    frame = pd.DataFrame(
        {
            "user": prepared.users,
            "item": prepared.items,
            "rating": prepared.ratings,
            "train": prepared.train,
        }
    )

    content = ContentIndex.load(CONTENT_DIR)
    collab = CollabModel.load(COLLAB_DIR)
    catalogue_ids = content.movie_ids

    # Only users with both a history and something held out can be judged.
    train_rows = frame[frame["train"]]
    test_rows = frame[~frame["train"]]

    positives = test_rows[test_rows["rating"] >= RELEVANT_RATING].groupby("user")["item"].agg(list)
    history = train_rows.groupby("user").agg({"item": list, "rating": list})

    eligible = positives.index.intersection(history.index)
    if len(eligible) == 0:
        raise ValueError("no user has both training history and held-out likes")

    rng = np.random.default_rng(settings.collab.random_state)
    if len(eligible) > users:
        eligible = rng.choice(eligible, size=users, replace=False)

    rankers = {
        name: HybridRecommender(content, collab, replace(variant, mmr_lambda=mmr_lambda))
        for name, variant in _variants(settings.hybrid).items()
    }

    ndcg: dict[str, list[float]] = {name: [] for name in rankers}
    precision: dict[str, list[float]] = {name: [] for name in rankers}

    for user in eligible:
        items = history.loc[user, "item"]
        ratings = history.loc[user, "rating"]
        history_map: Mapping[int, float] = {
            int(catalogue_ids[item]): float(rating)
            for item, rating in zip(items, ratings, strict=True)
        }
        relevant = {int(catalogue_ids[item]) for item in positives.loc[user]}

        for name, ranker in rankers.items():
            recs = ranker.recommend(history_map, k=k)
            recommended = [rec.movie_id for rec in recs]

            ndcg[name].append(_ndcg_at_k(recommended, relevant, k))
            precision[name].append(
                len(set(recommended[:k]) & relevant) / k
            )

    report = RankingReport(
        k=k,
        users=len(eligible),
        ndcg={name: round(float(np.mean(values)), 4) for name, values in ndcg.items()},
        precision={
            name: round(float(np.mean(values)), 4) for name, values in precision.items()
        },
    )
    _persist(report)
    return report


def _persist(report: RankingReport) -> None:
    """Write the result where the API can serve it to the About page."""
    import json
    from datetime import datetime, timezone

    directory = ARTIFACTS_DIR / "eval"
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        **report.as_dict(),
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (directory / "ranking.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    log.info("ranking metrics written to %s", directory / "ranking.json")


def render_ranking_report(report: RankingReport) -> str:
    width = 62

    def row(label: str, value: object, note: str = "") -> str:
        suffix = f"   {note}" if note else ""
        return f"  {label:<24}{value!s:>{width - 26}}{suffix}"

    lines = [
        "",
        "-" * width,
        f"  Ranking quality  (k={report.k}, users={report.users})",
        "-" * width,
        row("variant", "NDCG", "precision"),
    ]
    for name in report.ndcg:
        marker = "  <- shipped" if name == "hybrid" else ""
        lines.append(
            row(
                name,
                f"{report.ndcg[name]:.4f}",
                f"{report.precision[name]:.4f}{marker}",
            )
        )

    best_single = max(
        (report.ndcg[name] for name in report.ndcg if name != "hybrid"), default=0.0
    )
    gain = report.ndcg["hybrid"] - best_single
    lines += [
        "-" * width,
        row("hybrid vs best single", f"{gain:+.4f}", f"{gain / best_single:+.1%}" if best_single else ""),
        "-" * width,
        "",
    ]
    return "\n".join(lines)
