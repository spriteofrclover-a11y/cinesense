"""Measure whether MMR fixes the intra-list duplication found in Phase 2.

Phase 2's pure content model returned lists with mean pairwise cosine 0.94 -
ten slots, one film. This compares MMR settings on the shipped hybrid hurt.
"""

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cinesense.config import ARTIFACTS_DIR, PROCESSED_DIR, load_settings  # noqa: E402
from cinesense.recommend.collab import CollabModel  # noqa: E402
from cinesense.recommend.content import ContentIndex  # noqa: E402
from cinesense.recommend.hybrid import HybridRecommender  # noqa: E402

settings = load_settings()
content = ContentIndex.load(ARTIFACTS_DIR / "content")
collab = CollabModel.load(ARTIFACTS_DIR / "collab")

ratings = pd.read_parquet(
    PROCESSED_DIR / "ml-32m" / "raw" / "ratings.parquet",
    columns=["user_id", "movie_id", "rating"],
)
users = ratings.groupby("user_id").size()
eligible = users[users >= 20].index
rng = np.random.default_rng(11)
sample = rng.choice(eligible, size=120, replace=False)

frames = []
for user in sample:
    rows = ratings[ratings["user_id"] == user]
    frames.append({int(r.movie_id): float(r.rating) for r in rows.itertuples(index=False)})

print(f"{'mmr_lambda':>12}{'diversity':>12}{'unique genres':>16}{'latency ms':>12}")
print("-" * 54)

import time

for lam in (1.0, 0.85, 0.75, 0.5, 0.25):
    ranker = HybridRecommender(content, collab, replace(settings.hybrid, mmr_lambda=lam))
    diversities = []
    genres = []
    start = time.perf_counter()

    for history in frames:
        recs = ranker.recommend(history, k=10)
        vectors = [content.vector(rec.movie_id) for rec in recs]
        vectors = [v for v in vectors if v is not None]
        if len(vectors) >= 2:
            stacked = np.vstack(vectors)
            sims = stacked @ stacked.T
            upper = sims[np.triu_indices_from(sims, k=1)]
            diversities.append(1.0 - upper.mean())
        genres.append(len({g for rec in recs for g in rec.genres}))

    elapsed = (time.perf_counter() - start) / len(frames) * 1000
    print(f"{lam:>12.2f}{np.mean(diversities):>11.1%}{np.mean(genres):>16.1f}{elapsed:>12.1f}")

print()
print("Diversity = 1 - mean pairwise cosine. Phase 2 (content only) was 6.4%.")
