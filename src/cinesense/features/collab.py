"""Latent-factor collaborative filtering (offline training).

Standard biased matrix factorisation, computed with SVD rather than iterated
gradient descent:

    r_ui ~= mu + b_u + b_i + <x_u, p_i>

The biases absorb the fact that some users rate everything generously and some
films are universally liked. Only the *residual* is factorised, which is what
makes 64 factors enough.

Two decisions worth stating:

* **Item factors only are shipped.** User factors would be 200,948 x 64 (~51 MB)
  and still could not serve anyone outside MovieLens. Instead the runtime folds
  a user's ratings into a vector at request time: x_u = residual_u @ P. Same
  maths, 11 MB, and it works for arbitrary new users.
* **The evaluated model is the shipped model.** We hold out 10% of ratings
  before training and report metrics against it. Training a second "production"
  model on all data would look better and prove nothing.

Dependencies: pandas, numpy, scipy, scikit-learn. Offline only.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

from cinesense.config import ARTIFACTS_DIR, PROCESSED_DIR, Settings, load_settings

__all__ = [
    "CollabBuild",
    "build_collab",
    "load_item_axis",
    "prepare_ratings",
    "render_collab_report",
]

log = logging.getLogger(__name__)

COLLAB_DIR = ARTIFACTS_DIR / "collab"
FACTORS_FILE = "item_factors.npy"
BIAS_FILE = "item_bias.npy"
POPULARITY_FILE = "item_popularity.npy"
META_FILE = "meta.json"


class CollabError(RuntimeError):
    """Training could not produce a usable model."""


@dataclass(slots=True)
class PreparedRatings:
    """Ratings aligned to the catalogue's item axis, split into train/test."""

    users: np.ndarray          # user code per interaction
    items: np.ndarray          # catalogue row per interaction
    ratings: np.ndarray        # float32 rating
    train: np.ndarray          # boolean mask
    n_users: int
    n_items: int


@dataclass(slots=True)
class CollabBuild:
    directory: Path
    meta: dict
    stats: dict


def load_item_axis(settings: Settings) -> np.ndarray:
    """The catalogue's movie ids, which every artifact is aligned to.

    Content and collaborative factors must share row order or the hybrid blend
    would silently compare different films.
    """
    path = ARTIFACTS_DIR / "content" / "movie_ids.npy"
    if not path.exists():
        raise CollabError(
            f"{path} not found. Run `cinesense build-features` before `build-model`."
        )
    return np.load(path)


def prepare_ratings(settings: Settings, *, verbose: bool = True) -> PreparedRatings:
    """Load ratings, filter by the configured thresholds, and split."""
    catalogue_ids = load_item_axis(settings)
    item_index = {int(mid): row for row, mid in enumerate(catalogue_ids)}

    ratings_path = PROCESSED_DIR / settings.dataset.slug / "raw" / "ratings.parquet"
    if not ratings_path.exists():
        raise CollabError(f"{ratings_path} not found. Run `cinesense build-catalog`.")

    frame = pd.read_parquet(ratings_path, columns=["user_id", "movie_id", "rating"])
    if verbose:
        log.info("loaded %s ratings", f"{len(frame):,}")

    mapped = frame["movie_id"].map(item_index)
    frame = frame[mapped.notna()].copy()
    frame["item"] = mapped[mapped.notna()].astype(np.int32)
    frame.drop(columns=["movie_id"], inplace=True)

    # Thresholds are applied before the split so train and test describe the
    # same population. An item must have min_movie_ratings to be modelled at all.
    item_counts = frame["item"].value_counts()
    user_counts = frame["user_id"].value_counts()
    keep = (frame["item"].map(item_counts) >= settings.collab.min_movie_ratings) & (
        frame["user_id"].map(user_counts) >= settings.collab.min_user_ratings
    )
    frame = frame[keep]

    if verbose:
        log.info(
            "after thresholds (>=%d movie, >=%d user ratings): %s interactions, "
            "%s users, %s movies",
            settings.collab.min_movie_ratings,
            settings.collab.min_user_ratings,
            f"{len(frame):,}",
            f"{frame['user_id'].nunique():,}",
            f"{frame['item'].nunique():,}",
        )

    if frame.empty:
        raise CollabError("no interactions survive the configured thresholds")

    users, _ = pd.factorize(frame["user_id"])
    rng = np.random.default_rng(settings.collab.random_state)
    train_mask = rng.random(len(frame)) >= settings.collab.test_fraction

    return PreparedRatings(
        users=users.astype(np.int32),
        items=frame["item"].to_numpy(dtype=np.int32),
        ratings=frame["rating"].to_numpy(dtype=np.float32),
        train=train_mask,
        n_users=int(users.max()) + 1,
        n_items=len(catalogue_ids),
    )


def _fit_biases(
    users: np.ndarray,
    items: np.ndarray,
    ratings: np.ndarray,
    n_users: int,
    n_items: int,
    *,
    iterations: int,
    regularisation: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Alternating regularised bias fit.

    Ninety-nine percent of this matrix is empty, so an unregularised item bias on
    a film with three ratings would be wild. The regularisation term pulls each
    bias toward zero in proportion to how little evidence supports it.
    """
    mu = float(ratings.mean())
    user_bias = np.zeros(n_users, dtype=np.float32)
    item_bias = np.zeros(n_items, dtype=np.float32)

    user_count = np.bincount(users, minlength=n_users).astype(np.float32)
    item_count = np.bincount(items, minlength=n_items).astype(np.float32)

    for _ in range(iterations):
        residual = ratings - mu - user_bias[users]
        item_bias = (
            np.bincount(items, weights=residual, minlength=n_items).astype(np.float32)
            / (item_count + regularisation)
        ).astype(np.float32)

        residual = ratings - mu - item_bias[items]
        user_bias = (
            np.bincount(users, weights=residual, minlength=n_users).astype(np.float32)
            / (user_count + regularisation)
        ).astype(np.float32)

    return mu, user_bias, item_bias


def build_collab(settings: Settings | None = None, *, force: bool = False) -> CollabBuild:
    settings = settings or load_settings()
    started = time.perf_counter()

    COLLAB_DIR.mkdir(parents=True, exist_ok=True)
    meta_path = COLLAB_DIR / META_FILE

    if meta_path.exists() and not force:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        log.info("cached: collaborative model (rmse %.4f)", meta.get("rmse", float("nan")))
        return CollabBuild(directory=COLLAB_DIR, meta=meta, stats=meta)

    prepared = prepare_ratings(settings)
    train = prepared.train

    log.info("fitting biases (%d iterations) ...", settings.collab.bias_iterations)
    mu, user_bias, item_bias = _fit_biases(
        prepared.users[train],
        prepared.items[train],
        prepared.ratings[train],
        prepared.n_users,
        prepared.n_items,
        iterations=settings.collab.bias_iterations,
        regularisation=settings.collab.regularisation,
    )

    residual = (
        prepared.ratings[train]
        - mu
        - user_bias[prepared.users[train]]
        - item_bias[prepared.items[train]]
    ).astype(np.float32)

    matrix = sparse.csr_matrix(
        (residual, (prepared.users[train], prepared.items[train])),
        shape=(prepared.n_users, prepared.n_items),
        dtype=np.float32,
    )
    log.info("training matrix: %s users x %s items, %s nonzeros",
             f"{matrix.shape[0]:,}", f"{matrix.shape[1]:,}", f"{matrix.nnz:,}")

    log.info("factorising residuals into %d dimensions ...", settings.collab.factors)
    svd = TruncatedSVD(
        n_components=settings.collab.factors,
        algorithm="randomized",
        n_iter=6,
        random_state=settings.collab.random_state,
    )
    user_factors = svd.fit_transform(matrix).astype(np.float32)
    item_factors = svd.components_.T.astype(np.float32)
    explained = float(svd.explained_variance_ratio_.sum())

    log.info("evaluating on the held-out %d%% ...", int(settings.collab.test_fraction * 100))
    test = ~train
    rmse_model, rmse_baseline = _evaluate(
        prepared, test, mu, user_bias, item_bias, user_factors, item_factors
    )

    popularity = np.bincount(
        prepared.items[train], minlength=prepared.n_items
    ).astype(np.int32)

    np.save(COLLAB_DIR / FACTORS_FILE, item_factors.astype(np.float16))
    np.save(COLLAB_DIR / BIAS_FILE, item_bias)
    np.save(COLLAB_DIR / POPULARITY_FILE, popularity)

    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "factors": int(item_factors.shape[1]),
        "n_users": int(prepared.n_users),
        "n_items": int(prepared.n_items),
        "n_train": int(train.sum()),
        "n_test": int(test.sum()),
        "global_mean": round(mu, 6),
        "rmse": round(rmse_model, 4),
        "rmse_baseline": round(rmse_baseline, 4),
        "explained_variance": round(explained, 4),
        "wall_seconds": round(time.perf_counter() - started, 2),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    manifest = {
        "artifacts": [
            {
                "name": name,
                "bytes": (COLLAB_DIR / name).stat().st_size,
                "rows": int(prepared.n_items),
            }
            for name in (FACTORS_FILE, BIAS_FILE, POPULARITY_FILE)
        ]
    }
    meta["artifacts"] = manifest["artifacts"]
    meta["artifact_mb"] = round(
        sum(a["bytes"] for a in manifest["artifacts"]) / 1024**2, 2
    )

    return CollabBuild(directory=COLLAB_DIR, meta=meta, stats=meta)


def _evaluate(
    prepared: PreparedRatings,
    test_mask: np.ndarray,
    mu: float,
    user_bias: np.ndarray,
    item_bias: np.ndarray,
    user_factors: np.ndarray,
    item_factors: np.ndarray,
) -> tuple[float, float]:
    """RMSE of the full model, and of the bias-only baseline."""
    users = prepared.users[test_mask]
    items = prepared.items[test_mask]
    truth = prepared.ratings[test_mask]

    baseline = mu + user_bias[users] + item_bias[items]
    interaction = np.einsum(
        "ij,ij->i", user_factors[users], item_factors[items], optimize=True
    )
    prediction = np.clip(baseline + interaction, 0.5, 5.0)

    if prediction.size == 0:
        raise CollabError("held-out set is empty; lower test_fraction")

    rmse_model = float(np.sqrt(np.mean((prediction - truth) ** 2)))
    rmse_baseline = float(np.sqrt(np.mean((baseline - truth) ** 2)))
    return rmse_model, rmse_baseline


def render_collab_report(built: CollabBuild) -> str:
    m = built.meta
    width = 62

    def row(label: str, value: object, note: str = "") -> str:
        suffix = f"   {note}" if note else ""
        return f"  {label:<26}{value!s:>{width - 28}}{suffix}"

    improvement = 0.0
    if m["rmse_baseline"]:
        improvement = 1 - m["rmse"] / m["rmse_baseline"]

    return "\n".join(
        [
            "",
            "-" * width,
            "  CineSense collaborative model",
            "-" * width,
            row("users", f"{m['n_users']:,}"),
            row("items", f"{m['n_items']:,}"),
            row("train interactions", f"{m['n_train']:,}"),
            row("held-out", f"{m['n_test']:,}"),
            row("factors", m["factors"]),
            row("explained variance", f"{m['explained_variance']:.1%}"),
            "-" * width,
            row("rmse (bias baseline)", f"{m['rmse_baseline']:.4f}"),
            row("rmse (with factors)", f"{m['rmse']:.4f}", f"-{improvement:.1%}"),
            "-" * width,
            row("shipped artifacts", f"{m['artifact_mb']:.1f} MB", str(built.directory)),
            row("elapsed", f"{m['wall_seconds']:.1f}s"),
            "-" * width,
            "",
        ]
    )
