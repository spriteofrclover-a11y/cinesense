"""Collaborative inference: fold-in, scoring, and artifact alignment."""

from __future__ import annotations

import json

import numpy as np
import pytest

from cinesense.recommend.collab import CollabModel, FoldedUser

IDS = np.array([1, 2, 3, 4], dtype=np.int32)

FACTORS = np.array(
    [
        [1.00, 0.00],
        [0.90, 0.10],
        [0.00, 1.00],
        [0.10, 0.90],
    ],
    dtype=np.float32,
)

BIAS = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
POPULARITY = np.array([500, 400, 300, 20], dtype=np.int32)


@pytest.fixture
def model() -> CollabModel:
    return CollabModel(FACTORS, BIAS, POPULARITY, IDS, global_mean=3.5)


def test_length_and_dimensions(model):
    assert len(model) == 4
    assert model.factors_dim == 2
    assert model.global_mean == 3.5


def test_fold_in_returns_none_without_known_ratings(model):
    assert model.fold_in({}) is None
    assert model.fold_in({999: 5.0}) is None
    assert model.scores({999: 5.0}) is None


def test_fold_in_uses_residuals_against_item_bias(model):
    folded = model.fold_in({1: 5.0, 3: 2.0})

    assert isinstance(folded, FoldedUser)
    assert not folded.is_empty
    # +1.5 on item 1 (factors [1,0]) and -1.5 on item 3 (factors [0,1]).
    np.testing.assert_allclose(folded.factors, [1.5, -1.5], atol=1e-5)


def test_scores_rank_liked_items_higher(model):
    scores = model.scores({1: 5.0, 3: 2.0})

    assert scores is not None
    assert scores[0] > scores[1] > scores[2]
    assert pytest.approx(float(scores[0]), abs=0.3) == 5.0


def test_scores_are_clamped_to_the_rating_scale(model):
    extreme = CollabModel(FACTORS * 10, BIAS, POPULARITY, IDS, global_mean=3.5)
    scores = extreme.scores({1: 5.0, 3: 1.0})

    assert scores is not None
    assert scores.min() >= 0.5
    assert scores.max() <= 5.0


def test_single_rating_yields_no_factor_direction(model):
    """One rating cannot establish a taste direction once the user bias is removed."""
    folded = model.fold_in({1: 5.0})

    assert folded is not None
    assert pytest.approx(float(np.abs(folded.factors).sum()), abs=1e-6) == 0.0


def test_user_bias_absorbs_generous_raters(model):
    """Someone who rates everything 5 is not evidence that every film is good."""
    folded = model.fold_in({1: 5.0, 2: 5.0, 3: 5.0})

    assert folded is not None
    assert folded.bias == pytest.approx(1.5, abs=1e-5)
    assert pytest.approx(float(np.abs(folded.factors).sum()), abs=1e-5) == 0.0


def test_similar_items_uses_factor_direction(model):
    results = model.similar_items(1, k=2)

    assert results[0].movie_id == 2
    assert results[0].score > results[1].score


def test_similar_items_excludes_self(model):
    assert 1 not in [rec.movie_id for rec in model.similar_items(1, k=4)]


def test_misaligned_artifacts_are_rejected():
    with pytest.raises(ValueError, match="misaligned"):
        CollabModel(FACTORS, BIAS[:2], POPULARITY, IDS)


def test_zero_norm_factors_do_not_divide_by_zero():
    zeros = np.zeros_like(FACTORS)
    flat = CollabModel(zeros, BIAS, POPULARITY, IDS)
    assert flat.similar_items(1, k=2) == []


def test_save_and_load_roundtrip(tmp_path):
    content = tmp_path / "content"
    collab = tmp_path / "collab"
    content.mkdir()
    collab.mkdir()

    np.save(content / "movie_ids.npy", IDS)
    np.save(collab / "item_factors.npy", FACTORS.astype(np.float16))
    np.save(collab / "item_bias.npy", BIAS)
    np.save(collab / "item_popularity.npy", POPULARITY)
    (collab / "meta.json").write_text(
        json.dumps({"global_mean": 3.5, "rmse": 0.9, "factors": 2}), encoding="utf-8"
    )

    restored = CollabModel.load(collab)

    assert len(restored) == 4
    assert restored.global_mean == 3.5
    # float16 storage costs a little precision: cos([1,0], [0.9,0.1]) = 0.994.
    assert restored.similar_items(1, k=1)[0].score == pytest.approx(0.9939, abs=1e-3)


def test_load_requires_the_shared_item_axis(tmp_path):
    collab = tmp_path / "collab"
    collab.mkdir()
    (collab / "meta.json").write_text(json.dumps({"global_mean": 3.5}), encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="item axis"):
        CollabModel.load(collab)
