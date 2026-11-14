"""Ranking metrics. A metric that cannot recognise a perfect ranking is useless."""

from __future__ import annotations

import pytest

from cinesense.config import HybridSettings
from cinesense.features.ranking_eval import _ndcg_at_k, _variants


def test_perfect_ranking_scores_one():
    assert _ndcg_at_k([1, 2, 3], {1, 2, 3}, k=3) == pytest.approx(1.0)


def test_no_hits_scores_zero():
    assert _ndcg_at_k([7, 8, 9], {1, 2, 3}, k=3) == 0.0


def test_ranking_above_the_cutoff_does_not_count():
    assert _ndcg_at_k([4, 5, 1, 2], {1, 2}, k=2) == 0.0


def test_higher_positions_are_worth_more():
    early = _ndcg_at_k([1, 9, 9], {1}, k=3)
    late = _ndcg_at_k([9, 9, 1], {1}, k=3)

    assert early > late > 0.0


def test_empty_relevant_set_is_zero_not_an_error():
    assert _ndcg_at_k([1, 2], set(), k=2) == 0.0


def test_ndcg_uses_the_ideal_gain_for_few_positives():
    """One relevant item ranked first is a perfect result, even for k=10."""
    assert _ndcg_at_k([1], {1}, k=10) == pytest.approx(1.0)


def test_variants_isolate_one_signal_each():
    variants = _variants(HybridSettings())

    assert variants["popularity"].content_weight == 0.0
    assert variants["popularity"].collab_weight == 0.0
    assert variants["content"].collab_weight == 0.0
    assert variants["content"].popularity_weight == 0.0
    assert variants["collaborative"].content_weight == 0.0
    assert variants["collaborative"].popularity_weight == 0.0


def test_variants_disable_diversification():
    """Diversification trades relevance for variety; ranking eval must not mix it in."""
    for variant in _variants(HybridSettings()).values():
        assert variant.mmr_lambda == 1.0


def test_hybrid_variant_keeps_the_configured_weights():
    configured = HybridSettings(collab_weight=2.5)
    assert _variants(configured)["hybrid"].collab_weight == 2.5
