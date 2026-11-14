"""Evaluation metrics. A metric that cannot distinguish good from bad is worse
than none, so these tests check the metric actually responds to quality.
"""

from __future__ import annotations

import numpy as np
import pytest

from cinesense.recommend.content import ContentIndex, MovieMeta
from cinesense.recommend.evaluate import evaluate, genre_jaccard_at_k, render_eval_report

#: Two clean clusters. Sci-Fi movies point one way, Romance movies another, so a
#: correct model must never cross clusters.
VECTORS = np.array(
    [
        [1.0, 0.0],
        [0.98, 0.02],
        [0.96, 0.04],
        [0.94, 0.06],
        [0.0, 1.0],
        [0.02, 0.98],
        [0.04, 0.96],
        [0.06, 0.94],
    ],
    dtype=np.float32,
)

IDS = np.arange(1, 9, dtype=np.int32)

META = {
    1: MovieMeta(1, "Sci-Fi One", 2000, ("Sci-Fi",)),
    2: MovieMeta(2, "Sci-Fi Two", 2001, ("Sci-Fi",)),
    3: MovieMeta(3, "Sci-Fi Three", 2002, ("Sci-Fi",)),
    4: MovieMeta(4, "Sci-Fi Four", 2003, ("Sci-Fi",)),
    5: MovieMeta(5, "Romance One", 2004, ("Romance",)),
    6: MovieMeta(6, "Romance Two", 2005, ("Romance",)),
    7: MovieMeta(7, "Romance Three", 2006, ("Romance",)),
    8: MovieMeta(8, "Romance Four", 2007, ("Romance",)),
}


@pytest.fixture
def index() -> ContentIndex:
    return ContentIndex(IDS, VECTORS, META, {mid: () for mid in IDS})


def test_genre_agreement_is_perfect_on_separated_clusters(index):
    """Neighbours of a Sci-Fi film are all Sci-Fi, so Jaccard is 1.0."""
    score = genre_jaccard_at_k(index, [1, 5], k=3)
    assert score == pytest.approx(1.0, abs=1e-6)


def test_evaluate_beats_random_by_a_wide_margin(index):
    report = evaluate(index, k=3, sample=8, seed=1)

    assert report.queries == 8
    assert report.genre_agreement > 0.9
    assert report.random_baseline < 0.6
    assert report.lift > 1.5


def test_evaluate_reports_coverage_and_diversity(index):
    report = evaluate(index, k=3, sample=8, seed=1)

    assert 0.0 < report.catalogue_coverage <= 1.0
    # Identical within-cluster vectors mean low diversity by construction.
    assert 0.0 <= report.intra_list_diversity <= 1.0


def test_evaluate_rejects_a_catalogue_without_genres():
    meta = {1: MovieMeta(1, "Untagged", 2000, ())}
    empty = ContentIndex(np.array([1], dtype=np.int32), np.ones((1, 2), np.float32), meta, {})

    with pytest.raises(ValueError, match="no movies with genres"):
        evaluate(empty)


def test_agreement_is_zero_when_nothing_matches():
    meta = {
        1: MovieMeta(1, "A", 2000, ("Sci-Fi",)),
        2: MovieMeta(2, "B", 2001, ("Romance",)),
    }
    index = ContentIndex(
        np.array([1, 2], dtype=np.int32),
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        meta,
        {1: (), 2: ()},
    )

    assert genre_jaccard_at_k(index, [1], k=1) == 0.0


def test_report_renders_readable_text(index):
    text = render_eval_report(evaluate(index, k=3, sample=4, seed=1))

    assert "genre agreement" in text
    assert "lift over random" in text
    assert "catalogue coverage" in text
