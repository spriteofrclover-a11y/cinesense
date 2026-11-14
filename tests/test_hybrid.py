"""The hybrid ranker: blending, cold start, filters and diversification."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from cinesense.config import HybridSettings
from cinesense.recommend.collab import CollabModel
from cinesense.recommend.content import ContentIndex, MovieMeta
from cinesense.recommend.hybrid import HybridRecommender, PreferenceProfile

IDS = np.array([1, 2, 3, 4, 5, 6], dtype=np.int32)

#: 1-3 are near-identical, 4-5 are near-identical, 6 sits between.
VECTORS = np.array(
    [
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.7071, 0.7071],
    ],
    dtype=np.float32,
)

META = {
    1: MovieMeta(1, "Alpha", 2001, ("Sci-Fi",), 500),
    2: MovieMeta(2, "Beta", 2002, ("Sci-Fi",), 400),
    3: MovieMeta(3, "Gamma", 2003, ("Sci-Fi",), 300),
    4: MovieMeta(4, "Delta", 2004, ("Horror",), 200),
    5: MovieMeta(5, "Epsilon", 2005, ("Horror",), 100),
    6: MovieMeta(6, "Zeta", 2006, ("Drama",), 50),
}

POPULARITY = np.array([500, 400, 300, 200, 100, 50], dtype=np.int32)
ITEM_BIAS = np.zeros(6, dtype=np.float32)


@pytest.fixture
def content() -> ContentIndex:
    terms = {mid: (f"tags:{meta.title.lower()}",) for mid, meta in META.items()}
    return ContentIndex(IDS, VECTORS, META, terms)


@pytest.fixture
def collab() -> CollabModel:
    return CollabModel(VECTORS, ITEM_BIAS, POPULARITY, IDS, global_mean=3.5)


@pytest.fixture
def hybrid(content, collab) -> HybridRecommender:
    return HybridRecommender(content, collab, HybridSettings())


def test_recommends_from_explicit_ratings(hybrid):
    results = hybrid.recommend({1: 5.0}, k=3)

    assert len(results) == 3
    assert all(rec.movie_id != 1 for rec in results)


def test_rated_movies_are_never_returned(hybrid):
    results = hybrid.recommend({1: 5.0, 2: 4.0}, k=6)
    assert {rec.movie_id for rec in results}.isdisjoint({1, 2})


def test_exclusions_are_honoured(hybrid):
    results = hybrid.recommend({1: 5.0}, k=4, exclude=[2])
    assert 2 not in {rec.movie_id for rec in results}


def test_every_recommendation_carries_reasons_and_contributions(hybrid):
    for rec in hybrid.recommend({1: 5.0}, k=3):
        assert rec.explanation.reasons
        assert 0.0 <= rec.explanation.content <= 1.0
        assert 0.0 <= rec.explanation.collaborative <= 1.0
        assert 0.0 <= rec.explanation.popularity <= 1.0


def test_cold_start_uses_favourites_and_says_so(hybrid):
    prefs = PreferenceProfile(favorite_movie_ids=(1,))
    results = hybrid.recommend(prefs=prefs, k=3)

    assert results
    assert any("Alpha" in reason for reason in results[0].explanation.reasons)


def test_preference_profile_produces_pseudo_ratings():
    prefs = PreferenceProfile(favorite_movie_ids=(1, 2))
    assert set(prefs.pseudo_ratings()) == {1, 2}
    assert PreferenceProfile().is_empty


def test_empty_recommender_falls_back_to_popularity(hybrid):
    """With no signals at all, do not return an empty page."""
    results = hybrid.recommend(k=3)
    assert len(results) == 3
    assert results[0].movie_id == 1  # the most-rated film leads


def test_genre_filter_restricts_the_candidate_set(hybrid):
    prefs = PreferenceProfile(favorite_movie_ids=(1,), genres=("Horror",))
    results = hybrid.recommend(prefs=prefs, k=5)

    assert results
    assert all("Horror" in rec.genres for rec in results)


def test_year_filter_restricts_the_candidate_set(hybrid):
    prefs = PreferenceProfile(min_year=2004, max_year=2005)
    results = hybrid.recommend(prefs=prefs, k=5)

    assert results
    assert all(rec.year in (2004, 2005) for rec in results)


def test_impossible_filters_return_nothing(hybrid):
    prefs = PreferenceProfile(genres=("Documentary",))
    assert hybrid.recommend(prefs=prefs, k=5) == []


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    return float(left @ right)


def test_mmr_trades_relevance_for_variety(content, collab):
    """Without MMR the top three are all the same film in different costumes."""
    pure = HybridRecommender(content, collab, replace(HybridSettings(), mmr_lambda=1.0))
    varied = HybridRecommender(content, collab, replace(HybridSettings(), mmr_lambda=0.3))

    pure_ids = [rec.movie_id for rec in pure.recommend({1: 5.0}, k=3)]
    varied_ids = [rec.movie_id for rec in varied.recommend({1: 5.0}, k=3)]

    def max_pairwise(ids):
        vectors = [VECTORS[mid - 1] for mid in ids]
        return max(
            _cosine(vectors[i], vectors[j])
            for i in range(len(vectors))
            for j in range(i + 1, len(vectors))
        )

    assert max_pairwise(pure_ids) == pytest.approx(1.0)
    assert max_pairwise(varied_ids) < 1.0


def test_popularity_only_ranks_by_popularity(content, collab):
    settings = replace(
        HybridSettings(),
        content_weight=0.0,
        collab_weight=0.0,
        popularity_weight=1.0,
        # MMR would deliberately skip the identical items 2 and 3, which is the
        # right behaviour but not what this test is about.
        mmr_lambda=1.0,
    )
    ranker = HybridRecommender(content, collab, settings)

    assert [rec.movie_id for rec in ranker.recommend(k=3)] == [1, 2, 3]


def test_content_only_ranks_by_similarity(content, collab):
    settings = replace(
        HybridSettings(), content_weight=1.0, collab_weight=0.0, popularity_weight=0.0
    )
    ranker = HybridRecommender(content, collab, settings)

    results = ranker.recommend({1: 5.0}, k=3)
    assert results[0].movie_id in (2, 3, 6)


def test_works_without_a_collaborative_model(content):
    ranker = HybridRecommender(content, None, HybridSettings())
    results = ranker.recommend({1: 5.0}, k=2)

    assert results
    assert all(rec.explanation.collaborative == 0.0 for rec in results)


def test_disliked_movies_are_never_cited_as_reasons(hybrid):
    results = hybrid.recommend({1: 1.0, 4: 5.0}, k=3)

    for rec in results:
        assert not any("Alpha" in reason for reason in rec.explanation.reasons)


def test_mismatched_models_are_rejected(content):
    short = CollabModel(VECTORS[:3], ITEM_BIAS[:3], POPULARITY[:3], IDS[:3])
    with pytest.raises(ValueError, match="disagree on item count"):
        HybridRecommender(content, short, HybridSettings())


# --------------------------------------------------------------------------- #
# Eligibility and language filtering
# --------------------------------------------------------------------------- #


def _index(ratings_by_id: dict[int, int]) -> ContentIndex:
    meta = {
        mid: MovieMeta(
            mid,
            META[mid].title,
            META[mid].year,
            META[mid].genres,
            ratings_by_id.get(mid, 0),
        )
        for mid in IDS
    }
    return ContentIndex(IDS, VECTORS, meta, {mid: () for mid in IDS})


def test_films_below_the_training_threshold_stay_eligible():
    """A film with 5 ratings has no collaborative weight but is still showable.

    Eligibility used to come from the collaborative training counts, which only
    include films with 30+ ratings, so anything below that was excluded outright
    rather than merely ranked low.
    """
    index = _index({1: 500, 2: 5, 3: 300, 4: 200, 5: 100, 6: 50})
    # Movie 2 is absent from the training matrix, so its collaborative popularity is 0.
    collab = CollabModel(
        VECTORS, ITEM_BIAS, np.array([500, 0, 300, 200, 100, 50], dtype=np.int32), IDS,
        global_mean=3.5,
    )
    ranker = HybridRecommender(
        index,
        collab,
        HybridSettings(),
        language_of=lambda mid: "hi" if mid == 2 else "en",
        countries_of=lambda mid: ["IN"] if mid == 2 else ["US"],
    )

    results = ranker.recommend(prefs=PreferenceProfile(languages=("hi",)), k=5)

    assert [rec.movie_id for rec in results] == [2]


def test_films_with_no_ratings_at_all_are_never_eligible():
    index = _index({1: 500, 2: 0, 3: 300, 4: 200, 5: 100, 6: 50})
    ranker = HybridRecommender(
        index,
        None,
        HybridSettings(),
        language_of=lambda mid: "hi" if mid == 2 else "en",
        countries_of=lambda mid: ["IN"] if mid == 2 else ["US"],
    )

    assert ranker.recommend(prefs=PreferenceProfile(languages=("hi",)), k=5) == []


def test_language_and_region_filters_select_the_right_films():
    index = _index({mid: 100 for mid in IDS})
    ranker = HybridRecommender(
        index,
        None,
        HybridSettings(),
        language_of=lambda mid: {1: "hi", 2: "ta", 3: "ml"}.get(mid, "en"),
        countries_of=lambda mid: ["IN"] if mid <= 3 else ["US"],
    )

    by_language = [rec.movie_id for rec in ranker.recommend(prefs=PreferenceProfile(languages=("ta",)), k=6)]
    by_region = [rec.movie_id for rec in ranker.recommend(prefs=PreferenceProfile(regions=("IN",)), k=6)]

    assert by_language == [2]
    assert sorted(by_region) == [1, 2, 3]


def test_language_filter_combines_with_genre():
    index = _index({mid: 100 for mid in IDS})
    ranker = HybridRecommender(
        index,
        None,
        HybridSettings(),
        language_of=lambda mid: "hi" if mid == 4 else "en",
        countries_of=lambda mid: ["IN"] if mid == 4 else ["US"],
    )

    prefs = PreferenceProfile(languages=("hi",), genres=("Sci-Fi",))

    # Movie 4 is Horror, so the two filters intersect to nothing.
    assert ranker.recommend(prefs=prefs, k=6) == []
