"""The runtime content index: loading, similarity, fold-in, explanations."""

from __future__ import annotations

import numpy as np
import pytest

from cinesense.recommend.content import ContentIndex, MovieMeta

IDS = np.array([1, 2, 3, 4], dtype=np.int32)

VECTORS = np.array(
    [
        [1.0, 0.0, 0.0],        # 1 space adventure
        [0.9939, 0.1104, 0.0],  # 2 space sequel (near-duplicate)
        [0.0, 1.0, 0.0],        # 3 romantic comedy
        [0.0, 0.2, 0.9798],     # 4 crime drama
    ],
    dtype=np.float32,
)

META = {
    1: MovieMeta(1, "Space Adventure", 1999, ("Sci-Fi", "Adventure")),
    2: MovieMeta(2, "Space Sequel", 2003, ("Sci-Fi",)),
    3: MovieMeta(3, "Romantic Comedy", 2001, ("Romance", "Comedy")),
    4: MovieMeta(4, "Crime Drama", 2010, ("Crime", "Drama")),
}

TERMS = {
    1: ("tags:space", "genres:adventure"),
    2: ("tags:space", "title:sequel"),
    3: ("tags:love",),
    4: ("tags:crime",),
}


@pytest.fixture
def index() -> ContentIndex:
    return ContentIndex(IDS, VECTORS, META, TERMS)


def test_length_and_containment(index):
    assert len(index) == 4
    assert 1 in index
    assert 99 not in index
    assert index.dimensions == 3


def test_meta_and_terms_lookup(index):
    assert index.meta(2).title == "Space Sequel"
    assert index.meta(99) is None
    assert index.terms(1)[0] == "tags:space"
    assert index.terms(99) == ()


def test_display_includes_year():
    assert MovieMeta(1, "Arrival", 2016, ()).display == "Arrival (2016)"
    assert MovieMeta(1, "Unknown", None, ()).display == "Unknown"


def test_similar_to_ranks_nearest_first(index):
    results = index.similar_to(1, k=2)

    assert len(results) == 2
    assert results[0].movie_id == 2
    assert results[0].score > results[1].score
    assert results[1].score == pytest.approx(0.0, abs=1e-6)


def test_similar_to_never_returns_the_query(index):
    ids = [rec.movie_id for rec in index.similar_to(1, k=4)]
    assert 1 not in ids


def test_similar_to_honours_exclusions(index):
    ids = [rec.movie_id for rec in index.similar_to(1, k=3, exclude=[2])]
    assert 2 not in ids


def test_similar_to_carries_an_explanation(index):
    rec = index.similar_to(1, k=1)[0]
    assert rec.explanation.reasons
    assert "Space Adventure" in rec.explanation.reasons[0]
    assert "Sci-Fi" in rec.explanation.matched_genres


def test_similar_to_unknown_movie_returns_empty(index):
    assert index.similar_to(999) == []


def test_profile_centres_ratings(index):
    profile = index.profile({1: 5.0, 2: 1.0})

    assert profile is not None
    # Movie 1 liked, movie 2 disliked -> vector should point away from movie 2.
    similarity = index.vector(1) @ profile - index.vector(2) @ profile
    assert similarity > 0


def test_profile_falls_back_when_every_rating_is_equal(index):
    """All deviations zero would otherwise produce a useless zero vector."""
    profile = index.profile({3: 4.0})
    assert profile is not None
    assert np.linalg.norm(profile) == pytest.approx(1.0)
    assert index.vector(3) @ profile == pytest.approx(1.0)


def test_profile_returns_none_when_nothing_is_known(index):
    assert index.profile({}) is None
    assert index.profile({999: 5.0}) is None


def test_recommend_excludes_already_rated(index):
    results = index.recommend({3: 5.0}, k=4)
    ids = {rec.movie_id for rec in results}
    assert 3 not in ids
    assert results[0].movie_id == 4


def test_recommend_cites_a_liked_anchor(index):
    rec = index.recommend({3: 5.0}, k=1)[0]
    assert "Romantic Comedy" in rec.explanation.reasons[0]
    assert "5★" in rec.explanation.reasons[0]


def test_recommend_ignores_disliked_movies_as_anchors(index):
    """A 1-star rating must never be cited as a reason to recommend something."""
    results = index.recommend({3: 5.0, 1: 1.0}, k=2)
    for rec in results:
        assert "Space Adventure" not in rec.explanation.reasons[0]


def test_batch_similar_matches_individual_calls(index):
    batch = index.batch_similar([1, 3], k=2)

    assert len(batch) == 2
    assert batch[0][0][0] == index.similar_to(1, k=2)[0].movie_id
    assert batch[1][0][0] == index.similar_to(3, k=2)[0].movie_id


def test_search_prefers_exact_then_prefix(index):
    assert index.search("Space Sequel")[0].movie_id == 2
    assert index.search("space")[0].movie_id in (1, 2)
    assert index.search("nonexistent") == []


def test_search_is_case_and_punctuation_insensitive(index):
    assert index.search("ROMANTIC comedy!")[0].movie_id == 3


def test_search_handles_movielens_inverted_articles():
    """MovieLens stores "Godfather, The"; searching "The Godfather" must match."""
    meta = {
        10: MovieMeta(10, "Godfather, The", 1972, ("Crime", "Drama"), 100_000),
        11: MovieMeta(11, "The Godfather Legacy", 2012, ("Documentary",), 40),
    }
    inverted = ContentIndex(
        np.array([10, 11], dtype=np.int32),
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        meta,
        {10: (), 11: ()},
    )

    results = inverted.search("The Godfather")

    assert results[0].movie_id == 10
    assert results[0].year == 1972


def test_search_prefers_the_better_known_title():
    meta = {
        10: MovieMeta(10, "Solaris", 1972, ("Sci-Fi",), 30_000),
        11: MovieMeta(11, "Solaris", 2002, ("Sci-Fi",), 12_000),
    }
    index = ContentIndex(
        np.array([10, 11], dtype=np.int32),
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        meta,
        {10: (), 11: ()},
    )

    assert index.search("Solaris")[0].movie_id == 10


def test_search_matches_titles_that_carry_punctuation():
    """"KGF" must find "K.G.F: Chapter 1" - punctuation becomes separators."""
    meta = {
        40: MovieMeta(40, "K.G.F: Chapter 1", 2018, ("Action",), 42),
        41: MovieMeta(41, "2.0", 2018, ("Sci-Fi",), 18),
    }
    index = ContentIndex(
        np.array([40, 41], dtype=np.int32),
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        meta,
        {40: (), 41: ()},
    )

    assert index.search("kgf")[0].movie_id == 40
    assert index.search("K.G.F")[0].movie_id == 40
    assert index.search("2.0")[0].movie_id == 41


def test_article_forms_share_a_tier_with_literal_matches():
    """No tier is reserved for literal matches.

    Separating literal from article-insensitive gave "Master" (2021) the top
    slot but sent "The Godfather" (1972) below an obscure 1991 "GodFather".
    Both are in the same tier; popularity decides.
    """
    meta = {
        30: MovieMeta(30, "The Master", 2012, ("Drama",), 3_075),
        31: MovieMeta(31, "Master", 2021, ("Action",), 7),
    }
    index = ContentIndex(
        np.array([30, 31], dtype=np.int32),
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        meta,
        {30: (), 31: ()},
    )

    results = [m.movie_id for m in index.search("Master")]

    assert results == [30, 31]      # both found, the well-known one first


def test_exact_ties_are_broken_by_popularity():
    """Documentaries share one genre and no tags, so their vectors are equal."""
    meta = {
        20: MovieMeta(20, "Obscure Doc", 2001, ("Documentary",), 35),
        21: MovieMeta(21, "Famous Doc", 2001, ("Documentary",), 9_000),
        22: MovieMeta(22, "Other Doc", 2001, ("Documentary",), 120),
    }
    tied = ContentIndex(
        np.array([20, 21, 22], dtype=np.int32),
        np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        meta,
        {20: (), 21: (), 22: ()},
    )

    results = tied.similar_to(20, k=2)

    assert [rec.movie_id for rec in results] == [21, 22]


def test_similar_to_never_returns_an_excluded_tie(index):
    """Tie-break expansion must still honour the self/exclude mask."""
    meta = {
        20: MovieMeta(20, "A", 2001, ("Documentary",), 10),
        21: MovieMeta(21, "B", 2001, ("Documentary",), 20),
    }
    tied = ContentIndex(
        np.array([20, 21], dtype=np.int32),
        np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        meta,
        {20: (), 21: ()},
    )

    assert tied.similar_to(20, k=1)[0].movie_id == 21


def test_save_and_load_roundtrip(tmp_path, index):
    index.save(
        tmp_path,
        index.movie_ids,
        index._vectors,
        [META[mid] for mid in IDS],
        [TERMS[mid] for mid in IDS],
    )

    restored = ContentIndex.load(tmp_path)

    assert len(restored) == 4
    assert restored.meta(1).title == "Space Adventure"
    assert restored.meta(1).genres == ("Sci-Fi", "Adventure")
    assert restored.terms(1) == TERMS[1]
    assert restored.similar_to(1, k=1)[0].movie_id == 2


def test_load_rejects_mismatched_artifacts(tmp_path):
    np.save(tmp_path / "vectors.npy", np.zeros((3, 2), dtype=np.float16))
    np.save(tmp_path / "movie_ids.npy", IDS)

    import gzip
    import json

    with gzip.open(tmp_path / "catalog.json.gz", "wt", encoding="utf-8") as handle:
        json.dump({"ids": [1, 2, 3], "titles": ["a", "b", "c"], "years": [1, 2, 3],
                   "genres": [[], [], []], "terms": [[], [], []]}, handle)

    with pytest.raises(ValueError):
        ContentIndex.load(tmp_path)
