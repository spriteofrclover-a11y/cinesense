"""Explanation wording, kept deliberately string-based and testable."""

from __future__ import annotations

from cinesense.recommend.explain import (
    content_explanation,
    profile_explanation,
    shared_terms,
    shared_values,
)


def test_shared_values_is_ordered_and_deduplicated():
    left = ["Sci-Fi", "Drama", "Action", "Sci-Fi"]
    right = ["Action", "sci-fi", "Comedy"]

    result = shared_values(left, right)

    assert result == ("Sci-Fi", "Action")


def test_shared_values_respects_limit():
    left = ["a", "b", "c", "d", "e"]
    right = ["a", "b", "c", "d", "e"]
    assert len(shared_values(left, right, limit=2)) == 2


def test_shared_terms_strips_field_prefixes_and_drops_genres():
    """Genre tokens are reported separately, so repeating them here is noise."""
    left = ("tags:space", "genres:action", "title:arrival")
    right = ("tags:space", "genres:action", "title:arrival")

    assert shared_terms(left, right) == ("space", "arrival")


def test_shared_terms_never_leaks_the_genre_field():
    """'Shares Sci-Fi' followed by 'described as scifi' was the old bug."""
    left = ("genres:scifi", "tags:space")
    right = ("genres:scifi", "tags:space")

    terms = shared_terms(left, right)

    assert "scifi" not in terms
    assert terms == ("space",)


def test_content_explanation_mentions_genres_and_terms():
    explanation = content_explanation(
        score=0.8,
        query_title="Arrival",
        query_genres=["Sci-Fi", "Drama"],
        query_terms=["tags:space", "tags:aliens"],
        candidate_genres=["Drama", "Sci-Fi"],
        candidate_terms=["tags:space"],
        query_year=2016,
        candidate_year=2017,
    )

    text = " ".join(explanation.reasons)

    assert "Arrival" in text
    assert "Sci-Fi" in text
    assert "space" in text
    assert explanation.matched_genres == ("Sci-Fi", "Drama")
    assert explanation.matched_terms == ("space",)
    assert explanation.content == 1.0


def test_content_explanation_always_returns_a_reason():
    explanation = content_explanation(
        score=0.1,
        query_title="Obscure Film",
        query_genres=["Drama"],
        candidate_genres=["Comedy"],
    )
    assert len(explanation.reasons) == 1
    assert "Obscure Film" in explanation.reasons[0]


def test_profile_explanation_cites_the_anchor_and_rating():
    explanation = profile_explanation(
        score=0.7,
        anchor_title="Blade Runner 2049",
        anchor_rating=5.0,
        anchor_genres=["Sci-Fi"],
        candidate_genres=["Sci-Fi"],
        content_share=0.6,
    )

    assert "Blade Runner 2049" in explanation.reasons[0]
    assert "5★" in explanation.reasons[0]
    assert explanation.content == 0.6


def test_profile_explanation_clamps_content_share():
    low = profile_explanation(score=0.1, anchor_title="X", content_share=-3.0)
    high = profile_explanation(score=0.1, anchor_title="X", content_share=9.0)

    assert low.content == 0.0
    assert high.content == 1.0
