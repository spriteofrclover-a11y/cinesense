"""The explanation contract is load-bearing: the UI renders it directly."""

from __future__ import annotations

import pytest

from cinesense.recommend.contracts import (
    Explanation,
    Recommendation,
    ScoredMovie,
    contribution_weights,
)


def test_empty_explanation_is_valid():
    explanation = Explanation()
    assert explanation.dominant == "content"
    assert explanation.reasons == ()


def test_contributions_outside_unit_range_are_rejected():
    with pytest.raises(ValueError, match="content contribution"):
        Explanation(content=1.5)
    with pytest.raises(ValueError, match="collaborative contribution"):
        Explanation(collaborative=-0.1)


def test_dominant_picks_the_strongest_signal():
    assert Explanation(content=0.2, collaborative=0.7).dominant == "collaborative"
    assert Explanation(content=0.2, popularity=0.9).dominant == "popularity"


def test_explanation_is_hashable_and_frozen():
    explanation = Explanation(content=0.5, reasons=("because",))
    with pytest.raises(Exception):
        explanation.content = 0.9  # type: ignore[misc]


def test_recommendation_serialises_genres_as_a_list():
    rec = Recommendation(
        movie_id=1,
        title="Arrival",
        year=2016,
        genres=("Sci-Fi", "Drama"),
        score=0.9,
        explanation=Explanation(content=1.0, reasons=("Similar themes",)),
    )

    payload = rec.as_dict()

    assert payload["genres"] == ["Sci-Fi", "Drama"]
    assert payload["explanation"]["reasons"] == ("Similar themes",)
    assert payload["title"] == "Arrival"


def test_contribution_weights_normalise_to_one():
    weights = contribution_weights({"content": 3.0, "collaborative": 1.0})
    assert pytest.approx(sum(weights.values())) == 1.0
    assert weights["content"] == pytest.approx(0.75)


def test_contribution_weights_handle_zero_total():
    weights = contribution_weights({"content": 0.0, "collaborative": 0.0})
    assert weights == {"content": 0.0, "collaborative": 0.0}


def test_contribution_weights_clamp_negatives():
    weights = contribution_weights({"content": -5.0, "collaborative": 2.0})
    assert weights["content"] == 0.0
    assert weights["collaborative"] == 1.0


def test_scored_movie_rejects_nan():
    with pytest.raises(ValueError, match="NaN"):
        ScoredMovie(movie_id=1, score=float("nan"))
