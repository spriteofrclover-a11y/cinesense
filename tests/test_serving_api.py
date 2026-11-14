"""API tests against the real artifacts.

These need a built pipeline (content vectors, collaborative factors), so they
skip rather than fail on a clean checkout. Everything else in the suite runs
without any of that.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
HAS_MODELS = (ARTIFACTS / "content" / "vectors.npy").exists()

pytestmark = pytest.mark.skipif(
    not HAS_MODELS, reason="offline pipeline not built (run build-features)"
)


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from cinesense.serving.app import create_app
    from cinesense.serving.state import reset_bundle

    reset_bundle()
    with TestClient(create_app()) as test_client:
        yield test_client


def test_health_reports_loaded_models(client):
    payload = client.get("/api/health").json()

    assert payload["status"] == "ok"
    assert payload["models"]["content"] is True
    assert payload["movies"] > 80_000


def test_genres_are_ordered_by_count(client):
    genres = client.get("/api/genres").json()

    assert len(genres) > 10
    counts = [genre["count"] for genre in genres]
    assert counts == sorted(counts, reverse=True)


def test_search_prefers_the_famous_film(client):
    """Regression: "godfather" used to return the 1991 film over the 1972 one."""
    hits = client.get("/api/search?q=godfather").json()

    assert hits
    # Titles are naturalised on the way out: MovieLens stores "Godfather, The".
    assert hits[0]["title"] == "The Godfather"
    assert hits[0]["year"] == 1972


def test_titles_are_not_served_in_movielens_order(client):
    """MovieLens indexes "Matrix, The"; the API must return "The Matrix"."""
    payload = client.get("/api/movies/2571").json()

    assert payload["title"] == "The Matrix"


def test_movie_detail_includes_terms(client):
    payload = client.get("/api/movies/1").json()

    assert payload["title"] == "Toy Story"
    assert payload["terms"]
    assert payload["n_ratings"] > 1000


def test_unknown_movie_is_404(client):
    assert client.get("/api/movies/999999").status_code == 404


def test_similar_returns_the_requested_count(client):
    payload = client.get("/api/movies/1/similar?k=5").json()

    assert len(payload) == 5
    assert all(rec["explanation"]["reasons"] for rec in payload)


def test_bulk_lookup_skips_unknown_ids(client):
    payload = client.post("/api/movies", json={"ids": [1, 999999, 2]}).json()

    assert [movie["id"] for movie in payload] == [1, 2]


def test_recommendations_exclude_rated_films(client):
    payload = client.post(
        "/api/recommendations", json={"ratings": {"1": 5.0, "2": 4.0}, "k": 5}
    ).json()

    ids = {rec["id"] for rec in payload}
    assert len(payload) == 5
    assert ids.isdisjoint({1, 2})


def test_every_recommendation_has_an_explanation(client):
    payload = client.post("/api/recommendations", json={"ratings": {"1": 5.0}, "k": 3}).json()

    for rec in payload:
        explanation = rec["explanation"]
        assert explanation["reasons"]
        assert 0.0 <= explanation["content"] <= 1.0
        assert 0.0 <= explanation["collaborative"] <= 1.0


def test_out_of_range_rating_is_rejected(client):
    response = client.post("/api/recommendations", json={"ratings": {"1": 9.0}})
    assert response.status_code == 422


def test_too_many_ratings_are_rejected(client):
    payload = {"ratings": {str(i): 4.0 for i in range(1, 2100)}}
    assert client.post("/api/recommendations", json=payload).status_code == 422


def test_cold_start_with_genre_only(client):
    payload = client.post(
        "/api/recommendations",
        json={"ratings": {}, "prefs": {"genres": ["Animation"]}, "k": 4},
    ).json()

    assert len(payload) == 4
    assert all("Animation" in rec["genres"] for rec in payload)


def test_static_shell_is_served(client):
    response = client.get("/")

    assert response.status_code == 200
    assert "CineSense" in response.text


def test_stats_expose_model_metrics(client):
    payload = client.get("/api/stats").json()

    assert payload["dataset"]["movies"] > 80_000
    assert "rmse" in payload["collaborative"]
