"""Credential handling for poster enrichment.

The key is read from the environment or a local .env, never committed, and never
required by the running app. The fetch itself is tested with a fake session so
no network call is made.
"""

from __future__ import annotations

import pytest

from cinesense.config import PosterSettings
from cinesense.data import posters as posters_module


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    """Point the module at an empty project root and clear the environment."""
    monkeypatch.delenv("TMDB_API_KEY", raising=False)
    monkeypatch.setattr(posters_module, "PROJECT_ROOT", tmp_path)
    return tmp_path


# --------------------------------------------------------------------------- #
# Reading the key
# --------------------------------------------------------------------------- #


def test_missing_key_returns_none(clean_env):
    assert posters_module.api_key(PosterSettings()) is None


def test_empty_value_is_treated_as_missing(clean_env):
    (clean_env / ".env").write_text("TMDB_API_KEY=\n", encoding="utf-8")
    assert posters_module.api_key(PosterSettings()) is None


def test_reads_key_from_dotenv(clean_env):
    (clean_env / ".env").write_text("TMDB_API_KEY=abc123\n", encoding="utf-8")
    assert posters_module.api_key(PosterSettings()) == "abc123"


def test_strips_quotes_and_whitespace(clean_env):
    (clean_env / ".env").write_text('TMDB_API_KEY="  spaced key  "\n', encoding="utf-8")
    assert posters_module.api_key(PosterSettings()) == "spaced key"


def test_ignores_comments_and_other_variables(clean_env):
    (clean_env / ".env").write_text(
        "# TMDB_API_KEY=commented\nOTHER=1\nTMDB_API_KEY=real\n", encoding="utf-8"
    )
    assert posters_module.api_key(PosterSettings()) == "real"


def test_environment_wins_over_dotenv(clean_env, monkeypatch):
    (clean_env / ".env").write_text("TMDB_API_KEY=fromfile\n", encoding="utf-8")
    monkeypatch.setenv("TMDB_API_KEY", "fromenv")
    assert posters_module.api_key(PosterSettings()) == "fromenv"


# --------------------------------------------------------------------------- #
# Which credential shape authenticates how
# --------------------------------------------------------------------------- #


class _Response:
    def __init__(self, status: int, payload: dict | None = None) -> None:
        self.status_code = status
        self._payload = payload or {}

    def json(self) -> dict:
        return self._payload


class _Session:
    """Records the call and returns a canned response."""

    def __init__(self, response: _Response) -> None:
        self.response = response
        self.calls: list[dict] = []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append({"url": url, "params": params or {}, "headers": headers or {}})
        return self.response


def test_v3_key_goes_in_the_query_string():
    session = _Session(
        _Response(200, {"poster_path": "/abc.jpg", "original_language": "hi", "origin_country": ["IN"]})
    )

    record = posters_module._fetch_poster(session, "0123456789abcdef", 862)

    assert record["p"] == "/abc.jpg"
    assert record["l"] == "hi"
    assert record["c"] == ["IN"]
    assert session.calls[0]["params"]["api_key"] == "0123456789abcdef"
    assert "Authorization" not in session.calls[0]["headers"]


def test_v4_token_goes_in_the_authorization_header():
    token = "eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiJ4In0.signature"
    session = _Session(_Response(200, {"poster_path": "/abc.jpg"}))

    posters_module._fetch_poster(session, token, 862)

    assert session.calls[0]["headers"]["Authorization"] == f"Bearer {token}"
    assert "api_key" not in session.calls[0]["params"]


def test_a_film_without_a_poster_still_records_its_language():
    """Films can lack artwork but never lack a language in practice."""
    session = _Session(_Response(200, {"poster_path": None, "original_language": "ta"}))

    record = posters_module._fetch_poster(session, "key", 1)

    assert record["p"] == ""
    assert record["l"] == "ta"


def test_unknown_film_is_stored_as_blank():
    session = _Session(_Response(404))
    record = posters_module._fetch_poster(session, "key", 1)

    assert record["p"] == ""
    assert record["l"] == ""


def test_rejected_credential_explains_the_two_options():
    session = _Session(_Response(401))

    with pytest.raises(posters_module.PosterError, match="v4 'API Read Access Token'"):
        posters_module._fetch_poster(session, "bad-key", 1)


def test_legacy_entries_without_language_still_load(monkeypatch, tmp_path):
    """Older builds stored a bare poster path; those must not crash the loader."""
    import gzip
    import json

    # Never point this at the real artifact: it holds 19k resolved posters.
    target = tmp_path / "posters.json.gz"
    monkeypatch.setattr(posters_module, "POSTERS_FILE", target)

    legacy = {"1": "/abc.jpg", "2": {"p": "/def.jpg", "l": "hi", "c": ["IN"]}}
    with gzip.open(target, "wt", encoding="utf-8") as handle:
        json.dump(legacy, handle)

    meta = posters_module.load_tmdb_meta()

    assert meta[1]["p"] == "/abc.jpg"
    assert meta[1]["l"] == ""          # unknown, so it will be re-fetched
    assert meta[2]["l"] == "hi"
