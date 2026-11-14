"""Configuration must be strict: unknown keys are bugs, not typos to ignore."""

from __future__ import annotations

import textwrap

import pytest

from cinesense.config import (
    ConfigError,
    DEFAULT_CONFIG_PATH,
    Settings,
    load_settings,
)


def test_missing_file_returns_defaults(tmp_path):
    settings = load_settings(tmp_path / "nope.toml")
    assert settings.dataset.slug == "ml-32m"
    assert settings.quality.filters.min_movie_ratings == 30


def test_shipped_config_loads_and_is_valid():
    """The real configs/base.toml must always parse."""
    settings = load_settings(DEFAULT_CONFIG_PATH)
    assert settings.dataset.url.startswith("https://")
    assert 0.0 <= settings.quality.min_tmdb_coverage <= 1.0
    assert settings.ingest.threads >= 1


def test_override_nested_table(tmp_path):
    path = tmp_path / "c.toml"
    path.write_text(
        textwrap.dedent(
            """
            [dataset]
            slug = "ml-latest-small"
            url = "https://example.test/small.zip"

            [quality.filters]
            min_movie_ratings = 5
            """
        ),
        encoding="utf-8",
    )
    settings = load_settings(path)

    assert settings.dataset.slug == "ml-latest-small"
    # Unspecified keys keep their defaults rather than becoming None.
    assert settings.dataset.name == "MovieLens 32M"
    assert settings.quality.filters.min_movie_ratings == 5
    assert settings.quality.filters.min_user_ratings == 5


def test_unknown_key_is_rejected(tmp_path):
    path = tmp_path / "bad.toml"
    path.write_text("[dataset]\nslug = 'x'\nslugx = 'typo'\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="unknown key"):
        load_settings(path)


def test_invalid_toml_is_reported_with_path(tmp_path):
    path = tmp_path / "broken.toml"
    path.write_text("[dataset\nslug = 1", encoding="utf-8")

    with pytest.raises(ConfigError, match="invalid TOML"):
        load_settings(path)


def test_settings_are_immutable():
    settings = load_settings()
    with pytest.raises(Exception):
        settings.dataset.slug = "something-else"  # type: ignore[misc]


def test_default_settings_roundtrip():
    assert isinstance(Settings(), Settings)
