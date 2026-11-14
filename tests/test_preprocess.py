"""Preprocessing is where messy real-world titles get normalised.

These cases are lifted from actual MovieLens rows, not invented.
"""

from __future__ import annotations

import pandas as pd
import pytest

from cinesense.data.preprocess import (
    NO_GENRES,
    build_search_text,
    enrich_movies,
    naturalise_title,
    normalize_text,
    parse_genres,
    split_title_year,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Action|Sci-Fi", ["Action", "Sci-Fi"]),
        ("Drama", ["Drama"]),
        (NO_GENRES, []),
        ("", []),
        (None, []),
        (float("nan"), []),
        ("Action||Comedy", ["Action", "Comedy"]),
        (" Action | Comedy ", ["Action", "Comedy"]),
    ],
)
def test_parse_genres(raw, expected):
    assert parse_genres(raw) == expected


@pytest.mark.parametrize(
    ("raw", "title", "year"),
    [
        ("Toy Story (1995)", "Toy Story", 1995),
        ("Matrix, The (1999)", "Matrix, The", 1999),
        ("Band of Brothers (2001-2001)", "Band of Brothers", 2001),
        ("Band of Brothers (2001–2001)", "Band of Brothers", 2001),
        ("No Year Here", "No Year Here", None),
        ("Odd Title (nope) (2010)", "Odd Title (nope)", 2010),
        ("", "", None),
    ],
)
def test_split_title_year(raw, title, year):
    assert split_title_year(raw) == (title, year)


def test_normalize_text_strips_punctuation_and_case():
    assert normalize_text("Amélie!  (2001)") == "am lie 2001"
    assert normalize_text(None) == ""


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Matrix, The", "The Matrix"),
        ("Hunt for Red October, The", "The Hunt for Red October"),
        ("Band of Brothers, A", "A Band of Brothers"),
        ("Officer, An", "An Officer"),
        ("Godfather: Part II, The", "The Godfather: Part II"),
        # Nothing to move.
        ("Toy Story", "Toy Story"),
        ("No Article Here", "No Article Here"),
        ("", ""),
        # A comma that is not an article stays put.
        ("Good, Bad and Ugly", "Good, Bad and Ugly"),
    ],
)
def test_naturalise_title_moves_the_article_to_the_front(raw, expected):
    assert naturalise_title(raw) == expected


def test_enrich_movies_naturalises_titles(movies_frame):
    """The catalogue must not ship MovieLens' inverted indexing to the UI."""
    result = enrich_movies(movies_frame, None).set_index("movie_id")

    assert result.loc[2, "title"] == "The Matrix"


def test_build_search_text_does_not_truncate_by_default():
    """An earlier cap biased popular films toward tags starting with 'a'..'d'."""
    long_tag = "zebra " * 500
    text = build_search_text("Zulu", ["War"], long_tag)
    assert text.count("zebra") == 500


def test_build_search_text_caps_only_when_asked():
    long_tag = "zebra " * 500
    text = build_search_text("Zulu", ["War"], long_tag, max_tag_chars=100)
    assert len(text) < 200
    assert "zulu" in text
    assert "war" in text


def test_enrich_movies_shape_and_values(movies_frame, tags_frame):
    result = enrich_movies(movies_frame, tags_frame)

    assert list(result.columns) == [
        "movie_id",
        "title",
        "year",
        "genres",
        "genre_list",
        "n_genres",
        "tag_text",
        "n_tags",
        "search_text",
    ]
    assert result["movie_id"].is_monotonic_increasing
    assert len(result) == 4

    matrix = result.set_index("movie_id")
    assert matrix.loc[2, "genres"] == ""
    assert matrix.loc[2, "n_genres"] == 0
    assert matrix.loc[1, "year"] == 1995
    assert matrix.loc[4, "year"] is None or pd.isna(matrix.loc[4, "year"])
    assert matrix.loc[1, "tag_text"] == "pixar toys childhood"
    assert matrix.loc[1, "n_tags"] == 3
    # A movie with no tags must get an empty string and a zero count, never NaN.
    assert matrix.loc[3, "tag_text"] == ""
    assert matrix.loc[3, "n_tags"] == 0


def test_enrich_movies_without_tags(movies_frame):
    result = enrich_movies(movies_frame, None)
    assert (result["tag_text"] == "").all()


def test_enrich_movies_drops_duplicates(movies_frame):
    duplicated = pd.concat([movies_frame, movies_frame.iloc[[0]]], ignore_index=True)
    result = enrich_movies(duplicated)
    assert len(result) == 4


def test_enrich_movies_requires_known_columns():
    with pytest.raises(KeyError, match="missing columns"):
        enrich_movies(pd.DataFrame({"movie_id": [1]}))


def test_enrich_movies_search_text_combines_fields(movies_frame, tags_frame):
    result = enrich_movies(movies_frame, tags_frame).set_index("movie_id")
    text = result.loc[1, "search_text"]
    assert "toy story" in text
    assert "animation" in text
    assert "pixar" in text
