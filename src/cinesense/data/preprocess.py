"""Pure-ish dataframe shaping for the movie catalogue.

Every function here is small, deterministic and unit-testable without network
access or the full dataset - that is the point. Real MovieLens titles are messy
(comma-inverted, trailing year, "TV Series" ranges, "(no genres listed)"), and
those edge cases silently corrupt feature vectors if unhandled.
"""

from __future__ import annotations

import re

import pandas as pd

__all__ = [
    "NO_GENRES",
    "build_search_text",
    "enrich_movies",
    "naturalise_title",
    "normalize_text",
    "parse_genres",
    "split_title_year",
]

#: MovieLens uses this literal placeholder when a film carries no genre.
NO_GENRES = "(no genres listed)"

#: Trailing year, optionally a range: "(1995)", "(2010-2015)", "(2010–2015)".
_YEAR_RE = re.compile(r"\((\d{4})(?:\s*[-–]\s*\d{4})?\)\s*$")

#: MovieLens inverts titles carrying a leading article: "Matrix, The (1999)".
#: Two reasons to undo it: it reads wrong in the UI, and the title field feeds
#: the TF-IDF, where "matrix, the" produces the meaningless bigram "matrix the".
_TRAILING_ARTICLE_RE = re.compile(r"^(?P<body>.+),\s*(?P<article>the|a|an)$", re.IGNORECASE)

#: Everything that is not a letter or digit, for tokenising search text.
_NON_ALNUM_RE = re.compile(r"[^0-9a-z]+")


def parse_genres(raw: object) -> list[str]:
    """``"Action|Sci-Fi"`` -> ``["Action", "Sci-Fi"]``.

    Handles None/NaN, blank strings and the placeholder, all of which would
    otherwise become a genre literally named "(no genres listed)".
    """
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    text = str(raw).strip()
    if not text or text == NO_GENRES:
        return []
    return [part.strip() for part in text.split("|") if part.strip()]


def split_title_year(raw: object) -> tuple[str, int | None]:
    """``"Matrix, The (1999)"`` -> ``("Matrix, The", 1999)``."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return "", None
    text = str(raw).strip()
    match = _YEAR_RE.search(text)
    if match is None:
        return text, None
    return text[: match.start()].strip(), int(match.group(1))


def naturalise_title(title: str) -> str:
    """Move a trailing article back to the front.

    ``"Matrix, The"`` -> ``"The Matrix"``. Film titles are stored the way
    MovieLens indexes them, which is not how anyone writes them.
    """
    match = _TRAILING_ARTICLE_RE.match(title.strip())
    if match is None:
        return title
    return f"{match.group('article').capitalize()} {match.group('body').strip()}"


def normalize_text(text: object) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    # sub() replaces each run of non-alphanumerics with a single space, then
    # split()/join() collapses the leftovers ("am lie    2001" -> "am lie 2001").
    return " ".join(_NON_ALNUM_RE.sub(" ", str(text).lower()).split())


def build_search_text(
    title: str,
    genres: list[str],
    tag_text: str = "",
    *,
    max_tag_chars: int | None = None,
) -> str:
    """Concatenate the fields a text search should match on.

    ``max_tag_chars`` defaults to ``None`` (no cap) on purpose. An earlier
    version truncated tags at 400 characters, but ``tag_text`` is alphabetically
    ordered, so the cap silently kept only tags starting with 'a'..'d' for
    popular films - a bias, not a size saving. Vocabulary growth is handled
    properly by the vectoriser's ``min_df``/``max_features`` in Phase 2.
    """
    parts = [normalize_text(title)]
    if genres:
        parts.append(" ".join(normalize_text(g) for g in genres))
    if tag_text:
        trimmed = tag_text[:max_tag_chars] if max_tag_chars else tag_text
        parts.append(normalize_text(trimmed))
    return " ".join(part for part in parts if part)


def enrich_movies(
    movies: pd.DataFrame,
    tags: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Turn the raw ``movies`` table into the canonical catalogue frame.

    Output columns:
        movie_id, title, year, genres (pipe string), genre_list (list),
        n_genres, tag_text, n_tags, search_text
    """
    required = {"movie_id", "title", "genres_raw"}
    missing = required - set(movies.columns)
    if missing:
        raise KeyError(f"movies frame is missing columns: {sorted(missing)}")

    frame = movies.copy()
    frame = frame.drop_duplicates(subset="movie_id", keep="first")

    cleaned = frame["title"].map(split_title_year)
    frame["title"] = [naturalise_title(title) for title, _ in cleaned]
    frame["year"] = pd.array([year for _, year in cleaned], dtype="Int32")

    frame["genre_list"] = frame["genres_raw"].map(parse_genres)
    frame["genres"] = frame["genre_list"].map("|".join)
    frame["n_genres"] = frame["genre_list"].map(len).astype("int16")

    tags_frame = tags if tags is not None else _empty_tags()
    frame = frame.merge(tags_frame, on="movie_id", how="left")
    frame["tag_text"] = frame["tag_text"].fillna("").astype("string")
    # n_tags must come from the aggregation: tag_text is space-joined and tags
    # themselves contain spaces, so it cannot be recovered here.
    frame["n_tags"] = frame["n_tags"].fillna(0).astype("int32")

    frame["search_text"] = [
        build_search_text(title, genres, tag or "")
        for title, genres, tag in zip(
            frame["title"], frame["genre_list"], frame["tag_text"], strict=True
        )
    ]

    return frame[
        [
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
    ].sort_values("movie_id", ignore_index=True)


def _empty_tags() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "movie_id": pd.Series(dtype="int64"),
            "tag_text": pd.Series(dtype="string"),
            "n_tags": pd.Series(dtype="int32"),
        }
    )
