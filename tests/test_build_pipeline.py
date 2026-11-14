"""End-to-end pipeline test on a dataset small enough to assert by hand.

This exists because of a real bug: the ingest step and the enrich step both
wrote ``movies.parquet``, so a cache hit served the un-enriched table and stats
crashed on a missing column. Unit tests on the individual functions could never
have caught it - only the wiring could.

Network is monkeypatched away; everything else is the real pipeline.
"""

from __future__ import annotations

import zipfile
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from cinesense import data as data_pkg  # noqa: F401  (ensures subpackage import)
from cinesense.config import (
    DatasetSettings,
    FilterSettings,
    QualitySettings,
    Settings,
)
from cinesense.data import build as build_module

MOVIES_CSV = """movieId,title,genres
1,Toy Story (1995),Adventure|Animation|Children|Comedy|Fantasy
2,Heat (1995),Action|Crime|Thriller
3,No Genre Movie (2001),(no genres listed)
"""

RATINGS_CSV = """userId,movieId,rating,timestamp
1,1,5.0,1000
1,2,3.0,1001
2,1,4.0,1002
2,3,2.0,1003
"""

TAGS_CSV = """userId,movieId,tag,timestamp
1,1,pixar,1000
1,1,toys,1001
2,2,crime,1002
"""

LINKS_CSV = """movieId,imdbId,tmdbId
1,0114709,862
2,0113277,949
3,,
"""


def _make_archive(path: Path) -> Path:
    with zipfile.ZipFile(path, "w") as bundle:
        bundle.writestr("movies.csv", MOVIES_CSV)
        bundle.writestr("ratings.csv", RATINGS_CSV)
        bundle.writestr("tags.csv", TAGS_CSV)
        bundle.writestr("links.csv", LINKS_CSV)
    return path


def _tiny_settings() -> Settings:
    return replace(
        Settings(),
        dataset=replace(
            DatasetSettings(),
            name="Tiny",
            slug="tiny",
            url="https://example.test/tiny.zip",
            sha256="",
        ),
        quality=replace(
            QualitySettings(),
            min_tmdb_coverage=0.0,
            max_sparsity=1.0,
            min_ratings=0,
            min_genre_coverage=0.0,
            filters=replace(FilterSettings(), min_movie_ratings=1, min_user_ratings=1),
        ),
    )


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Redirect every path the pipeline writes to, and stub out the download."""
    archive = _make_archive(tmp_path / "tiny.zip")

    raw_dir = tmp_path / "data" / "raw"
    processed_dir = tmp_path / "data" / "processed"
    artifacts_dir = tmp_path / "artifacts"

    monkeypatch.setattr(build_module, "RAW_DIR", raw_dir)
    monkeypatch.setattr(build_module, "PROCESSED_DIR", processed_dir)
    monkeypatch.setattr(build_module, "ARTIFACTS_DIR", artifacts_dir)
    monkeypatch.setattr(build_module, "ensure_data_dirs", lambda: None)
    monkeypatch.setattr(build_module, "download", lambda *a, **k: archive)

    return {
        "root": tmp_path,
        "raw": raw_dir,
        "processed": processed_dir,
        "artifacts": artifacts_dir,
    }


def test_pipeline_produces_separate_raw_and_enriched_output(sandbox):
    built = build_module.build_catalog(_tiny_settings(), config_path=None)
    slug_dir = built.processed_dir

    # The regression: ingest output and enriched output must not share a path.
    assert (slug_dir / "raw" / "movies.parquet").exists()
    assert (slug_dir / "movies.parquet").exists()

    raw_movies = pd.read_parquet(slug_dir / "raw" / "movies.parquet")
    enriched = pd.read_parquet(slug_dir / "movies.parquet")

    assert list(raw_movies.columns) == ["movie_id", "title", "genres_raw"]
    assert "n_genres" in enriched.columns
    assert "search_text" in enriched.columns


def test_enrichment_values_survive_the_round_trip(sandbox):
    build_module.build_catalog(_tiny_settings(), config_path=None)
    frame = pd.read_parquet(sandbox["processed"] / "tiny" / "movies.parquet")
    matrix = frame.set_index("movie_id")

    assert matrix.loc[1, "title"] == "Toy Story"
    assert matrix.loc[1, "year"] == 1995
    assert matrix.loc[1, "genres"] == "Adventure|Animation|Children|Comedy|Fantasy"
    assert matrix.loc[1, "tag_text"] == "pixar toys"
    assert matrix.loc[1, "n_tags"] == 2
    assert matrix.loc[3, "n_genres"] == 0
    assert matrix.loc[3, "tag_text"] == ""
    assert matrix.loc[3, "n_tags"] == 0


def test_stats_are_measured_correctly(sandbox):
    built = build_module.build_catalog(_tiny_settings(), config_path=None)
    stats = built.stats

    assert stats["ratings"] == 4
    assert stats["users"] == 2
    assert stats["rows"]["movies"] == 3
    assert stats["rows"]["tags"] == 3
    assert stats["avg_rating"] == pytest.approx(3.5)
    assert stats["tmdb_id_coverage"] == pytest.approx(0.6667, abs=1e-3)
    assert stats["genre_coverage"] == pytest.approx(2 / 3, abs=1e-3)


def test_manifest_lists_every_artifact(sandbox):
    from cinesense.data.manifest import read_manifest

    built = build_module.build_catalog(_tiny_settings(), config_path=None)
    manifest = read_manifest(built.manifest_path)

    names = {entry["name"] for entry in manifest["artifacts"]}
    assert {"movies.parquet", "ratings.parquet", "tags.parquet", "links.parquet"} <= names
    assert manifest["total_bytes"] > 0
    assert manifest["stats"]["ratings"] == 4


def test_second_run_is_idempotent_and_does_not_re_enrich(sandbox):
    """A cache hit must reuse the derived table, not rebuild or corrupt it."""
    first = build_module.build_catalog(_tiny_settings(), config_path=None)
    before = (first.processed_dir / "movies.parquet").read_bytes()

    second = build_module.build_catalog(_tiny_settings(), config_path=None)
    after = (second.processed_dir / "movies.parquet").read_bytes()

    assert before == after
    assert second.stats["ratings"] == first.stats["ratings"]


def test_quality_gate_fails_the_build(sandbox):
    strict = replace(
        _tiny_settings(),
        quality=replace(_tiny_settings().quality, min_ratings=10_000),
    )

    with pytest.raises(build_module.QualityGateError, match="ratings"):
        build_module.build_catalog(strict, config_path=None)


def test_missing_required_csv_is_reported(sandbox, tmp_path, monkeypatch):
    """movies.csv present but ratings.csv absent must name the missing file."""
    broken = tmp_path / "broken.zip"
    with zipfile.ZipFile(broken, "w") as bundle:
        bundle.writestr("movies.csv", MOVIES_CSV)
    monkeypatch.setattr(build_module, "download", lambda *a, **k: broken)

    from cinesense.data.ingest import IngestError

    with pytest.raises(IngestError, match="ratings.csv"):
        build_module.build_catalog(_tiny_settings(), config_path=None)


def test_unexpected_archive_layout_is_reported(sandbox, tmp_path, monkeypatch):
    """An archive with no movies.csv anywhere must not fail with a KeyError."""
    empty = tmp_path / "empty.zip"
    with zipfile.ZipFile(empty, "w") as bundle:
        bundle.writestr("readme.txt", "not a dataset")
    monkeypatch.setattr(build_module, "download", lambda *a, **k: empty)

    with pytest.raises(build_module.BuildError, match="no movies.csv"):
        build_module.build_catalog(_tiny_settings(), config_path=None)
