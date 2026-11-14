"""The `build-catalog` pipeline: raw zip in, verified parquet + manifest out.

Design notes worth keeping:

* Every step is idempotent. A second run re-uses the downloaded archive, the
  extracted tree, the parquet files and the enriched catalogue - only the
  manifest is rewritten. That makes this cheap enough to run before every phase.
* Quality gates are enforced, not logged. A catalogue that lost most of its
  ratings to a bad ingest should stop the build, not produce a quietly worse
  model four phases later.
* Thresholds are *reported* here and *applied* in Phase 3. Seeing "min 30
  ratings would keep 12,104 movies" is information; silently dropping them is
  not.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from cinesense.config import (
    ARTIFACTS_DIR,
    DEFAULT_CONFIG_PATH,
    PROCESSED_DIR,
    RAW_DIR,
    Settings,
    ensure_data_dirs,
    load_settings,
)
from cinesense.data import preprocess
from cinesense.data.download import download, extract_zip
from cinesense.data.ingest import TABLE_SPECS, connect, ingest_table, table_row_count
from cinesense.data.manifest import artifact_entry, write_manifest

__all__ = ["BuildError", "QualityGateError", "build_catalog", "render_report"]

log = logging.getLogger(__name__)


class BuildError(RuntimeError):
    """The pipeline could not complete."""


class QualityGateError(BuildError):
    """The dataset built, but is too poor to model from."""


@dataclass(slots=True)
class BuiltCatalog:
    """Where everything ended up, plus the numbers we measured."""

    processed_dir: Path
    manifest_path: Path
    stats: dict[str, Any]


def build_catalog(
    settings: Settings | None = None,
    *,
    force: bool = False,
    config_path: Path | str | None = DEFAULT_CONFIG_PATH,
) -> BuiltCatalog:
    """Run the whole Phase 1 pipeline. Safe to call repeatedly."""
    settings = settings or load_settings()
    ensure_data_dirs()

    started = time.perf_counter()
    slug = settings.dataset.slug
    raw_root = RAW_DIR / slug
    processed = PROCESSED_DIR / slug
    # Ingested tables and derived tables live in separate directories. Sharing
    # one directory meant "movies.parquet" was both the ingest target and the
    # enrich target, so a cache hit silently served the un-enriched table.
    ingest_dir = processed / "raw"
    ingest_dir.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)

    # ---- 1. fetch ---------------------------------------------------------
    archive = download(
        settings.dataset.url,
        RAW_DIR / settings.dataset.archive_name,
        expected_sha256=settings.dataset.sha256 or None,
        force=force,
    )

    # ---- 2. unpack --------------------------------------------------------
    extracted = extract_zip(archive, raw_root, force=force)
    csv_dir = _locate_csv_dir(extracted)

    # ---- 3. CSV -> parquet ------------------------------------------------
    con = connect(settings.ingest)
    try:
        paths: dict[str, Path] = {}
        for name, spec in TABLE_SPECS.items():
            out = ingest_dir / f"{name}.parquet"
            ingest_table(con, spec, csv_dir / spec.filename, out, force=force)
            if out.exists():
                paths[name] = out

        missing = {"movies", "ratings"} - set(paths)
        if missing:
            raise BuildError(f"missing required tables after ingest: {sorted(missing)}")

        # ---- 4. aggregate tags (duckdb: 2M rows, low memory) ---------------
        tags_by_movie = processed / "tags_by_movie.parquet"
        if "tags" in paths:
            _aggregate_tags(con, paths["tags"], tags_by_movie, force=force)
            paths["tags_by_movie"] = tags_by_movie

        # ---- 5. enrich the catalogue --------------------------------------
        movies_parquet = processed / "movies.parquet"
        movies_frame = _enrich_movies(
            con,
            paths["movies"],
            tags_by_movie if tags_by_movie.exists() else None,
            movies_parquet,
            force=force,
        )

        # ---- 6. measure ----------------------------------------------------
        stats = _collect_stats(con, paths, movies_frame, settings)
    finally:
        con.close()

    _enforce_gates(stats, settings)

    # ---- 7. manifest ------------------------------------------------------
    artifacts = [
        artifact_entry(path, rows=stats["rows"].get(name))
        for name, path in sorted({**paths, "movies": movies_parquet}.items())
        if path.exists()
    ]
    manifest_path = ARTIFACTS_DIR / "manifest.json"
    write_manifest(
        manifest_path,
        dataset={"name": settings.dataset.name, "slug": slug, "url": settings.dataset.url},
        settings=settings,
        artifacts=artifacts,
        stats=stats,
        config_path=config_path,
    )

    stats["wall_seconds"] = round(time.perf_counter() - started, 2)
    return BuiltCatalog(processed_dir=processed, manifest_path=manifest_path, stats=stats)


# --------------------------------------------------------------------------- #
# Steps
# --------------------------------------------------------------------------- #


def _locate_csv_dir(extracted: Path) -> Path:
    """MovieLens zips nest their CSVs one folder down. Find it."""
    if (extracted / "movies.csv").exists():
        return extracted
    for candidate in sorted(extracted.iterdir()):
        if candidate.is_dir() and (candidate / "movies.csv").exists():
            return candidate
    raise BuildError(
        f"no movies.csv found under {extracted}. The archive layout is unexpected."
    )


def _aggregate_tags(
    con: duckdb.DuckDBPyConnection,
    tags_parquet: Path,
    out_path: Path,
    *,
    force: bool = False,
) -> Path:
    """Collapse per-user tags into one deduplicated string per movie."""
    if out_path.exists() and not force:
        log.info("cached: tags_by_movie.parquet")
        return out_path

    source = str(tags_parquet).replace("\\", "/").replace("'", "''")
    target = str(out_path).replace("\\", "/").replace("'", "''")
    log.info("aggregating tags per movie ...")
    con.execute(
        f"""
        COPY (
            SELECT movie_id,
                   string_agg(tag, ' ' ORDER BY tag) AS tag_text,
                   count(DISTINCT tag)               AS n_tags
            FROM (SELECT DISTINCT movie_id, tag FROM read_parquet('{source}'))
            GROUP BY movie_id
        ) TO '{target}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    return out_path


def _enrich_movies(
    con: duckdb.DuckDBPyConnection,
    movies_parquet: Path,
    tags_by_movie: Path | None,
    out_path: Path,
    *,
    force: bool = False,
) -> pd.DataFrame:
    """Apply preprocess.enrich_movies and persist the canonical catalogue.

    Always returns the frame as read back from parquet, so the fresh and cached
    paths cannot diverge in schema. (They did once: the cached branch returned a
    frame that was missing the derived columns, and stats crashed on it.)
    """
    if out_path.exists() and not force:
        log.info("cached: movies.parquet")
        return pd.read_parquet(out_path)

    log.info("enriching movie catalogue ...")
    movies = pd.read_parquet(movies_parquet)
    tags = pd.read_parquet(tags_by_movie) if tags_by_movie else None
    frame = preprocess.enrich_movies(movies, tags)

    # Parquet has no native list-of-strings column we want to depend on, so the
    # genre list is stored pipe-joined and re-parsed by consumers.
    to_write = frame.drop(columns=["genre_list"])
    to_write.to_parquet(out_path, engine="pyarrow", compression="zstd", index=False)
    return pd.read_parquet(out_path)


def _collect_stats(
    con: duckdb.DuckDBPyConnection,
    paths: dict[str, Path],
    movies_frame: pd.DataFrame,
    settings: Settings,
) -> dict[str, Any]:
    """One pass per table; all aggregation happens inside duckdb."""

    def pq(name: str) -> str:
        return str(paths[name]).replace("\\", "/").replace("'", "''")

    n_movies, n_ratings, n_users, avg_rating, first_ts, last_ts = con.execute(
        f"""
        SELECT (SELECT count(*) FROM read_parquet('{pq("movies")}')),
               count(*),
               count(DISTINCT user_id),
               round(avg(rating), 4),
               min(ts),
               max(ts)
        FROM read_parquet('{pq("ratings")}')
        """
    ).fetchone()

    n_movies = int(n_movies)
    n_ratings = int(n_ratings)
    n_users = int(n_users)
    density = n_ratings / (n_users * n_movies) if n_users and n_movies else 0.0

    rated_movies = con.execute(
        f"SELECT count(DISTINCT movie_id) FROM read_parquet('{pq('ratings')}')"
    ).fetchone()[0]
    rated_movies = int(rated_movies)

    tags_rows = 0
    if "tags" in paths:
        tags_rows = table_row_count(con, paths["tags"])

    tmdb_present = 0
    imdb_present = 0
    links_rows = 0
    if "links" in paths:
        links_rows, tmdb_present, imdb_present = con.execute(
            f"""
            SELECT count(*),
                   count(*) FILTER (WHERE tmdb_id IS NOT NULL),
                   count(*) FILTER (WHERE imdb_id IS NOT NULL)
            FROM read_parquet('{pq("links")}')
            """
        ).fetchone()
        links_rows = int(links_rows)

    thresholds = settings.quality.filters

    # Reported, not applied: how much of the matrix survives Phase 3 pruning.
    movies_over = con.execute(
        f"""
        SELECT count(*) FROM (
            SELECT movie_id FROM read_parquet('{pq("ratings")}')
            GROUP BY movie_id HAVING count(*) >= {int(thresholds.min_movie_ratings)}
        )
        """
    ).fetchone()[0]
    users_over = con.execute(
        f"""
        SELECT count(*) FROM (
            SELECT user_id FROM read_parquet('{pq("ratings")}')
            GROUP BY user_id HAVING count(*) >= {int(thresholds.min_user_ratings)}
        )
        """
    ).fetchone()[0]

    return {
        "rows": {
            "movies": n_movies,
            "ratings": n_ratings,
            "tags": tags_rows,
            "links": links_rows,
            "movies_enriched": int(len(movies_frame)),
        },
        "users": n_users,
        "rated_movies": rated_movies,
        "ratings": n_ratings,
        "avg_rating": float(avg_rating or 0.0),
        "density": round(density, 8),
        "sparsity": round(1.0 - density, 8),
        "timestamp_range": [int(first_ts or 0), int(last_ts or 0)],
        "tmdb_id_present": int(tmdb_present),
        "tmdb_id_coverage": round(tmdb_present / links_rows, 4) if links_rows else 0.0,
        "imdb_id_coverage": round(imdb_present / links_rows, 4) if links_rows else 0.0,
        "genre_coverage": round(float((movies_frame["n_genres"] > 0).mean()), 4) if n_movies else 0.0,
        "tag_coverage": round(float((movies_frame["tag_text"].str.len() > 0).mean()), 4)
        if n_movies
        else 0.0,
        "min_movie_ratings": int(thresholds.min_movie_ratings),
        "movies_over_min_ratings": int(movies_over),
        "min_user_ratings": int(thresholds.min_user_ratings),
        "users_over_min_ratings": int(users_over),
    }


def _enforce_gates(stats: dict[str, Any], settings: Settings) -> None:
    """Turn 'this dataset looks wrong' into a non-zero exit code."""
    problems: list[str] = []

    if stats["ratings"] < settings.quality.min_ratings:
        problems.append(
            f"only {stats['ratings']:,} ratings (need {settings.quality.min_ratings:,})"
        )
    if stats["sparsity"] > settings.quality.max_sparsity:
        problems.append(
            f"sparsity {stats['sparsity']:.4f} exceeds {settings.quality.max_sparsity}"
        )
    if stats["tmdb_id_coverage"] < settings.quality.min_tmdb_coverage:
        problems.append(
            f"tmdb id coverage {stats['tmdb_id_coverage']:.1%} below "
            f"{settings.quality.min_tmdb_coverage:.0%}"
        )
    if stats["genre_coverage"] < settings.quality.min_genre_coverage:
        problems.append(
            f"genre coverage only {stats['genre_coverage']:.1%} "
            f"(need {settings.quality.min_genre_coverage:.0%})"
        )

    if problems:
        raise QualityGateError(
            "catalogue failed quality gates:\n  - " + "\n  - ".join(problems)
        )


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def render_report(built: BuiltCatalog, settings: Settings) -> str:
    """A compact, ASCII-only summary (PowerShell mangles box drawing)."""
    s = built.stats
    width = 62

    def row(label: str, value: object, note: str = "") -> str:
        suffix = f"   {note}" if note else ""
        return f"  {label:<26}{value!s:>{width - 28}}{suffix}"

    lines = [
        "",
        "-" * width,
        f"  CineSense catalogue  |  {settings.dataset.name}",
        "-" * width,
        row("ratings", f"{s['ratings']:,}"),
        row("users", f"{s['users']:,}"),
        row("movies (catalogue)", f"{s['rows']['movies']:,}"),
        row("movies (rated)", f"{s['rated_movies']:,}"),
        row("tags applied", f"{s['rows']['tags']:,}"),
        row("mean rating", f"{s['avg_rating']:.3f}"),
        row("sparsity", f"{s['sparsity']:.4%}", "(1.0 = empty)"),
        "-" * width,
        row("genre coverage", f"{s['genre_coverage']:.1%}"),
        row("tag coverage", f"{s['tag_coverage']:.1%}"),
        row("tmdb id coverage", f"{s['tmdb_id_coverage']:.1%}", "enrichment join key"),
        row("imdb id coverage", f"{s['imdb_id_coverage']:.1%}"),
        "-" * width,
        row(
            f"movies >= {s['min_movie_ratings']} ratings",
            f"{s['movies_over_min_ratings']:,}",
            "Phase 3 matrix",
        ),
        row(
            f"users >= {s['min_user_ratings']} ratings",
            f"{s['users_over_min_ratings']:,}",
            "Phase 3 matrix",
        ),
        "-" * width,
        row("parquet output", "", str(built.processed_dir)),
        row("manifest", built.manifest_path.name, str(built.manifest_path.parent)),
        row("elapsed", f"{s.get('wall_seconds', 0):.1f}s"),
        "-" * width,
        "",
    ]
    return "\n".join(lines)
