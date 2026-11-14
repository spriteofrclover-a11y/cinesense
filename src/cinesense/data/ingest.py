"""CSV -> typed parquet via duckdb.

Why duckdb and not ``pandas.read_csv``: the ratings table is 32,000,204 rows.
Loading that through pandas on this machine spikes several GB and risks paging.
duckdb streams it inside a fixed memory ceiling and writes compact ZSTD parquet
that every later stage reads instead of the CSV.

Explicit casts everywhere, because ``all_varchar=true`` makes the reader immune
to inference surprises (MovieLens leaves ``tmdbId`` blank for some rows).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import duckdb

from cinesense.config import IngestSettings

__all__ = [
    "IngestError",
    "TableSpec",
    "TABLE_SPECS",
    "connect",
    "ingest_table",
    "table_row_count",
]

log = logging.getLogger(__name__)


class IngestError(RuntimeError):
    """A required CSV is missing or produced an unusable parquet file."""


@dataclass(frozen=True, slots=True)
class TableSpec:
    """One MovieLens CSV and the typed projection we keep from it."""

    name: str
    filename: str
    select_sql: str
    required: bool = True


TABLE_SPECS: dict[str, TableSpec] = {
    "movies": TableSpec(
        name="movies",
        filename="movies.csv",
        select_sql="""
            SELECT CAST(movieId AS INTEGER)        AS movie_id,
                   CAST(title   AS VARCHAR)        AS title,
                   CAST(genres  AS VARCHAR)        AS genres_raw
            FROM read_csv({source}, header=true, all_varchar=true)
        """,
    ),
    "ratings": TableSpec(
        name="ratings",
        filename="ratings.csv",
        select_sql="""
            SELECT CAST(userId    AS INTEGER)      AS user_id,
                   CAST(movieId   AS INTEGER)      AS movie_id,
                   CAST(rating    AS FLOAT)        AS rating,
                   CAST(timestamp AS BIGINT)       AS ts
            FROM read_csv({source}, header=true, all_varchar=true)
        """,
    ),
    "tags": TableSpec(
        name="tags",
        filename="tags.csv",
        select_sql="""
            SELECT CAST(userId    AS INTEGER)      AS user_id,
                   CAST(movieId   AS INTEGER)      AS movie_id,
                   lower(trim(CAST(tag AS VARCHAR))) AS tag,
                   CAST(timestamp AS BIGINT)       AS ts
            FROM read_csv({source}, header=true, all_varchar=true)
        """,
    ),
    "links": TableSpec(
        name="links",
        filename="links.csv",
        select_sql="""
            SELECT CAST(movieId AS INTEGER)                                  AS movie_id,
                   TRY_CAST(NULLIF(trim(imdbId), '') AS INTEGER)             AS imdb_id,
                   TRY_CAST(NULLIF(trim(tmdbId), '') AS INTEGER)             AS tmdb_id
            FROM read_csv({source}, header=true, all_varchar=true)
        """,
    ),
}


def _quote(value: Path | str) -> str:
    """Single-quote a path for safe inlining into a duckdb statement."""
    return "'" + str(value).replace("\\", "/").replace("'", "''") + "'"


def connect(settings: IngestSettings) -> duckdb.DuckDBPyConnection:
    """Open an in-memory duckdb connection with bounded resources."""
    con = duckdb.connect(database=":memory:")
    con.execute(f"SET memory_limit = '{settings.memory_limit}'")
    con.execute(f"SET threads = {int(settings.threads)}")
    con.execute("SET preserve_insertion_order = false")
    return con


def ingest_table(
    con: duckdb.DuckDBPyConnection,
    spec: TableSpec,
    csv_path: Path,
    out_path: Path,
    *,
    force: bool = False,
) -> Path:
    """Project one CSV into parquet. No-op when the output already exists."""
    csv_path = Path(csv_path)
    out_path = Path(out_path)

    if out_path.exists() and not force:
        log.info("cached: %s.parquet", spec.name)
        return out_path

    if not csv_path.exists():
        if spec.required:
            raise IngestError(
                f"required file {csv_path.name} not found in {csv_path.parent}. "
                f"Re-run with --force to re-extract the archive."
            )
        log.info("skipped: %s (not present)", spec.filename)
        return out_path

    out_path.parent.mkdir(parents=True, exist_ok=True)
    select = spec.select_sql.format(source=_quote(csv_path))
    statement = (
        f"COPY ({select}) TO {_quote(out_path)} "
        f"(FORMAT PARQUET, COMPRESSION ZSTD)"
    )

    log.info("ingesting %s ...", spec.filename)
    con.execute(statement)

    if not out_path.exists():
        raise IngestError(f"duckdb produced no output for {spec.name}")
    return out_path


def table_row_count(con: duckdb.DuckDBPyConnection, parquet_path: Path | str) -> int:
    """Row count of a parquet file without materialising it."""
    path = Path(parquet_path)
    if not path.exists():
        return 0
    row = con.execute(f"SELECT count(*) FROM read_parquet({_quote(path)})").fetchone()
    return int(row[0]) if row else 0
