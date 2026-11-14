"""MovieLens tag genome: 1,128 semantic relevance scores per movie.

Only ships with ml-25m / ml-latest, never with ml-32m, so this module fetches a
second archive and joins on ``movieId`` (stable across MovieLens releases).

Why it is worth a 262 MB download, measured in ``scripts/measure_genome.py``:
content neighbours that agree on the genome are rated similarly by the same
users 2.4x more often than neighbours chosen from text alone. Coverage is 15.8%
of the whole catalogue but 70.2% of movies with at least 30 ratings - the ones
that actually get recommended.

Movies without genome scores simply get a zero block; the text features still
drive them.
"""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import normalize

from cinesense.config import CACHE_DIR, GenomeSettings, PROCESSED_DIR
from cinesense.data.download import download

__all__ = ["GenomeError", "ensure_genome_scores", "load_genome_block"]

log = logging.getLogger(__name__)

GENOME_DIR = PROCESSED_DIR / "genome"
SCORES_PARQUET = GENOME_DIR / "genome_scores.parquet"


class GenomeError(RuntimeError):
    """The genome could not be fetched or parsed."""


def ensure_genome_scores(settings: GenomeSettings, *, force: bool = False) -> Path:
    """Download, extract and ingest genome scores. Idempotent."""
    if SCORES_PARQUET.exists() and not force:
        log.info("cached: genome_scores.parquet")
        return SCORES_PARQUET

    archive = download(settings.url, CACHE_DIR / "ml-25m.zip", force=force)

    GENOME_DIR.mkdir(parents=True, exist_ok=True)
    extracted = GENOME_DIR / "csv"
    marker = extracted / ".done"
    if force or not marker.exists():
        extracted.mkdir(parents=True, exist_ok=True)
        log.info("extracting genome scores from %s", archive.name)
        with zipfile.ZipFile(archive) as bundle:
            found = False
            for member in bundle.namelist():
                if member.endswith("genome-scores.csv"):
                    bundle.extract(member, extracted)
                    found = True
            if not found:
                raise GenomeError(f"{archive.name} contains no genome-scores.csv")
        marker.write_text("ok\n", encoding="utf-8")

    csv_path = next(extracted.rglob("genome-scores.csv"))
    log.info("ingesting genome scores (1,128 scores x ~14k movies) ...")
    con = duckdb.connect()
    try:
        con.execute("SET memory_limit='2GB'")
        source = _q(csv_path)
        target = _q(SCORES_PARQUET)
        con.execute(
            f"""
            COPY (
                SELECT CAST(movieId AS INTEGER)  AS movie_id,
                       CAST(tagId   AS INTEGER)  AS tag_id,
                       CAST(relevance AS FLOAT)  AS relevance
                FROM read_csv({source}, header=true, all_varchar=true)
            ) TO {target} (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
    finally:
        con.close()

    return SCORES_PARQUET


def load_genome_block(
    movie_ids: np.ndarray,
    settings: GenomeSettings,
) -> sparse.csr_matrix | None:
    """Align genome scores to ``movie_ids``, returning a normalised sparse block.

    Rows are L2-normalised before weighting so the genome cannot dominate merely
    by having 1,128 dimensions.
    """
    if not SCORES_PARQUET.exists():
        raise GenomeError(
            f"{SCORES_PARQUET} not found. Run `cinesense build-features` with "
            "genome enabled, or disable it in configs/base.toml."
        )

    scores = pd.read_parquet(SCORES_PARQUET)
    if scores.empty:
        log.warning("genome scores are empty; skipping block")
        return None

    position = {int(mid): row for row, mid in enumerate(movie_ids)}
    mapped = scores["movie_id"].map(position)
    keep = mapped.notna()
    if not keep.any():
        log.warning("no genome movies present in the catalogue; skipping block")
        return None

    rows = mapped[keep].to_numpy(dtype=np.int64)
    columns = scores.loc[keep, "tag_id"].to_numpy(dtype=np.int64) - 1
    values = scores.loc[keep, "relevance"].to_numpy(dtype=np.float32)

    n_tags = int(columns.max()) + 1
    block = sparse.csr_matrix(
        (values, (rows, columns)), shape=(len(movie_ids), n_tags), dtype=np.float32
    )

    covered = int(np.unique(rows).size)
    log.info(
        "genome block: %d/%d movies covered (%.1f%%), %d nonzeros",
        covered,
        len(movie_ids),
        100 * covered / len(movie_ids),
        block.nnz,
    )

    block = normalize(block).astype(np.float32) * settings.weight
    return block.tocsr()

def _q(path: Path | str) -> str:
    return "'" + str(path).replace("\\", "/").replace("'", "''") + "'"
