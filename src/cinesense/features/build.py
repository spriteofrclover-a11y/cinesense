"""Build the content model and write the artifacts the API will load.

Runs offline. Reads the Phase 1 catalogue parquet, produces:

    artifacts/content/vectors.npy      float16 latent matrix
    artifacts/content/movie_ids.npy    aligned movie ids
    artifacts/content/catalog.json.gz  title/year/genres/top-terms (stdlib-loadable)
    artifacts/content/manifest.json    provenance

Nothing here is needed at request time; the runtime only reads the artifacts.

The genome block is appended *after* top-terms are extracted, so explanations
only ever quote real words. A genome column is a number, not a term - naming it
"genome:412" in a "why recommended" chip would be noise.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from cinesense.config import ARTIFACTS_DIR, PROCESSED_DIR, Settings, load_settings
from cinesense.features.text import (
    build_field_matrices,
    reduce_to_latent,
    top_terms_per_row,
)
from cinesense.recommend.content import ContentIndex, MovieMeta

__all__ = ["ContentBuild", "build_content", "render_content_report"]

log = logging.getLogger(__name__)


@dataclass(slots=True)
class ContentBuild:
    directory: Path
    manifest_path: Path
    stats: dict


def build_content(
    settings: Settings | None = None,
    *,
    force: bool = False,
) -> ContentBuild:
    settings = settings or load_settings()
    started = time.perf_counter()

    catalogue_path = PROCESSED_DIR / settings.dataset.slug / "movies.parquet"
    if not catalogue_path.exists():
        raise FileNotFoundError(
            f"{catalogue_path} not found. Run `cinesense build-catalog` first."
        )

    output = ARTIFACTS_DIR / "content"
    output.mkdir(parents=True, exist_ok=True)

    log.info("loading catalogue %s", catalogue_path.name)
    catalogue = pd.read_parquet(catalogue_path).reset_index(drop=True)
    movie_ids = catalogue["movie_id"].to_numpy(dtype=np.int32)
    log.info("catalogue: %d movies", len(catalogue))

    log.info("vectorising text fields ...")
    bundle = build_field_matrices(catalogue, settings.features)

    # Terms come from the text matrix only, before the genome is appended.
    log.info("extracting top %d terms per movie ...", settings.features.top_terms)
    term_rows = top_terms_per_row(bundle.matrix, bundle.feature_names, settings.features.top_terms)

    matrix = bundle.matrix
    genome_stats: dict | None = None

    if settings.genome.enabled:
        from cinesense.data.genome import ensure_genome_scores, load_genome_block

        log.info("loading tag genome ...")
        ensure_genome_scores(settings.genome, force=force)
        block = load_genome_block(movie_ids, settings.genome)
        if block is not None:
            matrix = sparse.hstack([matrix, block], format="csr")
            # A row is covered if it has at least one stored value.
            covered = int((np.diff(block.indptr) > 0).sum())
            genome_stats = {
                "weight": settings.genome.weight,
                "nonzeros": int(block.nnz),
                "covered": covered,
                "coverage": round(covered / len(movie_ids), 4),
            }
    else:
        log.info("tag genome disabled in config")

    log.info(
        "combined matrix: %d x %d, %d nonzeros (%.0f MB dense equivalent)",
        matrix.shape[0],
        matrix.shape[1],
        matrix.nnz,
        matrix.shape[0] * matrix.shape[1] * 4 / 1024**2,
    )

    log.info("reducing to %d latent dimensions ...", settings.features.lsa_components)
    vectors, explained = reduce_to_latent(
        matrix, settings.features.lsa_components, settings.features.random_state
    )

    ratings_path = PROCESSED_DIR / settings.dataset.slug / "raw" / "ratings.parquet"
    rating_counts: dict[int, int] = {}
    rating_means: dict[int, float] = {}
    if ratings_path.exists():
        log.info("summarising ratings per movie ...")
        summary = (
            pd.read_parquet(ratings_path, columns=["movie_id", "rating"])
            .groupby("movie_id")["rating"]
            .agg(["count", "mean"])
        )
        rating_counts = summary["count"].to_dict()
        rating_means = summary["mean"].to_dict()

    metadata = [
        MovieMeta(
            movie_id=int(row.movie_id),
            title=str(row.title),
            year=None if pd.isna(row.year) else int(row.year),
            genres=tuple(g for g in str(row.genres).split("|") if g),
            n_ratings=int(rating_counts.get(int(row.movie_id), 0)),
            avg_rating=float(rating_means.get(int(row.movie_id), 0.0)),
        )
        for row in catalogue.itertuples(index=False)
    ]

    log.info("writing artifacts to %s", output)
    ContentIndex.save(output, movie_ids, vectors, metadata, term_rows)

    artifacts = [
        {
            "name": name,
            "bytes": (output / name).stat().st_size,
            "sha256": _sha256(output / name),
            "rows": int(len(catalogue)),
        }
        for name in ("vectors.npy", "movie_ids.npy", "catalog.json.gz")
    ]

    stats = {
        "movies": int(len(catalogue)),
        "dimensions": int(vectors.shape[1]),
        "field_vocabulary": bundle.field_sizes,
        "field_nonzeros": bundle.field_nnz,
        "combined_nonzeros": int(matrix.nnz),
        "text_nonzeros": int(bundle.matrix.nnz),
        "lsa_explained_variance": round(explained, 4),
        "sparse_dense_mb": round(matrix.shape[0] * matrix.shape[1] * 4 / 1024**2, 1),
        "genome": genome_stats,
        "artifact_mb": round(sum(a["bytes"] for a in artifacts) / 1024**2, 2),
        "wall_seconds": round(time.perf_counter() - started, 2),
    }

    from cinesense.data.manifest import write_manifest

    manifest_path = output / "manifest.json"
    write_manifest(
        manifest_path,
        dataset={"name": settings.dataset.name, "slug": settings.dataset.slug, "url": ""},
        settings=settings,
        artifacts=artifacts,
        stats=stats,
        config_path=None,
    )

    return ContentBuild(directory=output, manifest_path=manifest_path, stats=stats)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def render_content_report(built: ContentBuild) -> str:
    s = built.stats
    width = 62

    def row(label: str, value: object, note: str = "") -> str:
        suffix = f"   {note}" if note else ""
        return f"  {label:<26}{value!s:>{width - 28}}{suffix}"

    lines = ["", "-" * width, "  CineSense content model", "-" * width]
    for name, size in s["field_vocabulary"].items():
        lines.append(
            row(f"{name} vocabulary", f"{size:,}", f"{s['field_nonzeros'][name]:,} nonzeros")
        )

    if s["genome"]:
        genome = s["genome"]
        lines.append(
            row(
                "genome coverage",
                f"{genome['coverage']:.1%}",
                f"weight {genome['weight']}, {genome['nonzeros']:,} nnz",
            )
        )
    else:
        lines.append(row("genome", "disabled"))

    lines += [
        "-" * width,
        row("movies", f"{s['movies']:,}"),
        row("latent dimensions", s["dimensions"]),
        row("lsa variance kept", f"{s['lsa_explained_variance']:.1%}"),
        row("nonzeros (text)", f"{s['text_nonzeros']:,}"),
        row("nonzeros (total)", f"{s['combined_nonzeros']:,}"),
        row("sparse as dense", f"{s['sparse_dense_mb']:.0f} MB", "why LSA exists"),
        "-" * width,
        row("shipped artifacts", f"{s['artifact_mb']:.1f} MB", str(built.directory)),
        row("elapsed", f"{s['wall_seconds']:.1f}s"),
        "-" * width,
        "",
    ]
    return "\n".join(lines)
