"""Decide empirically whether the MovieLens tag genome earns its download.

ml-32m ships no genome. ml-25m does: 1,128 semantic relevance scores per movie,
covering 13,812 films (15.8% of the catalogue, but 70.2% of movies with >=30
ratings).

Two metrics, because one of them is biased:

* genre agreement - mean Jaccard of genre sets between a query and its top-k
  neighbours. **Biased**: a model built on genres trivially wins this, so it
  cannot judge thematic similarity. Reported for reference only.
* co-rating similarity - do the same users rate the neighbours similarly? This
  is independent of every feature we feed the model, so it measures something
  the model was not told about.

Genome weight is swept, because hstacking 1,128 genome columns against ~60k text
columns lets it dominate the SVD by dimension count alone. Comparing a single
arbitrary weight would conflate "uninformative" with "badly weighted".

    .\\.venv\\Scripts\\python.exe scripts/measure_genome.py
"""

from __future__ import annotations

import sys
import time
import zipfile
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cinesense.config import CACHE_DIR, PROCESSED_DIR  # noqa: E402
from cinesense.data.download import download  # noqa: E402

GENOME_URL = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
GENOME_MEMBERS = ("genome-scores.csv", "genome-tags.csv")
SAMPLE_SIZE = 400
K = 10
DIMS = 96
SEED = 42
WEIGHTS = (0.0, 0.25, 0.5, 1.0, 2.0)


def fetch_genome() -> Path:
    archive = download(GENOME_URL, CACHE_DIR / "ml-25m.zip")
    target = CACHE_DIR / "genome"
    marker = target / ".size"

    if marker.exists() and marker.read_text().strip() == str(archive.stat().st_size):
        return next(target.rglob("genome-scores.csv"))

    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.namelist():
            if member.endswith(GENOME_MEMBERS):
                bundle.extract(member, target)
    marker.write_text(str(archive.stat().st_size), encoding="utf-8")
    return next(target.rglob("genome-scores.csv"))


def ingest_scores(csv_path: Path) -> tuple[pd.DataFrame, float, float]:
    out = CACHE_DIR / "genome-scores.parquet"
    elapsed = 0.0

    if not out.exists():
        start = time.perf_counter()
        con = duckdb.connect()
        con.execute("SET memory_limit='2GB'")
        source = str(csv_path).replace("\\", "/").replace("'", "''")
        target = str(out).replace("\\", "/").replace("'", "''")
        con.execute(
            f"""
            COPY (
                SELECT CAST(movieId AS INTEGER) AS movie_id,
                       CAST(tagId AS INTEGER)   AS tag_id,
                       CAST(relevance AS FLOAT) AS relevance
                FROM read_csv('{source}', header=true, all_varchar=true)
            ) TO '{target}' (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
        con.close()
        elapsed = time.perf_counter() - start

    frame = pd.read_parquet(out).pivot(index="movie_id", columns="tag_id", values="relevance")
    return frame, elapsed, out.stat().st_size / 1024**2


def lsa(matrix, dims: int = DIMS) -> np.ndarray:
    components = min(dims, min(matrix.shape) - 1)
    reduced = TruncatedSVD(
        n_components=components, algorithm="randomized", n_iter=7, random_state=SEED
    ).fit_transform(matrix)
    return normalize(reduced).astype(np.float32)


def build_corating_matrix(
    ratings_path: Path, common: list[int]
) -> tuple[np.ndarray, dict[int, int]]:
    """Movies x users, mean-centred per user, L2-normalised.

    Mean-centring per user matters: some users rate everything 4-5, and without
    centring their ratings would make unrelated films look similar.
    """
    con = duckdb.connect()
    con.execute("SET memory_limit='2GB'")
    con.register("common", pd.DataFrame({"movie_id": common}))

    start = time.perf_counter()
    frame = con.execute(
        f"""
        SELECT r.movie_id AS movie_id,
               r.user_id  AS user_id,
               r.rating - avg(r.rating) OVER (PARTITION BY r.user_id) AS centred
        FROM read_parquet('{str(ratings_path).replace(chr(92), "/")}') r
        JOIN common USING (movie_id)
        """
    ).df()
    con.close()
    print(f"  co-rating pairs loaded: {len(frame):,} rows in {time.perf_counter() - start:.1f}s")

    movie_index = {mid: position for position, mid in enumerate(common)}
    rows = frame["movie_id"].map(movie_index).to_numpy()
    cols = pd.factorize(frame["user_id"])[0]
    values = frame["centred"].to_numpy(dtype=np.float32)

    matrix = sparse.csr_matrix(
        (values, (rows, cols)), shape=(len(common), int(cols.max()) + 1), dtype=np.float32
    )
    return normalize(matrix), movie_index


def corating_score(
    query_rows: list[int],
    neighbour_groups: list[list[int]],
    corating: np.ndarray,
) -> float:
    """Mean co-rating similarity for the predicted neighbour pairs."""
    similarities = corating[query_rows] @ corating.T
    values: list[float] = []
    for position, neighbours in enumerate(neighbour_groups):
        for row in neighbours:
            value = float(similarities[position, row])
            if np.isfinite(value):
                values.append(value)
    return float(np.mean(values)) if values else 0.0


def genre_agreement(
    vectors: np.ndarray,
    movie_ids: np.ndarray,
    genre_sets: dict[int, set[str]],
    query_rows: list[int],
    neighbour_groups: list[list[int]],
    *,
    k: int = K,
) -> float:
    del k
    values: list[float] = []
    for row, neighbours in zip(query_rows, neighbour_groups, strict=True):
        base = genre_sets.get(int(movie_ids[row]), set())
        for neighbour in neighbours:
            other = genre_sets.get(int(movie_ids[neighbour]), set())
            union = base | other
            if union:
                values.append(len(base & other) / len(union))
    return float(np.mean(values)) if values else 0.0


def neighbour_groups(vectors: np.ndarray, rows: list[int], k: int = K) -> list[list[int]]:
    scores = vectors[rows] @ vectors.T
    for position, row in enumerate(rows):
        scores[position, row] = -np.inf
    return [
        [int(i) for i in np.argpartition(-scores[position], k)[:k]]
        for position in range(len(rows))
    ]


def main() -> int:
    print("fetching ml-25m (genome only) ...")
    scores_csv = fetch_genome()

    print("ingesting genome scores ...")
    genome, elapsed, size_mb = ingest_scores(scores_csv)
    print(f"  shape    {genome.shape[0]:,} movies x {genome.shape[1]:,} tags")
    print(f"  ingest   {elapsed:.1f} s   parquet {size_mb:.0f} MB")
    print()

    catalogue = pd.read_parquet(PROCESSED_DIR / "ml-32m" / "movies.parquet").set_index("movie_id")
    genre_sets = {
        int(mid): {g for g in str(row).split("|") if g}
        for mid, row in catalogue["genres"].items()
    }

    genome_ids = genome.index.to_numpy()
    covered = catalogue.index.isin(genome_ids)

    ratings_path = PROCESSED_DIR / "ml-32m" / "raw" / "ratings.parquet"
    counts = pd.read_parquet(ratings_path, columns=["movie_id"]).groupby("movie_id").size()
    popular = counts[counts >= 30].index
    popular_covered = pd.Index(popular).isin(genome_ids)

    print("coverage")
    print(f"  catalogue movies       {len(catalogue):,}")
    print(f"  with genome            {int(covered.sum()):,}  ({covered.mean():.1%})")
    print(f"  movies >=30 ratings    {len(popular):,}")
    print(f"  ... with genome        {int(popular_covered.sum()):,}  ({popular_covered.mean():.1%})")
    print()

    common = [int(mid) for mid in genome_ids if mid in catalogue.index]
    ids_array = np.asarray(common)
    tags = catalogue.loc[common, "tag_text"].fillna("").astype(str)
    genres_text = (
        catalogue.loc[common, "genres"].fillna("").astype(str).str.replace("|", " ", regex=False)
    )

    print(f"building variants over {len(common):,} movies ...")
    tags_matrix = normalize(
        TfidfVectorizer(min_df=5, sublinear_tf=True, strip_accents="unicode", dtype=np.float32)
        .fit_transform(tags)
    )
    genres_matrix = normalize(TfidfVectorizer(min_df=1, dtype=np.float32).fit_transform(genres_text))
    text = sparse.hstack([genres_matrix * 1.0, tags_matrix * 1.5], format="csr")
    genome_matrix = sparse.csr_matrix(genome.loc[common].to_numpy(dtype=np.float32))

    variants: dict[str, np.ndarray] = {
        "genres only": lsa(genres_matrix),
        "tags only": lsa(tags_matrix),
        "text (production)": lsa(text),
    }
    for weight in WEIGHTS:
        if weight == 0.0:
            continue
        variants[f"text + genome x{weight}"] = lsa(
            sparse.hstack([text, genome_matrix * weight], format="csr")
        )

    print("loading co-rating signal (independent of all features) ...")
    corating, movie_index = build_corating_matrix(ratings_path, common)

    rng = np.random.default_rng(SEED)
    sample = [int(m) for m in rng.choice(common, size=min(SAMPLE_SIZE, len(common)), replace=False)]
    query_rows = [movie_index[mid] for mid in sample]

    # Random pairs, same shape as a real result set, for the baseline.
    random_rows = rng.integers(0, len(common), size=(len(sample), K)).tolist()

    print()
    print(f"  {'variant':<26}{'genre':>9}{'co-rating':>12}")
    print(f"  {'-' * 45}")
    results: dict[str, tuple[float, float]] = {}
    for name, vectors in variants.items():
        groups = neighbour_groups(vectors, query_rows)
        genre = genre_agreement(vectors, ids_array, genre_sets, query_rows, groups)
        co = corating_score(query_rows, groups, corating)
        results[name] = (genre, co)
        print(f"  {name:<26}{genre:>9.4f}{co:>12.4f}")

    genre_random = genre_agreement(ids_array, ids_array, genre_sets, query_rows, random_rows)
    co_random = corating_score(query_rows, random_rows, corating)
    print(f"  {'random baseline':<26}{genre_random:>9.4f}{co_random:>12.4f}")

    best = max(results.items(), key=lambda item: item[1][1])
    print()
    print(f"  best co-rating variant: {best[0]} ({best[1][1]:.4f})")
    print(f"  production text only  : {results['text (production)'][1]:.4f}")
    print(f"  genome lift over text : {best[1][1] / results['text (production)'][1]:.3f}x")
    print()
    print("  Note: genre agreement is biased toward genre-derived features and is")
    print("  shown only for reference. Co-rating similarity is the honest signal.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
