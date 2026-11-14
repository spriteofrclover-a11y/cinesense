"""Offline content feature construction.

Three text fields (title, genres, tags) are vectorised separately, L2-normalised,
weighted and concatenated. Doing it per field matters: tags have hundreds of
tokens for popular films while genres have three, so a single shared vectoriser
would let tag volume swamp genre signal entirely.

The concatenated TF-IDF matrix is then reduced with TruncatedSVD (LSA). That is
not primarily about quality - it is about what we can ship. The sparse matrix is
~160 MB; 96 latent dimensions is ~17 MB as float16, which is what makes the
Vercel bundle viable.

This module imports pandas and scikit-learn, so it is offline-only.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from cinesense.config import FeatureSettings

__all__ = [
    "FeatureBundle",
    "build_field_matrices",
    "build_feature_bundle",
    "reduce_to_latent",
    "top_terms_per_row",
]

log = logging.getLogger(__name__)

TITLE_PATTERN = r"[a-z0-9]{2,}"
TAG_PATTERN = r"[a-z][a-z0-9']{1,}"
GENRE_PATTERN = r"[A-Za-z]+"


@dataclass(frozen=True, slots=True)
class FieldSpec:
    """One text field and the vectoriser that will read it."""

    name: str
    weight: float
    extract: Callable[[pd.DataFrame], pd.Series]
    make_vectorizer: Callable[[], TfidfVectorizer]


@dataclass(slots=True)
class FeatureBundle:
    """Everything Phase 2 produces before latent reduction."""

    matrix: sparse.csr_matrix
    feature_names: np.ndarray
    field_sizes: dict[str, int]
    field_nnz: dict[str, int]


def _title_text(frame: pd.DataFrame) -> pd.Series:
    return frame["title"].fillna("").astype(str).str.lower()


def _genres_text(frame: pd.DataFrame) -> pd.Series:
    # "Sci-Fi" must not split into "sci" + "fi", hence hyphen removal before
    # the word tokeniser sees it.
    return (
        frame["genres"]
        .fillna("")
        .astype(str)
        .str.replace("-", "", regex=False)
        .str.replace("|", " ", regex=False)
        .str.lower()
    )


def _tags_text(frame: pd.DataFrame) -> pd.Series:
    # Tags are free text: keep letters/digits/apostrophes, drop the rest.
    return (
        frame["tag_text"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.replace(r"[^0-9a-z']+", " ", regex=True)
        .str.strip()
    )


def _field_specs(settings: FeatureSettings) -> list[FieldSpec]:
    return [
        FieldSpec(
            name="title",
            weight=settings.title_weight,
            extract=_title_text,
            make_vectorizer=lambda: TfidfVectorizer(
                analyzer="word",
                token_pattern=TITLE_PATTERN,
                ngram_range=(1, 2),
                min_df=2,
                sublinear_tf=True,
                max_features=settings.max_features_title,
                dtype=np.float32,
            ),
        ),
        FieldSpec(
            name="genres",
            weight=settings.genres_weight,
            extract=_genres_text,
            make_vectorizer=lambda: TfidfVectorizer(
                analyzer="word",
                token_pattern=GENRE_PATTERN,
                lowercase=False,
                min_df=1,
                dtype=np.float32,
            ),
        ),
        FieldSpec(
            name="tags",
            weight=settings.tags_weight,
            extract=_tags_text,
            make_vectorizer=lambda: TfidfVectorizer(
                analyzer="word",
                token_pattern=TAG_PATTERN,
                min_df=settings.min_df_tags,
                sublinear_tf=True,
                strip_accents="unicode",
                max_features=settings.max_features_tags,
                dtype=np.float32,
            ),
        ),
    ]


def build_field_matrices(
    catalogue: pd.DataFrame, settings: FeatureSettings
) -> FeatureBundle:
    """Vectorise each field, weight it, and hstack into one matrix."""
    blocks: list[sparse.csr_matrix] = []
    names: list[np.ndarray] = []
    field_sizes: dict[str, int] = {}
    field_nnz: dict[str, int] = {}

    for spec in _field_specs(settings):
        if spec.weight <= 0:
            log.info("field %-8s skipped (weight 0)", spec.name)
            continue

        texts = spec.extract(catalogue)
        vectorizer = spec.make_vectorizer()
        try:
            matrix = vectorizer.fit_transform(texts)
        except ValueError as exc:
            # sklearn's message names neither the field nor the cause. On a real
            # run this is the difference between a 10-second fix and an hour.
            raise ValueError(
                f"field {spec.name!r} produced no usable vocabulary: {exc}. "
                f"With only {len(catalogue):,} rows, lower min_df_tags in "
                f"configs/base.toml."
            ) from exc
        matrix = normalize(matrix).astype(np.float32)

        # Weight after normalising so token volume cannot dominate.
        matrix = matrix * spec.weight

        blocks.append(matrix.tocsr())
        field_sizes[spec.name] = len(vectorizer.vocabulary_)
        field_nnz[spec.name] = int(matrix.nnz)
        names.append(
            np.array(
                [f"{spec.name}:{term}" for term in vectorizer.get_feature_names_out()],
                dtype=object,
            )
        )
        log.info(
            "field %-8s %7d features  %10d nonzeros  weight %.2f",
            spec.name,
            field_sizes[spec.name],
            field_nnz[spec.name],
            spec.weight,
        )

    if not blocks:
        raise ValueError("every feature field has weight 0; nothing to build")

    combined = sparse.hstack(blocks, format="csr")
    return FeatureBundle(
        matrix=combined,
        feature_names=np.concatenate(names),
        field_sizes=field_sizes,
        field_nnz=field_nnz,
    )


def reduce_to_latent(
    matrix: sparse.csr_matrix,
    components: int,
    random_state: int = 42,
) -> tuple[np.ndarray, float]:
    """LSA reduction. Returns (L2-normalised float32 vectors, variance kept)."""
    if components >= min(matrix.shape):
        raise ValueError(
            f"lsa_components={components} must be smaller than "
            f"min(shape)={min(matrix.shape)}"
        )

    svd = TruncatedSVD(
        n_components=components,
        algorithm="randomized",
        n_iter=7,
        random_state=random_state,
    )
    reduced = svd.fit_transform(matrix)
    explained = float(svd.explained_variance_ratio_.sum())

    vectors = normalize(reduced).astype(np.float32)
    return vectors, explained


def top_terms_per_row(
    matrix: sparse.csr_matrix,
    feature_names: np.ndarray,
    k: int,
) -> list[tuple[str, ...]]:
    """Highest-weighted terms for each row.

    Row-wise on the sparse matrix rather than ``toarray()``: densifying an
    87k x 100k matrix would need ~35 GB.
    """
    results: list[tuple[str, ...]] = []
    indptr, indices, data = matrix.indptr, matrix.indices, matrix.data

    for row in range(matrix.shape[0]):
        start, end = indptr[row], indptr[row + 1]
        row_len = end - start
        if row_len == 0:
            results.append(())
            continue

        values = data[start:end]
        if row_len <= k:
            order = np.argsort(-values)
        else:
            order = np.argpartition(-values, k - 1)[:k]
            order = order[np.argsort(-values[order])]

        results.append(tuple(str(feature_names[indices[start + i]]) for i in order))

    return results
