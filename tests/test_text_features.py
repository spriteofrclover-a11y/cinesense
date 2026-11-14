"""Feature construction: shapes, weights, and the sparse-to-latent reduction."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from cinesense.config import FeatureSettings
from cinesense.features.text import (
    build_field_matrices,
    reduce_to_latent,
    top_terms_per_row,
)

#: The shipped config uses min_df_tags=5, which is right for 87k movies and
#: prunes a 6-row fixture to nothing. Tests use 1.
TEST_SETTINGS = FeatureSettings(min_df_tags=1)


@pytest.fixture
def settings() -> FeatureSettings:
    return TEST_SETTINGS


@pytest.fixture
def catalogue() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "movie_id": [1, 2, 3, 4, 5, 6],
            "title": [
                "Space Adventure",
                "Space Adventure Returns",
                "Space Wars",
                "Romantic Picnic",
                "Romantic Picnic Returns",
                "Space Crime Story",
            ],
            "genres": [
                "Sci-Fi|Adventure",
                "Sci-Fi|Adventure",
                "Sci-Fi|Action",
                "Romance|Comedy",
                "Romance|Comedy",
                "Crime|Drama",
            ],
            "tag_text": [
                "space opera",
                "space opera sequel",
                "space battle",
                "love story",
                "love story sequel",
                "murder mystery",
            ],
        }
    )


def test_matrix_is_one_row_per_movie(catalogue, settings):
    bundle = build_field_matrices(catalogue, settings)
    assert bundle.matrix.shape[0] == len(catalogue)


def test_each_field_contributes_features(catalogue, settings):
    bundle = build_field_matrices(catalogue, settings)

    assert set(bundle.field_sizes) == {"title", "genres", "tags"}
    assert all(size > 0 for size in bundle.field_sizes.values())


def test_feature_names_are_prefixed_by_field(catalogue, settings):
    bundle = build_field_matrices(catalogue, settings)
    prefixes = {str(name).split(":", 1)[0] for name in bundle.feature_names}
    assert prefixes == {"title", "genres", "tags"}


def test_zero_weight_field_is_skipped(catalogue):
    bundle = build_field_matrices(catalogue, replace(TEST_SETTINGS, tags_weight=0.0))

    assert "tags" not in bundle.field_sizes
    assert "tags" not in bundle.field_nnz
    assert all(str(name).startswith(("title:", "genres:")) for name in bundle.feature_names)


def test_all_weights_zero_is_an_error(catalogue):
    settings = replace(
        TEST_SETTINGS, title_weight=0.0, genres_weight=0.0, tags_weight=0.0
    )
    with pytest.raises(ValueError, match="weight 0"):
        build_field_matrices(catalogue, settings)


def test_pruned_field_names_itself_in_the_error(catalogue):
    """A min_df that empties a field must say which field and how to fix it."""
    with pytest.raises(ValueError, match="min_df_tags"):
        build_field_matrices(catalogue, replace(TEST_SETTINGS, min_df_tags=99))


def test_genre_hyphens_are_not_split(catalogue, settings):
    """"Sci-Fi" must stay one token, not become "sci" + "fi"."""
    bundle = build_field_matrices(catalogue, settings)
    names = [str(n) for n in bundle.feature_names]
    assert "genres:scifi" in names
    assert not any(n == "genres:sci" for n in names)


def test_rows_are_l2_normalised_per_field(catalogue):
    """Weighting happens after normalisation, so each row length equals the weight."""
    settings = replace(
        TEST_SETTINGS, title_weight=1.0, genres_weight=1.0, tags_weight=0.0
    )
    bundle = build_field_matrices(catalogue, settings)

    squared = np.ravel(bundle.matrix.multiply(bundle.matrix).sum(axis=1))
    assert np.allclose(np.sqrt(squared), np.sqrt(2.0), atol=1e-5)


def test_reduce_to_latent_returns_normalised_vectors(catalogue, settings):
    bundle = build_field_matrices(catalogue, settings)
    vectors, explained = reduce_to_latent(bundle.matrix, components=4)

    assert vectors.shape == (len(catalogue), 4)
    assert vectors.dtype == np.float32
    assert 0.0 < explained <= 1.0

    lengths = np.linalg.norm(vectors, axis=1)
    assert np.allclose(lengths, 1.0, atol=1e-5)


def test_reduce_to_latent_rejects_too_many_components(catalogue, settings):
    bundle = build_field_matrices(catalogue, settings)
    with pytest.raises(ValueError, match="lsa_components"):
        reduce_to_latent(bundle.matrix, components=len(catalogue))


def test_top_terms_returns_the_heaviest_terms():
    matrix = sparse.csr_matrix(
        np.array(
            [
                [0.0, 0.9, 0.1],
                [0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
    )
    names = np.array(["title:a", "tags:space", "tags:minor"])

    terms = top_terms_per_row(matrix, names, k=2)

    assert terms[0] == ("tags:space", "tags:minor")
    assert terms[1] == ()  # empty rows must not crash


def test_top_terms_length_is_capped(catalogue, settings):
    bundle = build_field_matrices(catalogue, settings)
    terms = top_terms_per_row(bundle.matrix, bundle.feature_names, k=3)

    assert len(terms) == len(catalogue)
    assert all(len(row) <= 3 for row in terms)
