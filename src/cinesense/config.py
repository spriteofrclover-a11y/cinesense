"""Typed configuration and canonical filesystem paths.

Two rules this module enforces:

1. No magic numbers in the pipeline. Thresholds and weights live in
   ``configs/base.toml`` so they can be tuned without touching code.
2. Unknown config keys are an error. A silently ignored typo is a bug that
   surfaces three phases later as a mysteriously wrong number.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping, get_type_hints

__all__ = [
    "ConfigError",
    "CollabSettings",
    "DatasetSettings",
    "FeatureSettings",
    "FilterSettings",
    "GenomeSettings",
    "HybridSettings",
    "IngestSettings",
    "PosterSettings",
    "QualitySettings",
    "RecommendSettings",
    "Settings",
    "ARTIFACTS_DIR",
    "CONFIG_DIR",
    "DATA_DIR",
    "DEFAULT_CONFIG_PATH",
    "PROCESSED_DIR",
    "PROJECT_ROOT",
    "RAW_DIR",
    "load_settings",
]


class ConfigError(ValueError):
    """Raised when configuration is missing, malformed, or has unknown keys."""


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

#: Repository root. ``src/cinesense/config.py`` -> ``src/cinesense`` -> ``src`` -> root.
PROJECT_ROOT = Path(
    os.environ.get("CINESENSE_ROOT", Path(__file__).resolve().parents[2])
).resolve()

CONFIG_DIR = PROJECT_ROOT / "configs"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "base.toml"

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class FilterSettings:
    """Rating-count thresholds for pruning the ratings matrix (Phase 3)."""

    min_movie_ratings: int = 30
    min_user_ratings: int = 5


@dataclass(frozen=True, slots=True)
class DatasetSettings:
    """Where the raw data comes from and how much of it to pull."""

    name: str = "MovieLens 32M"
    slug: str = "ml-32m"
    url: str = "https://files.grouplens.org/datasets/movielens/ml-32m.zip"
    sha256: str = ""

    @property
    def archive_name(self) -> str:
        return f"{self.slug}.zip"


@dataclass(frozen=True, slots=True)
class IngestSettings:
    """Resource ceilings for the duckdb CSV -> parquet pass."""

    memory_limit: str = "2GB"
    threads: int = 4


@dataclass(frozen=True, slots=True)
class QualitySettings:
    """Build-time sanity gates. Breaching these fails ``build-catalog``."""

    min_tmdb_coverage: float = 0.30
    max_sparsity: float = 0.999
    min_ratings: int = 100_000
    min_genre_coverage: float = 0.90
    filters: FilterSettings = field(default_factory=FilterSettings)


@dataclass(frozen=True, slots=True)
class FeatureSettings:
    """How the content-based feature matrix is assembled.

    Weights are applied to each L2-normalised field before concatenation, so a
    single field cannot dominate simply by having more tokens.
    """

    title_weight: float = 0.5
    genres_weight: float = 1.0
    tags_weight: float = 1.5
    max_features_title: int = 40_000
    max_features_tags: int = 60_000
    min_df_tags: int = 5
    lsa_components: int = 96
    top_terms: int = 8
    random_state: int = 42


@dataclass(frozen=True, slots=True)
class GenomeSettings:
    """MovieLens tag genome, which only ships with ml-25m / ml-latest.

    ``weight`` is relative to the already-normalised text blocks. Measured at
    scripts/measure_genome.py: genome-aware neighbours are rated similarly by
    the same users 2.4x more often than text-only neighbours.
    """

    enabled: bool = True
    url: str = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
    weight: float = 1.5


@dataclass(frozen=True, slots=True)
class RecommendSettings:
    """Ranking behaviour consumed at request time."""

    neighbours: int = 50
    default_k: int = 10


@dataclass(frozen=True, slots=True)
class CollabSettings:
    """Latent-factor collaborative filtering (offline training)."""

    factors: int = 64
    regularisation: float = 10.0
    bias_iterations: int = 5
    test_fraction: float = 0.10
    min_user_ratings: int = 5
    min_movie_ratings: int = 30
    random_state: int = 42


@dataclass(frozen=True, slots=True)
class HybridSettings:
    """How the three signals are blended, and how results are diversified.

    Weights apply to z-scored signals, so they are comparable across users and
    requests. ``mmr_lambda`` of 1.0 disables diversification.
    """

    content_weight: float = 1.0
    collab_weight: float = 1.2
    popularity_weight: float = 0.25
    mmr_lambda: float = 0.75
    #: Candidates to rank before MMR re-orders them.
    candidate_pool: int = 200


@dataclass(frozen=True, slots=True)
class PosterSettings:
    """TMDB poster enrichment.

    The key is read from the environment and never committed. Posters are fetched
    once, offline, and only the *path* is stored; the browser then loads images
    straight from TMDB's CDN, so serving needs no key and no API call.
    """

    enabled: bool = True
    api_key_env: str = "TMDB_API_KEY"
    image_base: str = "https://image.tmdb.org/t/p/"
    size: str = "w342"
    min_ratings: int = 5
    #: Concurrency. Network latency to TMDB (~500 ms/request here) dominates a
    #: sequential loop, so requests overlap instead of queueing.
    requests_per_second: float = 40.0
    workers: int = 20


@dataclass(frozen=True, slots=True)
class Settings:
    dataset: DatasetSettings = field(default_factory=DatasetSettings)
    ingest: IngestSettings = field(default_factory=IngestSettings)
    quality: QualitySettings = field(default_factory=QualitySettings)
    features: FeatureSettings = field(default_factory=FeatureSettings)
    genome: GenomeSettings = field(default_factory=GenomeSettings)
    recommend: RecommendSettings = field(default_factory=RecommendSettings)
    collab: CollabSettings = field(default_factory=CollabSettings)
    hybrid: HybridSettings = field(default_factory=HybridSettings)
    posters: PosterSettings = field(default_factory=PosterSettings)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def _coerce(cls: type, data: Mapping[str, Any], path: str = "") -> Any:
    """Recursively build a frozen dataclass from a TOML mapping."""
    if not is_dataclass(cls):
        raise ConfigError(f"{path or '<root>'}: not a dataclass")

    hints = get_type_hints(cls)
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        where = path or cls.__name__
        raise ConfigError(
            f"{where}: unknown key(s) {sorted(unknown)}. "
            f"Valid keys: {sorted(known)}"
        )

    kwargs: dict[str, Any] = {}
    for key, value in data.items():
        target = hints[key]
        child_path = f"{path}.{key}" if path else key
        if is_dataclass(target):
            if not isinstance(value, Mapping):
                raise ConfigError(f"{child_path}: expected a table, got {type(value).__name__}")
            kwargs[key] = _coerce(target, value, child_path)
        else:
            kwargs[key] = value
    return cls(**kwargs)


def load_settings(path: Path | str | None = None) -> Settings:
    """Load settings from TOML, falling back to dataclass defaults.

    A missing file is not an error: the defaults describe a working
    configuration, so the package is importable without the repo checked out.
    """
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH

    if not config_path.exists():
        return Settings()

    try:
        raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{config_path}: invalid TOML - {exc}") from exc

    return _coerce(Settings, raw)


def ensure_data_dirs() -> None:
    """Create the data tree on demand. Called by the pipeline, never at import."""
    for directory in (RAW_DIR, PROCESSED_DIR, CACHE_DIR, ARTIFACTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
