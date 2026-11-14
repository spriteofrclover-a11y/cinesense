"""Model loading, done once per process.

The artifacts total ~31 MB. Loading them on every request would add ~1 s to each
one; loading them once at startup makes the first request slow and every
subsequent request a few milliseconds. On a serverless platform each cold start
pays that cost once, which is the right trade.
"""

from __future__ import annotations

import gzip
import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path

from cinesense.config import ARTIFACTS_DIR, Settings, load_settings
from cinesense.recommend.collab import CollabModel
from cinesense.recommend.content import ContentIndex
from cinesense.recommend.hybrid import HybridRecommender

__all__ = ["ModelBundle", "get_bundle", "load_bundle", "reset_bundle"]

log = logging.getLogger(__name__)

CONTENT_DIR = ARTIFACTS_DIR / "content"
COLLAB_DIR = ARTIFACTS_DIR / "collab"
POSTERS_FILE = ARTIFACTS_DIR / "posters" / "posters.json.gz"


@dataclass(slots=True)
class ModelBundle:
    settings: Settings
    content: ContentIndex
    collab: CollabModel | None
    ranker: HybridRecommender
    tmdb: dict[int, dict]
    poster_base: str
    poster_size: str
    stats: dict

    def poster_url(self, movie_id: int) -> str | None:
        path = self.tmdb.get(movie_id, {}).get("p")
        if not path:
            return None
        return f"{self.poster_base}{self.poster_size}{path}"

    def language(self, movie_id: int) -> str | None:
        return self.tmdb.get(movie_id, {}).get("l") or None

    def countries(self, movie_id: int) -> list[str]:
        return list(self.tmdb.get(movie_id, {}).get("c") or [])


_bundle: ModelBundle | None = None
_lock = threading.Lock()


def load_bundle(settings: Settings | None = None) -> ModelBundle:
    """Load every artifact. Raises if the offline pipeline has not run."""
    global _bundle

    settings = settings or load_settings()

    if not (CONTENT_DIR / "vectors.npy").exists():
        raise FileNotFoundError(
            f"no content model at {CONTENT_DIR}. Run `cinesense build-features`."
        )

    content = ContentIndex.load(CONTENT_DIR)

    collab: CollabModel | None = None
    if (COLLAB_DIR / "meta.json").exists():
        collab = CollabModel.load(COLLAB_DIR)
    else:
        log.warning("collaborative model absent; serving content-only")

    # Loaded before the ranker: the ranker caches per-row language and country
    # so a language filter does not re-walk 87k films on every request.
    posters = _load_tmdb()
    if posters:
        log.info(
            "tmdb metadata for %d movies (%d with posters, %d with a language)",
            len(posters),
            sum(1 for record in posters.values() if record.get("p")),
            sum(1 for record in posters.values() if record.get("l")),
        )

    ranker = HybridRecommender(
        content,
        collab,
        settings.hybrid,
        language_of=lambda mid: posters.get(mid, {}).get("l") or None,
        countries_of=lambda mid: list(posters.get(mid, {}).get("c") or []),
    )

    stats = _collect_stats(content, collab, posters)

    bundle = ModelBundle(
        settings=settings,
        content=content,
        collab=collab,
        ranker=ranker,
        tmdb=posters,
        poster_base=settings.posters.image_base,
        poster_size=settings.posters.size,
        stats=stats,
    )
    return bundle


def get_bundle() -> ModelBundle:
    """Return the loaded bundle, loading it on first use."""
    global _bundle
    if _bundle is None:
        with _lock:
            if _bundle is None:
                _bundle = load_bundle()
    return _bundle


def reset_bundle() -> None:
    """Test hook: drop the cached models."""
    global _bundle
    _bundle = None


LANGUAGE_NAMES = {
    "hi": "Hindi", "ta": "Tamil", "te": "Telugu", "ml": "Malayalam",
    "kn": "Kannada", "bn": "Bengali", "mr": "Marathi", "pa": "Punjabi",
    "gu": "Gujarati", "ur": "Urdu", "en": "English", "fr": "French",
    "es": "Spanish", "de": "German", "it": "Italian", "ja": "Japanese",
    "ko": "Korean", "zh": "Chinese", "cn": "Cantonese", "ru": "Russian",
    "sv": "Swedish", "da": "Danish", "nl": "Dutch", "pt": "Portuguese",
    "fa": "Persian", "tr": "Turkish", "pl": "Polish", "th": "Thai",
    "si": "Sinhala", "ne": "Nepali", "id": "Indonesian",
}

COUNTRY_NAMES = {
    "IN": "Indian cinema", "US": "American", "GB": "British",
    "FR": "French", "JP": "Japanese", "KR": "Korean", "IT": "Italian",
    "ES": "Spanish", "DE": "German", "CN": "Chinese", "HK": "Hong Kong",
    "CA": "Canadian", "AU": "Australian", "RU": "Russian", "SE": "Swedish",
    "DK": "Danish", "NL": "Dutch", "BR": "Brazilian", "MX": "Mexican",
    "TR": "Turkish", "IR": "Iranian", "PK": "Pakistani", "LK": "Sri Lankan",
    "BD": "Bangladeshi", "NP": "Nepali", "TH": "Thai", "PL": "Polish",
}

#: A facet needs this many films to be worth showing.
MIN_FACET_FILMS = 5


def _build_facets(content: ContentIndex, tmdb: dict[int, dict]) -> dict:
    """Language and country counts over films that can actually be recommended."""
    languages: dict[str, int] = {}
    regions: dict[str, int] = {}

    for movie_id in content.movie_ids:
        meta = content.meta(int(movie_id))
        if meta is None or meta.n_ratings <= 0:
            continue
        record = tmdb.get(int(movie_id), {})
        code = record.get("l")
        if code:
            languages[code] = languages.get(code, 0) + 1
        for country in record.get("c") or []:
            regions[country] = regions.get(country, 0) + 1

    def pack(counts: dict[str, int], names: dict[str, str]) -> list[dict]:
        return [
            {"code": code, "name": names.get(code, code.upper()), "count": count}
            for code, count in sorted(counts.items(), key=lambda item: -item[1])
            if count >= MIN_FACET_FILMS
        ]

    return {"languages": pack(languages, LANGUAGE_NAMES), "regions": pack(regions, COUNTRY_NAMES)}


def _collect_stats(
    content: ContentIndex, collab: CollabModel | None, tmdb: dict[int, dict]
) -> dict:
    stats: dict = {
        "dataset": {
            "movies": len(content),
            "dimensions": content.dimensions,
        },
        "content": _read_json(CONTENT_DIR / "manifest.json", {}).get("stats", {}),
        "collaborative": _read_json(COLLAB_DIR / "meta.json", {}),
        "posters": {
            "found": sum(1 for record in tmdb.values() if record.get("p")),
            "resolved": len(tmdb),
            "languages": sum(1 for record in tmdb.values() if record.get("l")),
        },
        "ranking": _read_json(ARTIFACTS_DIR / "eval" / "ranking.json", None),
    }
    stats["facets"] = _build_facets(content, tmdb)
    if collab is not None:
        stats["dataset"]["users"] = stats["collaborative"].get("n_users")
    return stats


def _read_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default


def _load_tmdb() -> dict[int, dict]:
    """TMDB metadata per movie, read with the standard library.

    Deliberately not reusing ``cinesense.data.posters``: that module imports
    pandas, which must never enter the deployed bundle.
    """
    if not POSTERS_FILE.exists():
        return {}
    try:
        with gzip.open(POSTERS_FILE, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        log.warning("tmdb index is unreadable; falling back to generated art")
        return {}

    meta: dict[int, dict] = {}
    for key, value in payload.items():
        # Older builds stored a bare poster path.
        meta[int(key)] = value if isinstance(value, dict) else {"p": value or "", "l": "", "c": []}
    return meta
