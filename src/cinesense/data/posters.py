"""TMDB poster enrichment. Run once, offline, with a key.

Why fetch rather than call the API at request time: a page of 24 recommendations
would mean 24 API calls per user, each a chance to hit a rate limit, plus a key
that has to live in the deployed environment. Instead we resolve every eligible
film to a *path* here, store it, and let the browser pull images from TMDB's CDN.
Serving then needs no key, makes no API calls, and cannot rate-limit.

Only films that can actually be recommended are fetched (>= ``min_ratings``), and
a blank result is cached as an empty string so a missing poster is never retried.

    $env:TMDB_API_KEY="..."          # or put TMDB_API_KEY=... in .env
    cinesense fetch-posters
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, as_completed, wait
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

from cinesense.config import ARTIFACTS_DIR, PROCESSED_DIR, PROJECT_ROOT, PosterSettings, Settings

__all__ = [
    "PosterError",
    "PosterBuild",
    "api_key",
    "enrich_posters",
    "load_posters",
]

log = logging.getLogger(__name__)

#: Lives under artifacts/, not data/: everything the running app reads must ship
#: in the deployment bundle, and vercel.json excludes data/ as local scratch.
POSTER_DIR = ARTIFACTS_DIR / "posters"
POSTERS_FILE = POSTER_DIR / "posters.json.gz"
POSTER_META = POSTER_DIR / "meta.json"

API_ROOT = "https://api.themoviedb.org/3/movie/"


class PosterError(RuntimeError):
    """Posters could not be fetched."""


#: Stored for a film TMDB does not know, so it is never retried.
_EMPTY: dict = {"p": "", "l": "", "c": [], "v": 0.0, "vc": 0}


@dataclass(slots=True)
class PosterBuild:
    found: int
    missing: int
    total: int
    parquet: Path
    seconds: float


def api_key(settings: PosterSettings) -> str | None:
    """Read the key from the environment, else from a local .env file."""
    value = os.environ.get(settings.api_key_env, "").strip()
    if value:
        return value

    env_file = PROJECT_ROOT / ".env"
    if not env_file.exists():
        return None

    for line in env_file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, _, raw = stripped.partition("=")
        if name.strip() == settings.api_key_env:
            # Strip quotes first, then whitespace again: a pasted
            # `" key "` keeps its inner spaces otherwise, and a stray space
            # makes TMDB reject the credential.
            value = raw.strip().strip('"').strip("'").strip()
            return value or None
    return None


def load_tmdb_meta() -> dict[int, dict]:
    """Full TMDB record per movie: poster path, language, country, votes.

    Gzipped JSON rather than parquet: the serving process reads this file too,
    and it must not drag pandas into the deployed bundle.
    """
    if not POSTERS_FILE.exists():
        return {}
    with gzip.open(POSTERS_FILE, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)

    meta: dict[int, dict] = {}
    for key, value in payload.items():
        # Older builds stored a bare poster path; keep them readable.
        meta[int(key)] = value if isinstance(value, dict) else {"p": value or "", "l": "", "c": [], "v": 0.0, "vc": 0}
    return meta


def load_posters() -> dict[int, str]:
    """movie_id -> TMDB poster path ("" when the film has none)."""
    return {movie_id: record.get("p", "") for movie_id, record in load_tmdb_meta().items()}


def _eligible_movies(settings: Settings) -> pd.DataFrame:
    """Films worth a poster: those with enough ratings to be recommended."""
    links_path = PROCESSED_DIR / settings.dataset.slug / "raw" / "links.parquet"
    ratings_path = PROCESSED_DIR / settings.dataset.slug / "raw" / "ratings.parquet"
    if not links_path.exists() or not ratings_path.exists():
        raise PosterError("run `build-catalog` first; links or ratings are missing")

    counts = (
        pd.read_parquet(ratings_path, columns=["movie_id"])["movie_id"]
        .value_counts()
        .rename("n_ratings")
    )
    links = pd.read_parquet(links_path).dropna(subset=["tmdb_id"])
    links = links.merge(counts, left_on="movie_id", right_index=True, how="inner")
    links = links[links["n_ratings"] >= settings.posters.min_ratings]
    links["tmdb_id"] = links["tmdb_id"].astype("int64")
    return links[["movie_id", "tmdb_id"]].sort_values("movie_id").reset_index(drop=True)


def enrich_posters(
    settings: Settings,
    *,
    force: bool = False,
    limit: int | None = None,
) -> PosterBuild:
    key = api_key(settings.posters)
    if not key:
        raise PosterError(
            f"{settings.posters.api_key_env} is empty. Open "
            f"{PROJECT_ROOT / '.env'} and paste your key after "
            f"{settings.posters.api_key_env}=.  Either the v3 'API Key' or the "
            "v4 'API Read Access Token' works (get one at "
            "https://www.themoviedb.org/settings/api)."
        )

    started = time.perf_counter()
    POSTER_DIR.mkdir(parents=True, exist_ok=True)

    targets = _eligible_movies(settings)
    if limit is not None:
        targets = targets.head(limit)

    known = {} if force else load_tmdb_meta()

    def needs_fetch(movie_id: int) -> bool:
        # An entry without a language predates this schema (it used to store a
        # bare poster path), so it is worth one more call.
        return not known.get(movie_id, {}).get("l")

    pending = [
        (int(row.movie_id), int(row.tmdb_id))
        for row in targets.itertuples(index=False)
        if needs_fetch(int(row.movie_id))
    ]

    log.info(
        "%s films eligible, %s already resolved, %s to fetch",
        f"{len(targets):,}",
        f"{len(known):,}",
        f"{len(pending):,}",
    )

    if not pending:
        return _summary(targets, known, started)

    workers = max(1, int(settings.posters.workers))
    interval = 1.0 / max(settings.posters.requests_per_second, 0.1)

    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=workers, pool_maxsize=workers
    )
    session.mount("https://", adapter)

    # Pre-flight one request synchronously. A bad credential otherwise fills the
    # pool with 19,000 identical 401s before anything reports the real problem.
    total = len(pending)
    first_movie, first_tmdb = pending[0]
    known[first_movie] = _fetch_poster(session, key, first_tmdb)
    pending = pending[1:]
    completed = 1

    log.info("fetching with %d workers ...", workers)
    fatal: PosterError | None = None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        # A bounded in-flight window, drained as it fills. Submitting everything
        # first would work, but nothing is written or reported until the last
        # submission, so a 15-minute run looks frozen and a crash loses all of it.
        inflight: dict[Future, int] = {}
        window = max(workers * 4, 32)

        def record(future: Future) -> None:
            nonlocal completed, fatal
            movie_id = inflight.pop(future)
            try:
                known[movie_id] = future.result()
            except PosterError as exc:
                fatal = exc
                return
            except Exception:  # a transient failure is not worth stopping for
                known[movie_id] = ""

            completed += 1
            if completed % 250 == 0:
                _write(known)
                elapsed = time.perf_counter() - started
                rate = completed / elapsed if elapsed else 0
                log.info("  %s/%s fetched (%.0f/s)", f"{completed:,}", f"{total:,}", rate)

        for movie_id, tmdb_id in pending:
            if fatal is not None:
                break
            inflight[pool.submit(_fetch_poster, session, key, tmdb_id)] = movie_id
            if len(inflight) >= window:
                finished, _ = wait(inflight, return_when=FIRST_COMPLETED)
                for future in finished:
                    record(future)
            time.sleep(interval)

        for future in as_completed(list(inflight)):
            record(future)

    _write(known)
    if fatal is not None:
        raise fatal

    return _summary(targets, known, started)


def _fetch_poster(session: requests.Session, key: str, tmdb_id: int) -> dict:
    """TMDB metadata for one film: poster path, language, country, vote stats."""
    url = f"{API_ROOT}{tmdb_id}"

    # TMDB issues two credentials and they authenticate differently: a v3 API
    # key goes in a query parameter, a v4 read access token is a JWT that goes
    # in an Authorization header. The dashboard shows the v4 token most
    # prominently, so accept either one rather than making the user work out
    # which this code expects.
    params = {"language": "en-US"}
    headers: dict[str, str] = {}
    if key.startswith("eyJ"):
        headers["Authorization"] = f"Bearer {key}"
    else:
        params["api_key"] = key

    for attempt in range(4):
        try:
            response = session.get(url, params=params, headers=headers, timeout=15)
        except requests.RequestException as exc:
            log.debug("request failed for %s: %s", tmdb_id, exc)
            time.sleep(1.5 * (attempt + 1))
            continue

        if response.status_code == 200:
            payload = response.json()
            return {
                "p": payload.get("poster_path") or "",
                "l": payload.get("original_language") or "",
                "c": payload.get("origin_country") or [],
                "v": round(float(payload.get("vote_average") or 0.0), 1),
                "vc": int(payload.get("vote_count") or 0),
            }
        if response.status_code == 404:
            return _EMPTY
        if response.status_code == 401:
            raise PosterError(
                "TMDB rejected the credential (401). Use either the v3 'API Key' "
                "or the v4 'API Read Access Token' from "
                "https://www.themoviedb.org/settings/api"
            )
        if response.status_code == 429:
            log.warning("rate limited; backing off")
            time.sleep(6 * (attempt + 1))
            continue

        log.debug("unexpected %s for %s", response.status_code, tmdb_id)
        time.sleep(1.0 * (attempt + 1))
    return _EMPTY


def _write(known: dict[int, dict]) -> None:
    """Persist as gzipped JSON so the runtime can read it without pandas."""
    with gzip.open(POSTERS_FILE, "wt", encoding="utf-8") as handle:
        json.dump({str(key): value for key, value in known.items()}, handle, separators=(",", ":"))


def _summary(targets: pd.DataFrame, known: dict[int, dict], started: float) -> PosterBuild:
    relevant = {int(mid): known.get(int(mid), _EMPTY) for mid in targets["movie_id"]}
    found = sum(1 for record in relevant.values() if record.get("p"))
    missing = len(relevant) - found
    seconds = time.perf_counter() - started

    POSTER_META.write_text(
        json.dumps(
            {
                "targets": len(relevant),
                "found": found,
                "missing": missing,
                "coverage": round(found / len(relevant), 4) if relevant else 0.0,
                "seconds": round(seconds, 1),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return PosterBuild(
        found=found,
        missing=missing,
        total=len(relevant),
        parquet=POSTERS_FILE,
        seconds=seconds,
    )
