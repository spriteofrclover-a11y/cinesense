"""Command line entry point.

    python -m cinesense build-catalog [--force] [--dataset SLUG]
    python -m cinesense build-features
    python -m cinesense similar --movie "Arrival"
    python -m cinesense eval

Run through ``scripts/dev.ps1`` (or the venv directly); the ``python`` on PATH
is 3.11 and has no pandas.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from cinesense.config import ARTIFACTS_DIR, DEFAULT_CONFIG_PATH, Settings, load_settings
from cinesense.logging_setup import configure_logging

__all__ = ["main"]

log = logging.getLogger(__name__)

#: Datasets we know how to fetch. Keeping these here means `--dataset` can't
#: invent a URL that nothing verifies.
KNOWN_DATASETS: dict[str, tuple[str, str]] = {
    "ml-32m": ("MovieLens 32M", "https://files.grouplens.org/datasets/movielens/ml-32m.zip"),
    "ml-25m": ("MovieLens 25M", "https://files.grouplens.org/datasets/movielens/ml-25m.zip"),
    "ml-latest-small": (
        "MovieLens Latest Small",
        "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip",
    ),
}

CONTENT_DIR = ARTIFACTS_DIR / "content"
COLLAB_DIR = ARTIFACTS_DIR / "collab"


def _apply_dataset_override(settings: Settings, slug: str | None) -> Settings:
    """Swap the dataset without mutating the frozen config graph."""
    from dataclasses import replace

    if not slug or slug == settings.dataset.slug:
        return settings

    if slug not in KNOWN_DATASETS:
        raise SystemExit(
            f"unknown dataset {slug!r}. Choose from: {', '.join(sorted(KNOWN_DATASETS))}"
        )

    name, url = KNOWN_DATASETS[slug]
    return replace(settings, dataset=replace(settings.dataset, slug=slug, name=name, url=url))


# --------------------------------------------------------------------------- #
# Phase 1
# --------------------------------------------------------------------------- #


def cmd_build_catalog(args: argparse.Namespace) -> int:
    from cinesense.data.build import QualityGateError, build_catalog, render_report

    settings = _apply_dataset_override(load_settings(args.config), args.dataset)
    try:
        built = build_catalog(settings, force=args.force, config_path=args.config)
    except QualityGateError as exc:
        print(f"\nBUILD FAILED - {exc}\n", file=sys.stderr)
        return 2

    print(render_report(built, settings))
    return 0


# --------------------------------------------------------------------------- #
# Phase 2
# --------------------------------------------------------------------------- #


def cmd_build_features(args: argparse.Namespace) -> int:
    from cinesense.features.build import build_content, render_content_report

    settings = _apply_dataset_override(load_settings(args.config), args.dataset)
    built = build_content(settings, force=args.force)
    print(render_content_report(built))
    return 0


def cmd_build_model(args: argparse.Namespace) -> int:
    from cinesense.features.collab import build_collab, render_collab_report

    settings = _apply_dataset_override(load_settings(args.config), args.dataset)
    built = build_collab(settings, force=args.force)
    print(render_collab_report(built))
    return 0


def cmd_fetch_posters(args: argparse.Namespace) -> int:
    from cinesense.data.posters import PosterError, enrich_posters

    settings = _apply_dataset_override(load_settings(args.config), args.dataset)
    try:
        built = enrich_posters(settings, force=args.force, limit=args.limit)
    except PosterError as exc:
        print(f"\nPOSTERS UNAVAILABLE - {exc}\n", file=sys.stderr)
        return 2

    width = 62
    print()
    print("-" * width)
    print("  CineSense posters")
    print("-" * width)
    print(f"  {'resolved':<24}{built.found:>{width - 28},}")
    print(f"  {'without a poster':<24}{built.missing:>{width - 28},}")
    print(f"  {'eligible films':<24}{built.total:>{width - 28},}")
    print(f"  {'coverage':<24}{built.found / built.total:>{width - 26}.1%}")
    print(f"  {'elapsed':<24}{built.seconds:>{width - 28}.1f}s")
    print("-" * width)
    print()
    return 0


def _load_index() -> object:
    from cinesense.recommend.content import ContentIndex

    if not (CONTENT_DIR / "vectors.npy").exists():
        raise SystemExit(
            "no content model found. Run `cinesense build-features` first."
        )
    return ContentIndex.load(CONTENT_DIR)


def _resolve_movie(index, query: str) -> int:
    matches = index.search(query, limit=5)
    if not matches:
        raise SystemExit(f"no movie matching {query!r}")
    if len(matches) > 1:
        log.info("interpreted %r as %s", query, matches[0].display)
    return matches[0].movie_id


def cmd_similar(args: argparse.Namespace) -> int:
    index = _load_index()
    movie_id = _resolve_movie(index, args.movie)
    seed = index.meta(movie_id)
    results = index.similar_to(movie_id, k=args.k)

    print()
    print(f"  Because you asked for: {seed.display}  [{', '.join(seed.genres) or 'no genres'}]")
    print("  " + "-" * 74)
    for position, rec in enumerate(results, start=1):
        year = f" ({rec.year})" if rec.year else ""
        print(f"  {position:>2}. {rec.title}{year}   sim={rec.score:.3f}")
        for reason in rec.explanation.reasons:
            print(f"      - {reason}")
    print()
    return 0


def cmd_recommend(args: argparse.Namespace) -> int:
    from cinesense.config import load_settings
    from cinesense.recommend.hybrid import HybridRecommender, PreferenceProfile

    settings = load_settings(args.config)
    index = _load_index()
    ratings: dict[int, float] = {}

    for spec in args.rate or []:
        query, _, value = spec.rpartition("=")
        if not query:
            raise SystemExit(f"--rate expects 'Title=4.5', got {spec!r}")
        ratings[_resolve_movie(index, query)] = float(value)

    if args.user is not None:
        ratings.update(_ratings_for_user(args.user))

    favourites = tuple(_resolve_movie(index, title) for title in (args.like or []))
    prefs = PreferenceProfile(
        favorite_movie_ids=favourites,
        genres=tuple(args.genre or ()),
        min_year=args.from_year,
        max_year=args.to_year,
    )

    if not ratings and prefs.is_empty:
        raise SystemExit(
            "nothing to go on. Give --rate 'Title=4.5', --user ID, or --like 'Title'"
        )

    collab = _load_collab()
    if collab is not None:
        ranker = HybridRecommender(index, collab, settings.hybrid)
    else:
        ranker = None

    if ranker is not None:
        results = ranker.recommend(ratings, k=args.k, prefs=prefs)
        mode = "hybrid (content + collaborative + popularity)"
    else:
        results = index.recommend(ratings, k=args.k)
        mode = "content only (run `build-model` to enable collaborative signals)"

    print()
    if ratings:
        print("  Your ratings:")
        shown = 0
        for movie_id, value in sorted(ratings.items(), key=lambda kv: -kv[1]):
            meta = index.meta(movie_id)
            if meta is None:
                continue
            print(f"    {value:>4.1f}  {meta.display}")
            shown += 1
            if shown >= 8 and len(ratings) > 8:
                print(f"    ... and {len(ratings) - shown} more")
                break
    if prefs.genres or prefs.min_year or prefs.max_year or favourites:
        bits = []
        if favourites:
            bits.append("likes " + ", ".join(index.meta(m).title for m in favourites))
        if prefs.genres:
            bits.append("genres " + ", ".join(prefs.genres))
        if prefs.min_year or prefs.max_year:
            bits.append(f"years {prefs.min_year or 'any'}-{prefs.max_year or 'any'}")
        print("  Preferences: " + "; ".join(bits))

    print(f"  Mode: {mode}")
    print("  " + "-" * 74)
    for position, rec in enumerate(results, start=1):
        year = f" ({rec.year})" if rec.year else ""
        print(f"  {position:>2}. {rec.title}{year}   score={rec.score:.3f}")
        for reason in rec.explanation.reasons:
            print(f"      - {reason}")
        parts = {
            "content": rec.explanation.content,
            "collab": rec.explanation.collaborative,
            "popularity": rec.explanation.popularity,
        }
        breakdown = "  ".join(
            f"{name} {share:.0%}" for name, share in parts.items() if share >= 0.05
        )
        if breakdown:
            print(f"      [{breakdown}]")
    print()
    return 0


def _load_collab():
    from cinesense.recommend.collab import CollabModel

    if not (COLLAB_DIR / "meta.json").exists():
        return None
    try:
        return CollabModel.load(COLLAB_DIR)
    except FileNotFoundError as exc:
        log.warning("collaborative model unavailable: %s", exc)
        return None


def _ratings_for_user(user_id: int) -> dict[int, float]:
    """Load one MovieLens user's ratings, for demoing the collaborative path."""
    import pandas as pd

    from cinesense.config import PROCESSED_DIR

    path = PROCESSED_DIR / load_settings().dataset.slug / "raw" / "ratings.parquet"
    if not path.exists():
        raise SystemExit(f"{path} not found; run build-catalog first")

    frame = pd.read_parquet(path, filters=[("user_id", "==", user_id)])
    if frame.empty:
        raise SystemExit(f"user {user_id} has no ratings")
    return {int(row.movie_id): float(row.rating) for row in frame.itertuples(index=False)}


def cmd_eval(args: argparse.Namespace) -> int:
    if args.ranking:
        from cinesense.features.ranking_eval import evaluate_ranking, render_ranking_report

        settings = _apply_dataset_override(load_settings(args.config), args.dataset)
        print(
            render_ranking_report(
                evaluate_ranking(
                    settings, k=args.k, users=args.users, mmr_lambda=args.mmr
                )
            )
        )
        return 0

    from cinesense.recommend.evaluate import evaluate, render_eval_report

    index = _load_index()
    report = evaluate(index, k=args.k, sample=args.sample)
    print(render_eval_report(report))
    return 0


# --------------------------------------------------------------------------- #
# Misc
# --------------------------------------------------------------------------- #


def cmd_config(args: argparse.Namespace) -> int:
    """Print the resolved configuration - the first thing to check on a bug."""
    from dataclasses import asdict

    settings = _apply_dataset_override(load_settings(args.config), args.dataset)
    print(json.dumps(asdict(settings), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cinesense",
        description="Explainable hybrid movie recommender.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="config TOML")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")

    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build-catalog", help="download, ingest and verify the dataset")
    build.add_argument("--force", action="store_true", help="ignore all caches and rebuild")
    build.add_argument("--dataset", help=f"one of: {', '.join(sorted(KNOWN_DATASETS))}")
    build.set_defaults(func=cmd_build_catalog)

    features = sub.add_parser("build-features", help="build the content model artifacts")
    features.add_argument("--force", action="store_true")
    features.add_argument("--dataset")
    features.set_defaults(func=cmd_build_features)

    model = sub.add_parser("build-model", help="train collaborative factors")
    model.add_argument("--force", action="store_true")
    model.add_argument("--dataset")
    model.set_defaults(func=cmd_build_model)

    posters = sub.add_parser("fetch-posters", help="resolve TMDB poster paths (needs an API key)")
    posters.add_argument("--force", action="store_true")
    posters.add_argument("--limit", type=int, help="stop after N films, for a trial run")
    posters.add_argument("--dataset")
    posters.set_defaults(func=cmd_fetch_posters)

    similar = sub.add_parser("similar", help="movies similar to a given title")
    similar.add_argument("--movie", required=True, help='e.g. "Arrival"')
    similar.add_argument("-k", type=int, default=10)
    similar.set_defaults(func=cmd_similar)

    recommend = sub.add_parser("recommend", help="recommend from ratings or preferences")
    recommend.add_argument("--rate", action="append", metavar="TITLE=STARS")
    recommend.add_argument("--user", type=int, help="a MovieLens user id")
    recommend.add_argument("--like", action="append", metavar="TITLE", help="onboarding favourite")
    recommend.add_argument("--genre", action="append", help="preferred genre")
    recommend.add_argument("--from-year", type=int)
    recommend.add_argument("--to-year", type=int)
    recommend.add_argument("-k", type=int, default=10)
    recommend.set_defaults(func=cmd_recommend)

    ev = sub.add_parser("eval", help="measure model quality")
    ev.add_argument("-k", type=int, default=10)
    ev.add_argument("--sample", type=int, default=500)
    ev.add_argument("--ranking", action="store_true", help="NDCG/precision against held-out likes")
    ev.add_argument("--users", type=int, default=200, help="users to sample for ranking eval")
    ev.add_argument(
        "--mmr",
        type=float,
        default=1.0,
        help="MMR lambda for ranking eval; 1.0 disables diversification (default)",
    )
    ev.add_argument("--dataset")
    ev.set_defaults(func=cmd_eval)

    cfg = sub.add_parser("config", help="print the resolved settings as JSON")
    cfg.add_argument("--dataset")
    cfg.set_defaults(func=cmd_config)

    return parser


def _configure_console() -> None:
    """Force UTF-8 on stdout/stderr.

    Windows consoles default to cp1252, and any non-ASCII character the
    explanations use (a star, an accented title) raises UnicodeEncodeError
    mid-render. "replace" keeps output flowing instead of aborting a
    half-printed list.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def main(argv: list[str] | None = None) -> int:
    _configure_console()
    args = build_parser().parse_args(argv)
    configure_logging(logging.DEBUG if args.verbose else logging.INFO)

    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
