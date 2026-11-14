# CineSense

An explainable hybrid movie recommender: collaborative filtering + content
similarity + user preferences, with a reason attached to every suggestion.

**Dataset:** MovieLens 32M (32,000,204 ratings · 200,948 users · 87,585 movies).
**Deployment target:** Vercel (Hobby, Python runtime).

---

## Status

| Phase | Scope | State |
| --- | --- | --- |
| 1 | Data foundation: download, ingest, catalogue, quality gates | ✅ complete |
| 2 | Content-based engine (TF-IDF + genome, LSA, explanations, eval) | ✅ complete |
| 3 | Collaborative filtering, hybrid ranking, MMR, cold start | ✅ complete |
| 4 | FastAPI backend + web UI | ✅ complete |
| 5 | Deploy to Vercel | ⬜ next |

### Phase 1 results (measured, not estimated)

| Metric | Value |
| --- | --- |
| Ratings ingested | 32,000,204 |
| Users | 200,948 |
| Movies | 87,585 (84,432 rated) |
| Tags applied | 2,000,072 |
| Mean rating | 3.540 |
| Sparsity | 99.82% |
| Genre / tag coverage | 91.9% / 58.6% |
| TMDB id coverage | 99.9% |
| Movies with ≥30 ratings | 19,670 |
| Cold build | **7.9 s**, peak RSS **871 MB** |
| Cache-hit run | **2.6 s** |
| Tests | 149 passing, ~4 s, no network |

### Phase 3 results (measured on a held-out 10% of ratings)

| Metric | Value |
| --- | --- |
| Train / held-out interactions | 28,474,408 / 3,163,302 |
| Factors | 64 (15.7% of variance) |
| RMSE, bias baseline | 0.8543 |
| **RMSE, with factors** | **0.8167 (-4.4%)** |
| Shipped artifacts | **11.4 MB** |
| Training time | 79 s |

Ranking quality against held-out likes (rating ≥ 4), 300 users, k=10:

| variant | NDCG@10 | precision@10 |
| --- | --- | --- |
| popularity only | 0.1183 | 0.0750 |
| content only | 0.0267 | 0.0170 |
| collaborative only | 0.0556 | 0.0343 |
| **hybrid (shipped)** | **0.1317** | **0.0760** |

**+11.3% NDCG over the best single signal.** The content-only row is the most
instructive: it scores *below* popularity, because content similarity answers
"what is this like", not "what will you rate highly". It is the right tool for
"more like this" and the wrong tool for a ranking. The hybrid is what makes both
useful at once.

### Diversification: measured, and mostly not MMR's doing

| MMR lambda | diversity | genres in top-10 | NDCG@10 |
| --- | --- | --- | --- |
| 1.00 (off) | 28.8% | 9.2 | 0.1328 |
| **0.75 (shipped)** | **29.8%** | **9.6** | **0.1317** |
| 0.50 | 32.5% | 10.7 | 0.1262 |
| 0.25 | 37.1% | 13.0 | — |

Phase 2 returned lists at 6.4% diversity. The hybrid lift to 28.8% came from the
collaborative and popularity signals surfacing genuinely different films; **MMR
itself adds only ~1 point at the shipped setting**. It is kept because it costs
0.8% NDCG at 0.75, but the earlier assumption that MMR would be the fix was
wrong, and measuring it is what showed that.

### Phase 2 results (measured)

| Metric | Value |
| --- | --- |
| Text vocabularies | title 35,101 · genres 19 · tags 17,380 |
| Genome block | 1,128 scores · 13,812 movies · 15.6M nonzeros |
| Latent dimensions | 96 (LSA, 49.0% variance kept) |
| Shipped artifacts | **19.5 MB** (vectors 16.0 · catalogue 3.1 · ids 0.3) |
| Genre agreement@10 | 0.954 vs 0.141 random → **6.75× lift** |
| Intra-list diversity | 6.4% (random sets: 86.9%) — see limitation below |
| Feature build | 50 s excluding genome download, 238 s cold |

### Is the tag genome worth it? Yes — 2.4× on an independent metric

ml-32m does **not** ship the genome, so this required measuring a second archive.
Genre agreement turned out to be a biased metric (a genre-based model wins it by
construction), so the decision rests on **co-rating similarity**: do the same
users rate a film and its neighbours alike? That signal is independent of every
feature the model is given.

| variant | genre (biased) | co-rating (honest) |
| --- | --- | --- |
| genres only | 0.964 | 0.0110 |
| text (no genome) | 0.939 | 0.0228 |
| **text + genome ×2.0** | 0.555 | **0.0557** |
| random baseline | 0.170 | 0.0008 |

Genome-aware neighbours are rated similarly by the same users **2.4× more
often**, and the genre score falling is what a biased metric *should* do when
the model stops optimising for it. Coverage is 15.8% of the catalogue but
**70.2% of movies with ≥30 ratings** — the ones that actually get recommended.

**Known limitation (addressed in Phase 3):** content similarity alone is
precision-oriented, so a top-10 list was near-duplicates (mean pairwise cosine
0.936). The hybrid blend fixed this — see below.

---

## Setup

> **The `python` on PATH is 3.11 and cannot import pandas.** Always use the
> virtual environment. `scripts/dev.ps1` does this for you.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

Python 3.12 is deliberate: it matches the Vercel runtime. Python 3.15 is
installed locally but has no wheels for several packages (pyyaml failed to build
on the first attempt), so it is not used here.

## Commands

```powershell
.\scripts\dev.ps1 build-catalog      # download + ingest + verify (~2-4 min first run)
.\scripts\dev.ps1 build-features     # text + genome -> content vectors (~4 min cold)
.\scripts\dev.ps1 build-model        # train collaborative factors (~80 s)
.\scripts\dev.ps1 fetch-posters      # TMDB poster paths (needs a key, see below)
.\scripts\dev.ps1 config             # print resolved settings
.\scripts\dev.ps1 build-catalog --dataset ml-latest-small   # 1 MB, instant, for iteration
.\scripts\dev.ps1 build-catalog --force                     # ignore every cache

.\scripts\dev.ps1 similar --movie "The Godfather" -k 8
.\scripts\dev.ps1 recommend --user 1 -k 8
.\scripts\dev.ps1 recommend --rate "Inception=5" --rate "The Matrix=1" -k 8
.\scripts\dev.ps1 recommend --like "Inception" --genre "Sci-Fi" -k 8   # cold start
.\scripts\dev.ps1 eval -k 10 --sample 400                # content: genre agreement
.\scripts\dev.ps1 eval --ranking -k 10 --users 300       # NDCG / precision

# Serve the app
.\.venv\Scripts\python.exe -m uvicorn cinesense.serving.app:app --port 8011
# then open http://127.0.0.1:8011

.\.venv\Scripts\python.exe -m pytest             # 163 tests, no network
.\.venv\Scripts\python.exe scripts\check_ui.py   # 16 browser checks (needs the server)
```

### Posters

Real posters need one free TMDB key, used **once, offline**:

```powershell
# put TMDB_API_KEY=your_key in .env, then:
.\scripts\dev.ps1 fetch-posters
```

Measured on this dataset: **19,418 of 19,652 eligible films resolved (98.8%)** in
9.3 minutes; 234 films have no poster on TMDB and are stored as blank so they are
never retried.

The fetch is concurrent (20 workers, throttled to 40 req/s). A sequential loop
ran at 2 req/s because ~470 ms of network latency dominated, which would have
taken two hours — worth knowing if the rate limit ever needs tuning.

Both TMDB credential shapes work: the v3 API key (query parameter) and the v4
read access token (bearer header). Only the image *path* is stored — the browser
loads from TMDB's CDN, so serving needs no key and makes no API calls. Without a
key the UI falls back to generated monogram cards, so nothing is ever blank.

### Regional coverage

MovieLens 32M is overwhelmingly Western: films whose tags mention India or an
Indian language hold **0.28% of all 32M ratings**. Two things kept them out of
the UI, and both are fixed:

1. **The eligibility gate.** It used the collaborative *training* counts, which
   only include films with 30+ ratings, so anything below that was excluded
   outright rather than merely ranked low. It is now "at least one rating", and
   popularity counts every rating rather than the training subset.
2. **No way to ask.** TMDB enrichment now captures `original_language` and
   `origin_country` alongside posters, exposed as `GET /api/facets` and as
   `prefs.languages` / `prefs.regions` filters. Discover shows a LANGUAGE row
   led by *Indian cinema*.

Reachable Indian films went from **158 to 257** at the 30-rating metadata
threshold, and rise further once films down to 5 ratings are enriched.

**What this does not do:** films absent from MovieLens (KGF, Salaar, Animal,
Leo, Premalu…) cannot be recommended — no ratings means no collaborative signal
and no tag genome. Adding them is option B, a separate decision.

`build-catalog` is idempotent: the second run reuses the archive, the extracted
tree and every parquet file, then rewrites only the manifest.

## What Phase 1 produces

```
data/raw/ml-32m/            # archive + extracted CSVs (.gitignored, ~1.2 GB)
data/processed/ml-32m/
    raw/                    # exactly what the CSVs said, typed
        movies.parquet          1.3 MB
        ratings.parquet       151.5 MB
        tags.parquet           12.4 MB
        links.parquet           0.8 MB
    movies.parquet          # enriched catalogue, 12.5 MB
    tags_by_movie.parquet   # per-movie tags + tag counts, 4.2 MB
artifacts/manifest.json     # provenance: sizes, row counts, sha256, config, gates
```

`raw/` and the derived tables are deliberately separate directories. They once
shared one, so a cache hit served the un-enriched table as if it were enriched.

Catalogue columns: `movie_id, title, year, genres, n_genres, tag_text, n_tags,
search_text`.

## What Phase 4 produces

```
src/cinesense/serving/
    app.py          FastAPI factory; models load once in lifespan
    state.py        ModelBundle: content + collab + posters, one instance
    routes.py       all 12 endpoints
    schemas.py      pydantic request/response models
    static/         index.html, css/app.css, js/{app,api,store,ui,views}.js
api/index.py        Vercel entrypoint
```

| Endpoint | Purpose |
| --- | --- |
| `GET /api/health` | liveness and which models loaded |
| `GET /api/stats` | dataset and model metrics for the About page |
| `GET /api/genres` | genre facets with counts |
| `GET /api/search?q=` | title search |
| `GET /api/movies/{id}` | metadata + top terms |
| `POST /api/movies` | bulk metadata lookup |
| `GET /api/movies/{id}/similar` | content neighbours |
| `GET /api/movies/{id}/related` | collaborative neighbours |
| `POST /api/recommendations` | hybrid results with explanations |

Screens: **Discover** (taste fingerprint + grid + genre filters), **Movie**
(rating, why-panel with contribution bars, two rails), **Your taste** (genre
distribution, signal mix, rating manager), **How it works** (live metrics), and a
three-step **onboarding**. Ratings live in `localStorage`; the API is stateless.

Design: ink-black canvas, neutral chrome, one amber accent, posters are the only
saturated colour. Serif display against a neutral sans. No build step, no
webfonts — system fonts only, so nothing flashes and nothing blocks.

## Design decisions worth knowing

**duckdb, not `pandas.read_csv`.** 32M rows through pandas spikes several GB.
duckdb streams the CSVs into ZSTD parquet inside a fixed memory ceiling, and
every later stage reads the parquet.

**Quality gates fail the build.** If ratings collapse, sparsity goes weird, or
TMDB id coverage drops, `build-catalog` exits non-zero instead of producing a
quietly worse model.

**Thresholds are reported, not applied.** The report shows how many movies and
users survive `min_movie_ratings` / `min_user_ratings`; Phase 3 actually prunes.

**Runtime dependencies are deliberately tiny.** `requirements.txt` holds only
`fastapi`, `numpy`, `pydantic`. pandas/scipy/sklearn are offline-only, so the
deployed function bundle stays small and cold starts stay short.
`tests/test_dependency_split.py` enforces this across both `serving/` and
`recommend/`.

**Only item factors ship.** Storing 200,948 user factors would cost ~51 MB and
still could not serve anyone outside MovieLens. Instead the runtime folds a
user's ratings into a vector at request time (`x_u = residual_u @ P`) — same
algebra, 11 MB, and it works for a brand-new user with three ratings.

**Signals are z-scored before blending.** Content similarity lives in [-1, 1], a
collaborative prediction in [0.5, 5], popularity is a count. Adding them raw
would let whichever has the biggest numbers win regardless of configured weights.

**The evaluated model is the shipped model.** 10% of ratings are held out before
training, and every number reported — RMSE, NDCG — comes from that held-out set.
Training a second "production" model on all the data would look better and prove
nothing.

**LSA exists for size, not quality.** The sparse text matrix is 17.9 GB as dense;
96 latent dimensions ship as 16 MB float16. With the genome block appended the
input grows to 17.4M nonzeros but the output stays the same size.

**No precomputed neighbour table.** Scoring 87,585 movies against one 96-dim
vector is ~8M multiply-adds — milliseconds — and costs 16 MB of vectors instead
of ~35 MB of stored neighbours that would go stale on every rebuild.

**Dislikes exclude, they do not anti-recommend.** Mean-centring ratings meant a
1-star review *subtracted* that film's vector, pushing the profile away from
Sci-Fi and Action — nearly the whole catalogue — and surfacing obscure
Crime/Drama. Ratings below `NEUTRAL_RATING` now carry no weight; they are simply
excluded from results.

**Title search canonicalises MovieLens articles.** "The Godfather" is stored as
`Godfather, The (1972)`, so the leading article is moved back before matching;
exact matches then beat similar names by rating count.

**Tags are stored whole, not truncated.** An earlier version capped tag text at
400 characters for the search index. Because `tag_text` is alphabetically
ordered, that silently kept only tags starting with "a".."d" on popular films.
Vocabulary growth is instead handled by the vectoriser's `min_df` in Phase 2.
`n_tags` records how many tags a film has, which the joined string cannot
express (tags contain spaces).

## Layout

```
configs/base.toml          all thresholds and weights (no magic numbers in code)
src/cinesense/
    config.py              typed settings; rejects unknown keys
    logging_setup.py
    cli.py                 `python -m cinesense ...`
    data/                  OFFLINE ONLY (pandas, duckdb, sklearn)
        download.py        resumable, checksum-verified fetch
        ingest.py          CSV -> typed parquet via duckdb
        preprocess.py      title/year/genre/tag normalisation
        build.py           the pipeline + quality gates
        genome.py          tag genome fetch + alignment
        manifest.py        build provenance
    features/              OFFLINE ONLY
        text.py            per-field TF-IDF + LSA + top terms
        build.py           writes the content artifacts
        collab.py          bias fit + SVD factorisation + RMSE
        ranking_eval.py    NDCG/precision against held-out likes
    recommend/             RUNTIME: numpy + stdlib only
        contracts.py       Explanation / Recommendation
        content.py         ContentIndex: similarity, fold-in, search
        collab.py          CollabModel: fold-in, scoring
        hybrid.py          signal blend, MMR, cold start
        explain.py         human-readable reasons
        evaluate.py        genre agreement, diversity, coverage
tests/                     fixtures only; no network, no 32M rows
scripts/                   dev.ps1, measure_build.py, measure_genome.py, measure_mmr.py
artifacts/content/         vectors.npy, movie_ids.npy, catalog.json.gz, manifest.json
artifacts/collab/          item_factors.npy, item_bias.npy, item_popularity.npy, meta.json
artifacts/posters/         posters.json.gz (created by fetch-posters)
artifacts/eval/            ranking.json (written by eval --ranking)
```
