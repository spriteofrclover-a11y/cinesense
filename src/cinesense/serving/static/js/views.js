// Views. Each one renders into #view and owns its own loading and error states.

import { api } from "./api.js";
import { store } from "./store.js";
import {
  cardNode,
  cardReason,
  clear,
  contribBars,
  errorBox,
  h,
  posterNode,
  ratingLabel,
  skeletonGrid,
  starsNode,
  toast,
} from "./ui.js";

export function navigate(hash) {
  location.hash = hash;
}

const openMovie = (id) => navigate(`#/movie/${id}`);

function page(...children) {
  return h("div", { class: "page-enter" }, children);
}

function sectionHeading(title, note) {
  return h(
    "div",
    { class: "section-head" },
    h("div", {}, h("h2", {}, title)),
    note ? h("p", {}, note) : null,
  );
}

// ---------------------------------------------------------------------------
// Discover
// ---------------------------------------------------------------------------

export async function renderDiscover(root) {
  const ratings = store.ratings();
  const hasRatings = Object.keys(ratings).length > 0;
  const active = { genre: null, language: null, region: null };
  let recs = [];
  let allRecs = [];
  let genreFacets = [];
  let languageFacets = [];
  let regionFacets = [];

  clear(root);
  root.append(skeletonGrid(12));

  try {
    const ids = Object.keys(ratings).map(Number);
    const [library, facets, facetOptions, initial] = await Promise.all([
      ids.length ? api.moviesByIds(ids) : Promise.resolve([]),
      api.genres(),
      api.facets(),
      api.recommend({
        ratings,
        k: 24,
        prefs: hasRatings ? null : (store.prefs().genres?.length ? { genres: store.prefs().genres } : null),
      }),
    ]);
    genreFacets = facets.slice(0, 8);

    // The catalogue is overwhelmingly Western, so the language row leads with
    // Indian cinema rather than listing every language the dataset happens to hold.
    regionFacets = (facetOptions.regions || []).filter((region) => region.code === "IN");
    const INDIAN_LANGUAGES = ["hi", "ta", "te", "ml", "kn", "bn", "mr", "pa"];
    languageFacets = (facetOptions.languages || [])
      .filter((language) => INDIAN_LANGUAGES.includes(language.code))
      .slice(0, 8);
    allRecs = initial;
    recs = initial;
    clear(root);

    if (hasRatings) {
      root.append(fingerprint(library));
    } else {
      root.append(welcome());
    }

    const gridHost = h("div");

    const drawGrid = (items) => {
      clear(gridHost);
      if (!items.length) {
        gridHost.append(
          h(
            "div",
            { class: "empty" },
            h("h2", {}, "Nothing matches that filter"),
            h("p", {}, "Try another genre or language, or clear the filters."),
            h("button", { class: "btn btn-sm", type: "button", onclick: () => clearFilters() }, "Clear filters"),
          ),
        );
        return;
      }
      gridHost.append(
        h(
          "div",
          { class: "grid" },
          items.map((movie, index) =>
            cardNode(movie, {
              onOpen: openMovie,
              why: cardReason(movie),
              delay: Math.min(index, 11) * 28,
            }),
          ),
        ),
      );
    };

    // Genre, language and region all narrow the same pool, and they combine.
    const buildPrefs = () => {
      const prefs = {};
      if (active.genre) prefs.genres = [active.genre];
      if (active.language) prefs.languages = [active.language];
      if (active.region) prefs.regions = [active.region];
      return Object.keys(prefs).length ? prefs : null;
    };

    const clearFilters = () => {
      active.genre = null;
      active.language = null;
      active.region = null;
      root.querySelectorAll(".chip[data-kind]").forEach((chip) =>
        chip.setAttribute("aria-pressed", "false"),
      );
      recs = allRecs;
      drawGrid(recs);
    };

    const applyFilter = async (kind, value) => {
      const sameValue = active[kind] === value;

      // Language and region are two views of the same thing, so picking one
      // clears the other. Combined as a union, "Indian cinema" + "Kannada" is
      // simply "Indian cinema" and the second click appears to do nothing.
      if (kind === "language" || kind === "region") {
        active.language = null;
        active.region = null;
      }
      active[kind] = sameValue ? null : value;
      root.querySelectorAll(".chip[data-kind]").forEach((chip) => {
        const on = active[chip.dataset.kind] === chip.dataset.value;
        chip.setAttribute("aria-pressed", on ? "true" : "false");
      });

      const prefs = buildPrefs();
      if (!prefs) {
        recs = allRecs;
        drawGrid(recs);
        return;
      }
      gridHost.replaceChildren(skeletonGrid(8));
      recs = await api.recommend({ ratings, k: 24, prefs });
      drawGrid(recs);
    };

    const chips = h(
      "div",
      { class: "filters", role: "group", "aria-label": "Filter by genre" },
      genreFacets.map((genre) =>
        h(
          "button",
          {
            class: "chip",
            type: "button",
            "data-kind": "genre",
            "data-value": genre.name,
            "aria-pressed": "false",
            onclick: () => applyFilter("genre", genre.name),
          },
          `${genre.name} · ${genre.count}`,
        ),
      ),
      h(
        "button",
        {
          class: "chip",
          type: "button",
          onclick: async () => {
            const exclude = recs.map((movie) => movie.id);
            gridHost.replaceChildren(skeletonGrid(8));
            recs = await api.recommend({
              ratings,
              k: 24,
              exclude,
              prefs: buildPrefs(),
            });
            drawGrid(recs);
          },
        },
        "Surprise me",
      ),
    );

    const languageRow =
      regionFacets.length || languageFacets.length
        ? h(
            "div",
            { class: "filters", role: "group", "aria-label": "Filter by language" },
            h("span", { class: "filter-label" }, "Language"),
            regionFacets.map((region) =>
              h(
                "button",
                {
                  class: "chip chip-accent",
                  type: "button",
                  "data-kind": "region",
                  "data-value": region.code,
                  "aria-pressed": "false",
                  onclick: () => applyFilter("region", region.code),
                },
                `${region.name} · ${region.count}`,
              ),
            ),
            languageFacets.map((language) =>
              h(
                "button",
                {
                  class: "chip",
                  type: "button",
                  "data-kind": "language",
                  "data-value": language.code,
                  "aria-pressed": "false",
                  onclick: () => applyFilter("language", language.code),
                },
                `${language.name} · ${language.count}`,
              ),
            ),
          )
        : null;

    root.append(
      h(
        "section",
        {},
        sectionHeading(
          hasRatings ? "Picked for you" : "Popular right now",
          hasRatings
            ? `${recs.length} films ranked by your taste, the crowd's taste, and what is simply good`
            : "Rate a few films and this list becomes personal",
        ),
        chips,
        languageRow,
        gridHost,
      ),
    );

    drawGrid(recs);
  } catch (error) {
    clear(root);
    root.append(page(errorBox(error, () => renderDiscover(root))));
  }
}

function welcome() {
  return page(
    h(
      "div",
      { class: "fingerprint" },
      h(
        "div",
        {},
        h("p", { class: "eyebrow" }, "Start here"),
        h("h1", {}, "Rate five films you love. Get a taste profile."),
        h(
          "p",
          { class: "lede" },
          "CineSense blends what you rate with how similar films look and what people with " +
            "your taste rated highly — and it tells you which of the three drove each suggestion.",
        ),
        h(
          "button",
          { class: "btn btn-primary", type: "button", onclick: () => navigate("#/start") },
          "Build my taste profile",
        ),
      ),
      h(
        "div",
        { class: "fingerprint-figures" },
        figure("87,585", "films"),
        figure("32M", "ratings"),
        figure("3", "signals"),
      ),
    ),
  );
}

function figure(value, label) {
  return h("div", { class: "fingerprint-figure" }, h("b", {}, value), h("span", {}, label));
}

function fingerprint(library) {
  const byId = Object.fromEntries(library.map((movie) => [movie.id, movie]));
  const profile = store.genreProfile(byId).slice(0, 5);
  const top = profile[0]?.[0] || "Film";
  const ratings = Object.values(store.ratings());
  const rated = Object.keys(store.ratings()).length;
  const average = ratings.length
    ? (ratings.reduce((sum, value) => sum + value, 0) / ratings.length).toFixed(2)
    : "—";

  // One rated film makes every genre bar the same height, which reads as a
  // broken chart. Wait until the picture is actually a picture.
  const enough = rated >= 3 && profile.length > 1;

  return page(
    h(
      "div",
      { class: "fingerprint" },
      h(
        "div",
        {},
        h("p", { class: "eyebrow" }, "Your taste"),
        h("h1", {}, `Mostly ${top.toLowerCase()}, apparently.`),
        h(
          "p",
          { class: "lede" },
          "Built from what you have rated so far. Every suggestion below is scored against this, " +
            "not against what is merely popular.",
        ),
        enough
          ? h(
              "div",
              { class: "genre-bars" },
              profile.map(([genre, count], index) => {
                const max = profile[0][1] || 1;
                return h(
                  "div",
                  { class: "genre-bar" },
                  h("span", { class: "genre-bar-name" }, genre),
                  h(
                    "span",
                    { class: "genre-bar-track" },
                    h("span", {
                      class: "genre-bar-fill",
                      style: `width:${Math.max(8, (count / max) * 100)}%;animation-delay:${index * 70}ms`,
                    }),
                  ),
                  h("span", { class: "genre-bar-count" }, String(count)),
                );
              }),
            )
          : h(
              "p",
              { class: "lede", style: "margin:0;color:var(--muted-dim)" },
              `Rate ${3 - rated > 1 ? `${3 - rated} more films` : "one more film"} and this becomes a profile.`,
            ),
      ),
      h(
        "div",
        { class: "fingerprint-figures" },
        figure(String(rated), "rated"),
        figure(average, "avg score"),
        figure(String(store.genreCount(byId)), "genres"),
      ),
    ),
  );
}

// ---------------------------------------------------------------------------
// Movie detail
// ---------------------------------------------------------------------------

export async function renderMovie(root, movieId) {
  clear(root);
  root.append(h("div", { class: "skeleton", style: "height:380px" }));

  try {
    const [movie, similar, related] = await Promise.all([
      api.movie(movieId),
      api.similar(movieId, 12),
      api.related(movieId, 12),
    ]);

    clear(root);
    const whyHost = h("div");

    const drawWhy = async () => {
      clear(whyHost);
      if (!store.count()) {
        whyHost.append(
          h(
            "div",
            { class: "why" },
            h("h3", {}, "Why this might suit you"),
            h(
              "p",
              { class: "lede", style: "margin:0" },
              "Rate a few films and this panel explains which signals put this one in front of you.",
            ),
          ),
        );
        return;
      }

      whyHost.append(h("div", { class: "skeleton", style: "height:120px" }));
      try {
        // A film is excluded from its own recommendation list, so ask what would
        // have been suggested had the user not rated it. Otherwise every rated
        // film falls through to "not in your top 60", which is exactly the case
        // where the explanation is most useful.
        const ratings = { ...store.ratings() };
        delete ratings[movieId];

        const recs = await api.recommend({ ratings, k: 60 });
        const match = recs.find((rec) => rec.id === movieId);
        clear(whyHost);

        if (!match) {
          whyHost.append(
            h(
              "div",
              { class: "why" },
              h("h3", {}, "Not in your current top 60"),
              h(
                "p",
                { class: "lede", style: "margin:0" },
                store.rating(movieId)
                  ? "Even without your own score, your profile does not rank this highly. Rate a few films in this vein and it will climb."
                  : "Your ratings point elsewhere right now. Rate it yourself and the profile adapts.",
              ),
            ),
          );
          return;
        }

        whyHost.append(
          h(
            "div",
            { class: "why" },
            h("h3", {}, "Why this might suit you"),
            h("ul", {}, match.explanation.reasons.map((reason) => h("li", {}, reason))),
            match.explanation.matched_terms.length
              ? h(
                  "div",
                  { class: "genres", style: "margin-bottom:14px" },
                  match.explanation.matched_terms.map((term) =>
                    h("span", { class: "genre-tag" }, term),
                  ),
                )
              : null,
            contribBars(match.explanation),
          ),
        );
      } catch (error) {
        clear(whyHost);
        whyHost.append(errorBox(error));
      }
    };

    const current = store.rating(movieId);
    const ratingNote = h("span", { class: "rating-note" }, ratingLabel(current));

    const stars = starsNode(current, (value) => {
      store.setRating(movieId, value);
      toast(value ? `Rated ${value.toFixed(1)}` : "Rating removed");
      ratingNote.textContent = ratingLabel(value);
      drawWhy();
    });

    root.append(
      page(
        h(
          "div",
          { class: "detail" },
          h("div", { class: "detail-poster" }, posterNode(movie, { lazy: false })),
          h(
            "div",
            {},
            h("h1", { class: "detail-title" }, movie.title),
            h(
              "p",
              { class: "detail-sub" },
              [movie.year, movie.genres.join(", ")].filter(Boolean).join(" · ") || "No metadata",
            ),
            h(
              "ul",
              { class: "facts" },
              h("li", {}, h("b", {}, movie.avg_rating ? movie.avg_rating.toFixed(2) : "—"), h("span", {}, "average")),
              h("li", {}, h("b", {}, movie.n_ratings.toLocaleString()), h("span", {}, "ratings")),
              h("li", {}, h("b", {}, store.rating(movieId) ? `${store.rating(movieId)}` : "—"), h("span", {}, "your score")),
            ),
            movie.genres.length
              ? h("div", { class: "genres" }, movie.genres.map((genre) => h("span", { class: "genre-tag" }, genre)))
              : null,
            h(
              "div",
              { class: "rating-block" },
              h("div", { class: "rating-label" }, "Your rating"),
              h("div", { style: "display:flex;align-items:center" }, stars, ratingNote),
            ),
            h(
              "button",
              {
                class: "btn btn-sm btn-ghost",
                type: "button",
                onclick: () => navigate("#/"),
              },
              "Back to discover",
            ),
          ),
        ),
        h("section", { class: "section" }, whyHost),
        similar.length
          ? h(
              "section",
              { class: "section" },
              sectionHeading("More like this", "Content similarity: genre, tags and the tag genome"),
              h("div", { class: "rail" }, similar.map((rec) => cardNode(rec, { onOpen: openMovie }))),
            )
          : null,
        related.length
          ? h(
              "section",
              { class: "section" },
              sectionHeading("People with your taste also rated", "Collaborative filtering: shared rating patterns"),
              h("div", { class: "rail" }, related.map((rec) => cardNode(rec, { onOpen: openMovie }))),
            )
          : null,
      ),
    );

    drawWhy();
  } catch (error) {
    clear(root);
    root.append(page(errorBox(error, () => renderMovie(root, movieId))));
  }
}

// ---------------------------------------------------------------------------
// Taste
// ---------------------------------------------------------------------------

export async function renderTaste(root, { silent = false } = {}) {
  // `silent` re-renders after an edit without flashing skeletons over content
  // the user is looking at.
  if (!silent) {
    clear(root);
    root.append(skeletonGrid(6));
  }

  try {
    const ratings = store.ratings();
    const ids = Object.keys(ratings).map(Number);
    const library = ids.length ? await api.moviesByIds(ids) : [];
    const byId = Object.fromEntries(library.map((movie) => [movie.id, movie]));
    const profile = store.genreProfile(byId);

    clear(root);

    if (!ids.length) {
      root.append(
        page(
          h(
            "div",
            { class: "empty" },
            h("h2", {}, "You have not rated anything yet"),
            h("p", {}, "Rating films is the only way the recommender learns what you like."),
            h(
              "button",
              { class: "btn btn-primary", type: "button", onclick: () => navigate("#/start") },
              "Rate your first films",
            ),
          ),
        ),
      );
      return;
    }

    const recs = await api.recommend({ ratings, k: 24 });
    const mix = recs.reduce(
      (acc, rec) => ({
        collaborative: acc.collaborative + rec.explanation.collaborative,
        content: acc.content + rec.explanation.content,
        popularity: acc.popularity + rec.explanation.popularity,
      }),
      { collaborative: 0, content: 0, popularity: 0 },
    );
    const total = recs.length || 1;
    const average = (Object.values(ratings).reduce((a, b) => a + b, 0) / ids.length).toFixed(2);

    root.append(
      page(
        h("p", { class: "eyebrow" }, "Your taste"),
        h("h1", {}, `${ids.length} films rated.`),
        h(
          "p",
          { class: "lede" },
          "Everything here is computed in your browser or from the ratings you send with each request. " +
            "The server keeps nothing.",
        ),
        h(
          "ul",
          { class: "facts", style: "margin-top:24px" },
          h("li", {}, h("b", {}, String(ids.length)), h("span", {}, "rated")),
          h("li", {}, h("b", {}, average), h("span", {}, "average")),
          h("li", {}, h("b", {}, profile[0]?.[0] || "—"), h("span", {}, "top genre")),
        ),
        profile.length
          ? h(
              "section",
              { class: "section" },
              sectionHeading("Where your ratings concentrate", "Weighted by the score you gave"),
              h(
                "div",
                { class: "genre-bars" },
                profile.slice(0, 8).map(([genre, count], index) => {
                  const max = profile[0][1] || 1;
                  return h(
                    "div",
                    { class: "genre-bar" },
                    h("span", { class: "genre-bar-name" }, genre),
                    h(
                      "span",
                      { class: "genre-bar-track" },
                      h("span", {
                        class: "genre-bar-fill",
                        style: `width:${Math.max(8, (count / max) * 100)}%;animation-delay:${index * 60}ms`,
                      }),
                    ),
                    h("span", { class: "genre-bar-count" }, String(count)),
                  );
                }),
              ),
            )
          : null,
        h(
          "section",
          { class: "section" },
          sectionHeading("What is driving your recommendations", "Average across your current top picks"),
          contribBars({
            collaborative: mix.collaborative / total,
            content: mix.content / total,
            popularity: mix.popularity / total,
          }),
        ),
        h(
          "section",
          { class: "section" },
          sectionHeading("Your ratings", "Change any score, or remove it"),
          h(
            "div",
            { class: "rows" },
            Object.entries(ratings)
              .sort((a, b) => b[1] - a[1])
              .map(([id, value]) => {
                const movie = byId[id];
                return h(
                  "div",
                  { class: "row-item rating-row" },
                  h(
                    "div",
                    {},
                    h("div", { class: "row-title" }, movie?.title || `Film ${id}`),
                    h(
                      "div",
                      { class: "row-meta" },
                      [movie?.year, (movie?.genres || []).slice(0, 2).join(", ")].filter(Boolean).join(" · "),
                    ),
                  ),
                  // Two numbers, deliberately: the crowd's average for context,
                  // and your own score, which is the only one the model uses.
                  h(
                    "div",
                    { class: "rating-cell" },
                    movie && movie.avg_rating
                      ? h("span", { class: "row-site" }, `avg ${movie.avg_rating.toFixed(2)}`)
                      : null,
                    starsNode(
                      Number(value),
                      (next) => {
                        store.setRating(Number(id), next);
                        toast(next ? `Rated ${next.toFixed(1)}` : "Rating removed");
                        renderTaste(root, { silent: true });
                      },
                      { label: `Your rating for ${movie?.title || id}` },
                    ),
                  ),
                  h(
                    "button",
                    {
                      class: "btn btn-sm btn-ghost",
                      type: "button",
                      onclick: () => {
                        store.removeRating(Number(id));
                        renderTaste(root);
                      },
                    },
                    "Remove",
                  ),
                );
              }),
          ),
          h(
            "div",
            { class: "actions" },
            h(
              "button",
              {
                class: "btn btn-sm",
                type: "button",
                onclick: () => {
                  if (confirm("Remove every rating? This cannot be undone.")) {
                    store.clear();
                    renderTaste(root);
                  }
                },
              },
              "Clear all ratings",
            ),
          ),
        ),
      ),
    );
  } catch (error) {
    clear(root);
    root.append(page(errorBox(error, () => renderTaste(root))));
  }
}

// ---------------------------------------------------------------------------
// About
// ---------------------------------------------------------------------------

export async function renderAbout(root) {
  clear(root);
  root.append(h("div", { class: "skeleton", style: "height:320px" }));

  try {
    const stats = await api.stats();
    clear(root);

    const collab = stats.collaborative || {};
    const content = stats.content || {};
    const ranking = stats.ranking;

    const rows = ranking
      ? Object.keys(ranking.ndcg).map((name) =>
          h(
            "tr",
            { "data-highlight": name === "hybrid" ? "true" : "false" },
            h("td", {}, name === "hybrid" ? `${name} (shipped)` : name),
            h("td", { class: "num" }, ranking.ndcg[name].toFixed(4)),
            h("td", { class: "num" }, ranking.precision[name].toFixed(4)),
          ),
        )
      : [];

    root.append(
      page(
        h(
          "div",
          { class: "narrow" },
          h("p", { class: "eyebrow" }, "How it works"),
        h("h1", {}, "Three signals, one ranking, and a reason for each film."),
        h(
          "div",
          { class: "prose" },
          h(
            "p",
            {},
            "Most recommenders are one trick. CineSense blends three: what you rate, what similar films " +
              "look like, and what people with your taste rated highly. Each row below is measured on " +
              "ratings the model never trained on.",
          ),
        ),
        h(
          "section",
          { class: "section" },
          sectionHeading("Ranking quality", `NDCG and precision at ${ranking?.k ?? 10}, ${ranking?.users ?? "—"} held-out users`),
          ranking
            ? h(
                "table",
                { class: "metrics" },
                h("thead", {}, h("tr", {}, h("th", {}, "Signal"), h("th", { class: "num" }, "NDCG"), h("th", { class: "num" }, "Precision"))),
                h("tbody", {}, rows),
              )
            : h("p", { class: "lede" }, "Run `cinesense eval --ranking` to populate this table."),
          h(
            "p",
            { class: "lede", style: "margin-top:14px" },
            "Content similarity alone scores below popularity: it answers “what is this like”, not " +
              "“what will I rate highly”. The blend is what makes it useful.",
          ),
        ),
        h(
          "section",
          { class: "section" },
          sectionHeading("The model"),
          h(
            "table",
            { class: "metrics" },
            h(
              "tbody",
              {},
              h("tr", {}, h("td", {}, "Films"), h("td", { class: "num" }, (stats.dataset.movies || 0).toLocaleString())),
              h("tr", {}, h("td", {}, "Users in training"), h("td", { class: "num" }, (stats.dataset.users || 0).toLocaleString())),
              h("tr", {}, h("td", {}, "Content dimensions"), h("td", { class: "num" }, String(stats.dataset.dimensions ?? "—"))),
              h("tr", {}, h("td", {}, "Collaborative factors"), h("td", { class: "num" }, String(collab.factors ?? "—"))),
              h("tr", {}, h("td", {}, "RMSE (bias baseline)"), h("td", { class: "num" }, (collab.rmse_baseline ?? 0).toFixed(4))),
              h("tr", {}, h("td", {}, "RMSE (with factors)"), h("td", { class: "num" }, (collab.rmse ?? 0).toFixed(4))),
              h("tr", {}, h("td", {}, "Posters resolved"), h("td", { class: "num" }, (stats.posters?.found || 0).toLocaleString())),
            ),
          ),
        ),
        h(
          "div",
          { class: "prose", style: "margin-top:34px" },
          h("h2", {}, "What it does not do"),
          h(
            "p",
            {},
            "It has no idea what a film is about beyond its genres, the words people tag it with, and " +
              "1,128 semantic scores from the MovieLens tag genome. It cannot read a plot.",
          ),
          h(
            "p",
            {},
            "It also has no memory. Your ratings live in this browser and are sent with each request; " +
              "clear your site data and the profile is gone.",
          ),
          h("h2", {}, "Data"),
          h(
            "p",
            {},
            `${(stats.dataset.movies || 0).toLocaleString()} films and ` +
              `${((collab.n_train || 0) + (collab.n_test || 0)).toLocaleString()} ratings from MovieLens 32M. ` +
              "Posters from TMDB. This product uses the TMDB API but is not endorsed or certified by TMDB.",
          ),
        ),
        ),
      ),
    );
  } catch (error) {
    clear(root);
    root.append(page(errorBox(error, () => renderAbout(root))));
  }
}

// ---------------------------------------------------------------------------
// Onboarding
// ---------------------------------------------------------------------------

export async function renderOnboarding(root) {
  const picked = new Map();
  const genres = new Set();
  let step = 1;
  let facetList = [];

  try {
    facetList = (await api.genres()).slice(0, 12);
  } catch {
    facetList = [];
  }

  clear(root);

  const host = h("div", { class: "onboard-step" });

  const dots = h(
    "div",
    { class: "step-dots", "aria-hidden": "true" },
    [1, 2, 3].map((n) => h("span", { class: "step-dot", "data-on": "false" })),
  );

  const draw = () => {
    dots.querySelectorAll(".step-dot").forEach((dot, index) => {
      dot.setAttribute("data-on", index < step ? "true" : "false");
    });
    clear(host);

    if (step === 1) host.append(stepOne());
    else if (step === 2) host.append(stepTwo());
    else host.append(stepThree());

    host.prepend(dots);
  };

  const pickedList = () =>
    h(
      "div",
      { class: "picked" },
      [...picked.values()].map((movie) =>
        h(
          "span",
          { class: "picked-item" },
          `${movie.title}${movie.year ? ` (${movie.year})` : ""}`,
          h(
            "button",
            {
              type: "button",
              "aria-label": `Remove ${movie.title}`,
              onclick: () => {
                picked.delete(movie.id);
                draw();
              },
            },
            "×",
          ),
        ),
      ),
    );

  const stepOne = () => {
    const results = h("div", { class: "rows", style: "margin-top:14px" });
    let timer = null;

    const input = h("input", {
      type: "search",
      placeholder: "Search for a film you love",
      "aria-label": "Search for a film you love",
      autocomplete: "off",
      style: "width:100%;background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:11px 14px",
      oninput: (event) => {
        const query = event.target.value.trim();
        clearTimeout(timer);
        if (query.length < 2) {
          clear(results);
          return;
        }
        timer = setTimeout(async () => {
          try {
            const hits = await api.search(query, 6);
            clear(results);
            results.append(
              ...hits.map((movie) =>
                h(
                  "button",
                  {
                    class: "row-item",
                    type: "button",
                    style: "grid-template-columns:minmax(0,1fr) auto;text-align:left;cursor:pointer",
                    onclick: () => {
                      picked.set(movie.id, movie);
                      draw();
                    },
                  },
                  h(
                    "div",
                    {},
                    h("div", { class: "row-title" }, movie.title),
                    h("div", { class: "row-meta" }, [movie.year, (movie.genres || [])[0]].filter(Boolean).join(" · ")),
                  ),
                  h("span", { class: "row-meta" }, "Add"),
                ),
              ),
            );
          } catch {
            /* a failed keystroke is not worth an error dialog */
          }
        }, 220);
      },
    });

    return h(
      "div",
      {},
      h("p", { class: "eyebrow" }, "Step 1 of 3"),
      h("h1", {}, "Which films do you love?"),
      h("p", { class: "lede" }, "Pick at least three. They start at 5★ — change any of them later on Your taste."),
      input,
      results,
      pickedList(),
      h(
        "div",
        { class: "actions" },
        h(
          "button",
          {
            class: "btn btn-primary",
            type: "button",
            disabled: picked.size < 3,
            onclick: () => {
              step = 2;
              draw();
            },
          },
          picked.size < 3 ? `Add ${3 - picked.size} more` : "Continue",
        ),
      ),
    );
  };

  const stepTwo = () => {
    const chips = facetList.map((genre) =>
      h(
        "button",
        {
          class: "chip",
          type: "button",
          "aria-pressed": genres.has(genre.name) ? "true" : "false",
          onclick: (event) => {
            if (genres.has(genre.name)) genres.delete(genre.name);
            else if (genres.size < 3) genres.add(genre.name);
            event.currentTarget.setAttribute("aria-pressed", genres.has(genre.name) ? "true" : "false");
          },
        },
        genre.name,
      ),
    );

    return h(
      "div",
      {},
      h("p", { class: "eyebrow" }, "Step 2 of 3"),
      h("h1", {}, "Any genres you lean towards?"),
      h("p", { class: "lede" }, "Optional, and only a starting point — your ratings override this."),
      h("div", { class: "filters" }, chips),
      h(
        "div",
        { class: "actions" },
        h("button", { class: "btn", type: "button", onclick: () => { step = 1; draw(); } }, "Back"),
        h("button", { class: "btn btn-primary", type: "button", onclick: () => { step = 3; draw(); } }, "Continue"),
      ),
    );
  };

  const stepThree = () => {
    const from = h("input", { type: "number", min: "1900", max: "2030", placeholder: "Any", "aria-label": "Earliest year" });
    const to = h("input", { type: "number", min: "1900", max: "2030", placeholder: "Any", "aria-label": "Latest year" });

    return h(
      "div",
      {},
      h("p", { class: "eyebrow" }, "Step 3 of 3"),
      h("h1", {}, "Any era you prefer?"),
      h("p", { class: "lede" }, "Skip it if you do not care. Nothing here is permanent."),
      h(
        "div",
        { class: "field year-range" },
        from,
        h("span", { class: "row-meta" }, "to"),
        to,
      ),
      h(
        "div",
        { class: "actions" },
        h("button", { class: "btn", type: "button", onclick: () => { step = 2; draw(); } }, "Back"),
        h(
          "button",
          {
            class: "btn btn-primary",
            type: "button",
            onclick: () => {
              for (const movie of picked.values()) store.setRating(movie.id, 5);
              store.setPrefs({
                genres: [...genres],
                min_year: from.value ? Number(from.value) : null,
                max_year: to.value ? Number(to.value) : null,
              });
              toast(`Saved ${picked.size} films`);
              navigate("#/");
            },
          },
          "Build my profile",
        ),
      ),
    );
  };

  root.append(page(host));
  draw();
}
