// Router, nav state and the global search box.

import { api } from "./api.js";
import { store } from "./store.js";
import { clear, h } from "./ui.js";
import {
  renderAbout,
  renderDiscover,
  renderMovie,
  renderOnboarding,
  renderTaste,
} from "./views.js";

const view = document.getElementById("view");

const ROUTES = [
  { pattern: /^#\/?$/, render: (root) => renderDiscover(root), nav: "discover" },
  { pattern: /^#\/start$/, render: (root) => renderOnboarding(root), nav: null },
  { pattern: /^#\/movie\/(\d+)$/, render: (root, m) => renderMovie(root, Number(m[1])), nav: "discover" },
  { pattern: /^#\/taste$/, render: (root) => renderTaste(root), nav: "taste" },
  { pattern: /^#\/about$/, render: (root) => renderAbout(root), nav: "about" },
];

function setNav(name) {
  document.querySelectorAll("[data-nav]").forEach((link) => {
    if (link.dataset.nav === name) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  });
}

function route() {
  const hash = location.hash || "#/";

  for (const entry of ROUTES) {
    const match = hash.match(entry.pattern);
    if (!match) continue;

    setNav(entry.nav);
    clear(view);
    document.title = "CineSense — find the film you did not know you wanted";
    entry.render(view, match);
    return;
  }

  location.hash = "#/";
}

window.addEventListener("hashchange", () => {
  closeSearch();
  route();
});

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

const input = document.getElementById("search");
const results = document.getElementById("search-results");

let searchTimer = null;
let hits = [];
let cursor = -1;

function closeSearch() {
  results.hidden = true;
  results.replaceChildren();
  input?.setAttribute("aria-expanded", "false");
  hits = [];
  cursor = -1;
}

function drawSearch() {
  clear(results);

  if (!hits.length) {
    results.append(h("div", { class: "search-empty" }, "No films found"));
    results.hidden = false;
    input.setAttribute("aria-expanded", "true");
    return;
  }

  hits.forEach((movie, index) => {
    const button = h(
      "button",
      {
        class: "search-hit",
        type: "button",
        role: "option",
        "aria-selected": index === cursor ? "true" : "false",
        onclick: () => {
          closeSearch();
          if (input) input.value = "";
          location.hash = `#/movie/${movie.id}`;
        },
      },
      movie.poster
        ? h("img", { src: movie.poster, alt: "", decoding: "async" })
        : h("div", { class: "poster-mini", "aria-hidden": "true" },
            (movie.title || "?").charAt(0).toUpperCase()),
      h(
        "div",
        {},
        h("div", { class: "search-hit-title" }, movie.title),
        h(
          "div",
          { class: "search-hit-meta" },
          [movie.year, (movie.genres || [])[0]].filter(Boolean).join(" · "),
        ),
      ),
    );
    results.append(button);
  });

  results.hidden = false;
  input.setAttribute("aria-expanded", "true");
}

input?.addEventListener("input", (event) => {
  const query = event.target.value.trim();
  clearTimeout(searchTimer);

  if (query.length < 2) {
    closeSearch();
    return;
  }

  searchTimer = setTimeout(async () => {
    try {
      hits = await api.search(query, 6);
      cursor = hits.length ? 0 : -1;
      drawSearch();
    } catch {
      closeSearch();
    }
  }, 200);
});

input?.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    closeSearch();
    input.blur();
    return;
  }

  if (!hits.length) return;

  if (event.key === "ArrowDown" || event.key === "ArrowUp") {
    event.preventDefault();
    const delta = event.key === "ArrowDown" ? 1 : -1;
    cursor = (cursor + delta + hits.length) % hits.length;
    drawSearch();
    return;
  }

  if (event.key === "Enter") {
    event.preventDefault();
    const movie = hits[cursor] ?? hits[0];
    if (movie) {
      closeSearch();
      input.value = "";
      location.hash = `#/movie/${movie.id}`;
    }
  }
});

input?.addEventListener("blur", () => {
  // Delay so a click on a result registers before the list disappears.
  setTimeout(closeSearch, 150);
});

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------

if (!location.hash) location.hash = "#/";
route();

// Re-render Discover when ratings change elsewhere (e.g. after onboarding).
store.subscribe(() => {
  if (/^#\/?$/.test(location.hash || "#/")) {
    // Let the current interaction finish before rebuilding the page.
    requestAnimationFrame(route);
  }
});
