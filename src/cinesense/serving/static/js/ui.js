// Shared DOM builders. No framework: these are the only abstractions the views
// need, and they keep every view declarative.

const STAR_OUTLINE =
  '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 2.8l2.85 5.78 6.38.93-4.62 4.5 1.09 6.35L12 17.36l-5.7 3-1.09-6.35L.59 9.51l6.38-.93z" fill="none" stroke="currentColor" stroke-width="1.4"/></svg>';
const STAR_FILLED =
  '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 2.8l2.85 5.78 6.38.93-4.62 4.5 1.09 6.35L12 17.36l-5.7 3-1.09-6.35L.59 9.51l6.38-.93z" fill="currentColor"/></svg>';

/** Muted film-poster tints. No violet, per the design rules. */
const TINTS = [
  "#1d3b3a", "#3a2320", "#2e3620", "#22303f",
  "#3a2f1c", "#243029", "#33252a", "#1f2a33",
];

export function h(tag, props = {}, ...children) {
  const node = document.createElement(tag);

  for (const [key, value] of Object.entries(props || {})) {
    if (value == null || value === false) continue;
    if (key === "class") node.className = value;
    else if (key === "style") node.style.cssText = value;
    else if (key === "html") node.innerHTML = value;
    else if (key === "value") node.value = value;
    else if (key.startsWith("on") && typeof value === "function") {
      node.addEventListener(key.slice(2).toLowerCase(), value);
    } else if (value === true) node.setAttribute(key, "");
    else node.setAttribute(key, String(value));
  }

  for (const child of children.flat(3)) {
    if (child == null || child === false) continue;
    node.append(child.nodeType ? child : document.createTextNode(String(child)));
  }
  return node;
}

export function clear(node) {
  node.replaceChildren();
}

export function posterNode(movie, { lazy = true } = {}) {
  const wrapper = h("div", { class: "poster" });

  // The fallback is artwork, not a second title: the card already shows the
  // title below, and repeating it read as a mistake.
  const fallback = h(
    "div",
    { class: "poster-fallback", style: `--tint:${TINTS[Math.abs(movie.id) % TINTS.length]}` },
    h("span", { class: "poster-monogram", "aria-hidden": "true" }, (movie.title || "?").charAt(0).toUpperCase()),
    h("span", { class: "poster-genre" }, (movie.genres || [])[0] || ""),
    movie.year ? h("span", { class: "poster-year" }, String(movie.year)) : null,
  );

  if (movie.poster) {
    const img = h("img", {
      src: movie.poster,
      alt: "",
      loading: lazy ? "lazy" : "eager",
      decoding: "async",
      width: "342",
      height: "513",
    });
    img.addEventListener("error", () => {
      img.remove();
      wrapper.append(fallback);
    });
    wrapper.append(img);
  } else {
    wrapper.append(fallback);
  }
  return wrapper;
}

/** The reason worth showing on a card: only when it is personal. */
export function cardReason(movie) {
  const explanation = movie.explanation;
  if (!explanation?.reasons?.length) return "";

  const { content = 0, collaborative = 0, popularity = 0 } = explanation;
  const personal = Math.max(content, collaborative) >= popularity;
  return personal ? explanation.reasons[0] : "";
}

export function cardNode(movie, { onOpen, why = "", delay = 0 } = {}) {
  const meta = [movie.year ? String(movie.year) : null, (movie.genres || [])[0] || null]
    .filter(Boolean)
    .join(" · ");

  const card = h(
    "button",
    {
      class: "card",
      type: "button",
      style: delay ? `animation-delay:${delay}ms` : null,
      "aria-label": `${movie.title}${movie.year ? ` (${movie.year})` : ""}`,
      onclick: () => onOpen?.(movie.id),
    },
    posterNode(movie),
    h("div", { class: "card-title" }, movie.title),
    meta ? h("div", { class: "card-meta" }, meta) : null,
    why ? h("div", { class: "card-why" }, why) : null,
  );
  return card;
}

function fillPercent(value, index) {
  return Math.min(1, Math.max(0, value - (index - 1))) * 100;
}

export function starsNode(value, onRate, { label = "Your rating" } = {}) {
  const wrapper = h("div", { class: "stars", role: "group", "aria-label": label });
  const fills = [];
  const buttons = [];

  // The widget owns its own display. Without this, clicking a star updated the
  // stored value and the text label but left the stars themselves unchanged
  // until the page was re-rendered.
  const apply = (next) => {
    fills.forEach((fill, position) => {
      fill.style.setProperty("--fill", `${fillPercent(next, position + 1)}%`);
    });
    buttons.forEach((button, position) => {
      button.setAttribute("aria-pressed", next >= position + 1 ? "true" : "false");
    });
  };

  for (let index = 1; index <= 5; index += 1) {
    const fill = h("span", { class: "star-fill" });
    fill.innerHTML = STAR_FILLED;
    fills.push(fill);

    const button = h("button", {
      class: "star",
      type: "button",
      "aria-label": `${index} star${index === 1 ? "" : "s"}`,
    });
    button.innerHTML = STAR_OUTLINE;
    button.append(fill);
    buttons.push(button);

    button.addEventListener("click", (event) => {
      const rect = button.getBoundingClientRect();
      // Only the left third means "half". Putting the boundary at the centre
      // made clicking a star's middle a coin flip between 3 and 2.5; clicking a
      // star should plainly mean that many stars.
      const leftEdge = event.clientX - rect.left < rect.width * 0.33;
      const next = leftEdge ? index - 0.5 : index;
      apply(next);
      onRate?.(next);
    });

    wrapper.append(button);
  }

  apply(value);
  return wrapper;
}

export function ratingLabel(value) {
  if (!value) return "Not rated";
  return `${value.toFixed(1)} / 5`;
}

export function contribBars(explanation) {
  const rows = [
    ["Collaborative", explanation.collaborative],
    ["Content", explanation.content],
    ["Popularity", explanation.popularity],
  ].filter(([, share]) => share >= 0.02);

  if (!rows.length) return null;

  return h(
    "div",
    { class: "contrib" },
    rows.map(([name, share], index) =>
      h(
        "div",
        { class: "contrib-row" },
        h("span", { class: "contrib-name" }, name),
        h(
          "span",
          { class: "contrib-track" },
          h("span", {
            class: "contrib-fill",
            style: `width:${(share * 100).toFixed(1)}%;animation-delay:${index * 90}ms`,
          }),
        ),
        h("span", { class: "contrib-value" }, `${Math.round(share * 100)}%`),
      ),
    ),
  );
}

let toastTimer = null;

export function toast(message) {
  const node = document.getElementById("toast");
  if (!node) return;
  node.textContent = message;
  node.hidden = false;
  requestAnimationFrame(() => node.setAttribute("data-show", "true"));

  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    node.removeAttribute("data-show");
    setTimeout(() => {
      node.hidden = true;
    }, 200);
  }, 1900);
}

export function skeletonGrid(count = 12) {
  return h(
    "div",
    { class: "grid", "aria-hidden": "true" },
    Array.from({ length: count }, () =>
      h("div", {}, h("div", { class: "skeleton skeleton-poster" }),
        h("div", { class: "skeleton", style: "height:12px;margin-top:9px;width:80%" })),
    ),
  );
}

export function errorBox(error, onRetry) {
  return h(
    "div",
    { class: "error-box" },
    h("h2", {}, "Something went wrong"),
    h("p", {}, error?.message || "Unknown error."),
    onRetry ? h("button", { class: "btn btn-sm", type: "button", onclick: onRetry }, "Try again") : null,
  );
}
