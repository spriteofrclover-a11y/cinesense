// Ratings and preferences live in the browser. The server is stateless: it
// scores whatever it is handed and forgets it.

const RATINGS_KEY = "cinesense.ratings.v1";
const PREFS_KEY = "cinesense.prefs.v1";

const listeners = new Set();

function read(key, fallback) {
  try {
    const raw = localStorage.getItem(key);
    return raw ? JSON.parse(raw) : fallback;
  } catch {
    return fallback;
  }
}

function write(key, value) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    /* private mode or full storage: the session still works, just not persisted */
  }
  listeners.forEach((fn) => fn());
}

export const store = {
  subscribe(fn) {
    listeners.add(fn);
    return () => listeners.delete(fn);
  },

  ratings() {
    const value = read(RATINGS_KEY, {});
    return value && typeof value === "object" ? value : {};
  },

  rating(id) {
    return this.ratings()[id] ?? 0;
  },

  count() {
    return Object.keys(this.ratings()).length;
  },

  setRating(id, rating) {
    const ratings = { ...this.ratings() };
    const value = Math.round(rating * 2) / 2;
    if (value <= 0) delete ratings[id];
    else ratings[id] = Math.min(5, Math.max(0.5, value));
    write(RATINGS_KEY, ratings);
  },

  removeRating(id) {
    const ratings = { ...this.ratings() };
    delete ratings[id];
    write(RATINGS_KEY, ratings);
  },

  clear() {
    write(RATINGS_KEY, {});
  },

  prefs() {
    const value = read(PREFS_KEY, {});
    return value && typeof value === "object" ? value : {};
  },

  setPrefs(prefs) {
    write(PREFS_KEY, prefs);
  },

  /** Films per genre across everything rated, highest first. */
  genreProfile(moviesById) {
    const counts = new Map();
    for (const id of Object.keys(this.ratings())) {
      const movie = moviesById[id];
      if (!movie) continue;
      for (const genre of movie.genres || []) {
        counts.set(genre, (counts.get(genre) || 0) + 1);
      }
    }
    return [...counts.entries()].sort((a, b) => b[1] - a[1]);
  },

  /** Distinct genres covered by the ratings. */
  genreCount(moviesById) {
    return this.genreProfile(moviesById).length;
  },
};
