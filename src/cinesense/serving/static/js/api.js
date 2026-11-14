// Thin wrapper over the JSON API. Every failure becomes an ApiError with a
// message worth showing a human.

const BASE = "/api";

export class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function request(path, { method = "GET", body } = {}) {
  let response;
  try {
    response = await fetch(`${BASE}${path}`, {
      method,
      headers: body ? { "Content-Type": "application/json" } : undefined,
      body: body ? JSON.stringify(body) : undefined,
    });
  } catch (cause) {
    throw new ApiError("Could not reach the server. Check your connection.", 0);
  }

  if (!response.ok) {
    let detail = `Request failed (${response.status})`;
    try {
      const payload = await response.json();
      if (typeof payload.detail === "string") detail = payload.detail;
      else if (Array.isArray(payload.detail) && payload.detail.length) {
        detail = payload.detail[0].msg || detail;
      }
    } catch {
      /* keep the generic message */
    }
    throw new ApiError(detail, response.status);
  }

  return response.json();
}

export const api = {
  health: () => request("/health"),
  stats: () => request("/stats"),
  genres: () => request("/genres"),
  facets: () => request("/facets"),
  search: (q, limit = 8) => request(`/search?q=${encodeURIComponent(q)}&limit=${limit}`),
  movie: (id) => request(`/movies/${id}`),
  moviesByIds: (ids) => request("/movies", { method: "POST", body: { ids } }),
  similar: (id, k = 12) => request(`/movies/${id}/similar?k=${k}`),
  related: (id, k = 12) => request(`/movies/${id}/related?k=${k}`),
  recommend: (payload) => request("/recommendations", { method: "POST", body: payload }),
};
