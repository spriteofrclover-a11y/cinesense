"""Visual and interaction check against the running server.

Uses the installed Edge via Playwright's channel option, so no browser download.
Captures every route at desktop and phone widths and exercises the two flows that
matter: rating a film, and onboarding.

    .\\.venv\\Scripts\\python.exe scripts/check_ui.py
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

#: Point at a deployment with CINESENSE_BASE, e.g. a Vercel URL.
BASE = os.environ.get("CINESENSE_BASE", "http://127.0.0.1:8011").rstrip("/")
SHOTS = Path("C:/Users/shela/AppData/Local/Temp/opencode/ui")
SHOTS.mkdir(parents=True, exist_ok=True)

DESKTOP = {"width": 1440, "height": 900}
PHONE = {"width": 390, "height": 844}

problems: list[str] = []


def check(condition: bool, message: str) -> None:
    print(f"{'PASS' if condition else 'FAIL'}  {message}")
    if not condition:
        problems.append(message)


def overflow(page) -> tuple[int, int]:
    return page.evaluate(
        "() => [document.documentElement.scrollWidth, document.documentElement.clientWidth]"
    )


def main() -> int:
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)

        # -- desktop -------------------------------------------------------
        page = browser.new_page(viewport=DESKTOP, reduced_motion="reduce")
        console: list[str] = []
        page.on("console", lambda m: console.append(m.text) if m.type == "error" else None)
        page.on("pageerror", lambda e: console.append(str(e)))

        page.goto(BASE, wait_until="networkidle")
        page.wait_for_selector(".card")
        check(page.locator("h1").count() == 1, "discover: one h1")
        check(page.title().startswith("CineSense"), "discover: title set")
        cards = page.locator(".card").count()
        check(cards >= 12, f"discover: {cards} cards rendered")

        # A poster that 404s still renders (there is an onerror fallback), so
        # counting <img> elements proves nothing. naturalWidth proves the bytes
        # actually arrived.
        page.wait_for_timeout(1500)
        loaded = page.evaluate(
            "() => [...document.images].filter(i => i.naturalWidth > 0).length"
        )
        total_imgs = page.locator("img").count()
        check(loaded >= 1, f"discover: {loaded}/{total_imgs} poster images decoded")

        page.screenshot(path=str(SHOTS / "discover-1440.png"))

        # Search dropdown: MovieLens stores "Matrix, The"; the UI must not.
        page.fill("#search", "matrix")
        page.wait_for_selector(".search-hit", timeout=15000)
        titles = page.locator(".search-hit .search-hit-title").all_inner_texts()
        check(len(titles) > 0, f"search: {len(titles)} suggestions")
        inverted = [t for t in titles if re.search(r",\s*(The|A|An)$", t, re.I)]
        check(not inverted, f"search: no inverted titles in {titles[:3]}")

        # Every suggestion must show something: a decoded poster or a lettered
        # placeholder. An <img> that failed to load is an empty black box.
        page.wait_for_timeout(1200)
        shown = page.evaluate(
            "() => [...document.querySelectorAll('.search-hit')].filter((hit) => {"
            "  const img = hit.querySelector('img');"
            "  const tile = hit.querySelector('.poster-mini');"
            "  return (img && img.naturalWidth > 0) || Boolean(tile);"
            "}).length"
        )
        check(shown == len(titles), f"search: {shown}/{len(titles)} thumbnails visible")
        page.screenshot(path=str(SHOTS / "search-1440.png"))
        page.fill("#search", "")

        # Rating three films, so the genre profile has something to show.
        for movie_id, star in ((1, 4), (2, 5), (3, 3)):
            page.goto(f"{BASE}/#/movie/{movie_id}", wait_until="networkidle")
            page.wait_for_selector(".detail-title")
            page.locator(".star").nth(star - 1).click()
            page.wait_for_timeout(250)

        page.goto(f"{BASE}/#/movie/1", wait_until="networkidle")
        page.wait_for_selector(".detail-title")
        note = page.locator(".rating-note").inner_text()
        check(note.strip() == "4.0 / 5", f"movie: clicking the 4th star records {note!r}")

        # The why-panel resolves from an API call, so wait for it rather than
        # sampling the DOM at whatever moment the previous assertion finished.
        try:
            page.wait_for_selector(".why", timeout=15000)
            has_why = True
        except Exception:
            has_why = False
        check(has_why, "movie: why panel present")

        # A rated film must produce real reasons, not the "not in your top 60"
        # fallback: it is excluded from its own list unless we ask for it.
        report = page.locator(".why ul li").count()
        check(report >= 1, f"movie: why panel contains {report} reason(s)")

        contribution = page.evaluate(
            "() => [...document.querySelectorAll('.contrib-fill')]"
            ".map(el => el.getBoundingClientRect().width)"
        )
        check(contribution and all(w > 1 for w in contribution),
              f"movie: all {len(contribution)} contribution bars painted {[round(w) for w in contribution]}")
        page.screenshot(path=str(SHOTS / "movie-1440.png"))

        # The rating survives a reload (localStorage)
        page.reload(wait_until="networkidle")
        check(page.locator(".rating-note").inner_text().strip() == "4.0 / 5",
              "movie: rating persists across reload")

        # Half-star rendering: the fill must CLIP a full-size star. Sizing the
        # container instead scaled the star down, so a "half" star looked like a
        # small star.
        page.locator(".star").nth(4).click(position={"x": 4, "y": 12})
        page.wait_for_timeout(400)
        half = page.evaluate(
            "() => { const s = document.querySelectorAll('.star-fill')[4];"
            " const star = document.querySelectorAll('.star')[4];"
            " return { clip: getComputedStyle(s).clipPath,"
            " fill: Math.round(s.getBoundingClientRect().width),"
            " full: Math.round(star.getBoundingClientRect().width) }; }"
        )
        check("50%" in half["clip"], f"movie: half star clips to 50% ({half['clip']})")
        check(half["fill"] == half["full"],
              f"movie: half star keeps full size ({half['fill']}px of {half['full']}px)")

        # Personalised discover now shows the fingerprint
        page.goto(BASE, wait_until="networkidle")
        page.wait_for_selector(".genre-bars")
        check(page.locator(".fingerprint").count() == 1, "discover: taste fingerprint shown")
        check(page.locator(".genre-bar").count() >= 3, "discover: genre bars render")

        # Counting bars proves nothing: the fills were inline spans with zero
        # width while every element still existed. Measure painted widths.
        bar_widths = page.evaluate(
            "() => [...document.querySelectorAll('.genre-bar-fill')]"
            ".map(el => el.getBoundingClientRect().width)"
        )
        check(bar_widths and all(w > 1 for w in bar_widths),
              f"discover: all {len(bar_widths)} bar fills painted")
        check(len(set(round(w) for w in bar_widths)) > 1,
              f"discover: bar lengths vary with their values {[round(w) for w in bar_widths[:5]]}")

        # Language / region filter
        indian = page.locator(".chip[data-kind='region'][data-value='IN']")
        if indian.count():
            before = page.locator(".card .card-title").all_inner_texts()[:3]
            indian.click()
            page.wait_for_timeout(1800)
            after = page.locator(".card .card-title").all_inner_texts()[:3]
            check(before != after, f"discover: Indian cinema filter changes the grid ({after[:2]})")
            check(indian.get_attribute("aria-pressed") == "true", "discover: region chip reflects state")
            page.screenshot(path=str(SHOTS / "discover-indian-1440.png"))
            indian.click()
            page.wait_for_timeout(1200)
        else:
            check(False, "discover: Indian cinema chip missing")
        page.screenshot(path=str(SHOTS / "discover-personal-1440.png"))

        # Taste + About
        page.goto(f"{BASE}/#/taste", wait_until="networkidle")
        page.wait_for_selector(".genre-bars, .empty")
        check(page.locator(".rating-row").count() >= 1, "taste: rated films listed")

        # The list must be editable, not just readable: static star glyphs were
        # a real complaint ("I can't change them").
        before = page.evaluate("() => localStorage.getItem('cinesense.ratings.v1')")
        page.locator(".rating-row .star").first.click()
        page.wait_for_timeout(900)
        after = page.evaluate("() => localStorage.getItem('cinesense.ratings.v1')")
        check(before != after, "taste: rating editable from the list")
        page.screenshot(path=str(SHOTS / "taste-1440.png"))

        page.goto(f"{BASE}/#/about", wait_until="networkidle")
        # Waiting for h1 races the previous view: the old heading satisfies the
        # selector before About has finished fetching /api/stats. Wait for the
        # thing being asserted instead.
        try:
            page.wait_for_selector("table.metrics", timeout=20000)
            has_table = True
        except Exception:
            has_table = False
        check(has_table, "about: metrics table present")
        page.screenshot(path=str(SHOTS / "about-1440.png"), full_page=True)

        # -- phone ---------------------------------------------------------
        phone = browser.new_page(viewport=PHONE, reduced_motion="reduce")
        phone.goto(BASE, wait_until="networkidle")
        phone.wait_for_selector(".card")
        width, client = overflow(phone)
        check(width <= client + 1, f"phone: no horizontal overflow (scroll {width} vs client {client})")
        check(phone.evaluate("() => document.documentElement.clientWidth") == 390,
              "phone: viewport really is 390 CSS px")
        phone.screenshot(path=str(SHOTS / "discover-390.png"))

        phone.goto(f"{BASE}/#/movie/1", wait_until="networkidle")
        phone.wait_for_selector(".detail-title")
        width, client = overflow(phone)
        check(width <= client + 1, f"phone movie: no overflow (scroll {width} vs client {client})")
        phone.screenshot(path=str(SHOTS / "movie-390.png"))

        # -- onboarding ----------------------------------------------------
        flow = browser.new_page(viewport=DESKTOP, reduced_motion="reduce")
        flow.goto(f"{BASE}/#/start", wait_until="networkidle")
        onboarding_search = flow.locator(".onboard-step input[type=search]")
        onboarding_search.wait_for()
        onboarding_search.fill("matrix")
        # .count() does not auto-wait the way .click() does, so wait for the
        # debounced results instead of sleeping a fixed number of milliseconds.
        try:
            flow.wait_for_selector(".onboard-step .row-item", timeout=15000)
            flow.wait_for_timeout(150)
        except Exception:
            pass
        hits = flow.locator(".onboard-step .row-item")
        check(hits.count() >= 1, f"onboarding: search returns {hits.count()} hits")
        hits.first.click()
        flow.wait_for_timeout(250)
        check(flow.locator(".picked-item").count() == 1, "onboarding: picking adds a chip")
        flow.screenshot(path=str(SHOTS / "onboarding-1440.png"))

        check(not console, f"no console errors ({console[:3]})")

        browser.close()

    print()
    print(f"{len(problems)} problem(s)")
    for item in problems:
        print(f"  - {item}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
