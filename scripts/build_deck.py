r"""Build the CineSense presentation.

Not part of the application: this only produces a file for presenting the
project. Needs `pip install python-pptx`, which is deliberately not in
requirements-dev.txt because the app does not depend on it.

    .\\.venv\\Scripts\\python.exe scripts\\build_deck.py
"""

from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "CineSense-Presentation.pptx"

BG = RGBColor(0x0A, 0x0A, 0x0C)
SURFACE = RGBColor(0x15, 0x15, 0x1B)
BORDER = RGBColor(0x2A, 0x2A, 0x34)
INK = RGBColor(0xF3, 0xF3, 0xF6)
BODY = RGBColor(0xC4, 0xC4, 0xD0)
MUTED = RGBColor(0x8E, 0x8E, 0x9E)
ACCENT = RGBColor(0xE9, 0xB4, 0x4C)

HEAD = "Georgia"
SANS = "Segoe UI"

W, H = 13.333, 7.5
LEFT = 0.85
CONTENT_W = W - 2 * LEFT

prs = Presentation()
prs.slide_width = Inches(W)
prs.slide_height = Inches(H)
BLANK = prs.slide_layouts[6]


def new_slide():
    slide = prs.slides.add_slide(BLANK)
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = BG
    return slide


def text(slide, x, y, w, h, value, *, size=18, color=BODY, font=SANS, bold=False,
         align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, spacing=1.0):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = box.text_frame
    frame.word_wrap = True
    frame.vertical_anchor = anchor
    frame.margin_left = 0
    frame.margin_right = 0
    frame.margin_top = 0
    frame.margin_bottom = 0

    lines = value.split("\n")
    for index, line in enumerate(lines):
        para = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        para.alignment = align
        para.line_spacing = spacing
        run = para.add_run()
        run.text = line
        run.font.size = Pt(size)
        run.font.color.rgb = color
        run.font.name = font
        run.font.bold = bold
    return box


def rect(slide, x, y, w, h, text_value, *, fill=SURFACE, edge=BORDER, size=14,
         color=BODY, font=SANS, bold=False, rounded=True, align=PP_ALIGN.CENTER):
    shape_type = MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE
    shape = slide.shapes.add_shape(shape_type, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.color.rgb = edge
    shape.line.width = Pt(0.75)
    shape.shadow.inherit = False
    if rounded:
        shape.adjustments[0] = 0.08

    frame = shape.text_frame
    frame.word_wrap = True
    frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    frame.margin_left = Inches(0.1)
    frame.margin_right = Inches(0.1)
    para = frame.paragraphs[0]
    para.alignment = align
    run = para.add_run()
    run.text = text_value
    run.font.size = Pt(size)
    run.font.color.rgb = color
    run.font.name = font
    run.font.bold = bold
    return shape


def arrow(slide, x, y, w, h):
    shape = slide.shapes.add_shape(MSO_SHAPE.DOWN_ARROW, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = RGBColor(0x3A, 0x3A, 0x46)
    shape.line.fill.background()
    shape.shadow.inherit = False


def content_slide(title, kicker=None):
    slide = new_slide()
    y = 0.55
    if kicker:
        text(slide, LEFT, y, CONTENT_W, 0.3, kicker.upper(), size=11, color=ACCENT, bold=True)
        y += 0.34
    text(slide, LEFT, y, CONTENT_W, 0.75, title, size=30, color=INK, font=HEAD)
    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(LEFT), Inches(y + 0.86),
                                  Inches(1.1), Inches(0.035))
    line.fill.solid()
    line.fill.fore_color.rgb = ACCENT
    line.line.fill.background()
    line.shadow.inherit = False
    return slide


def bullets(slide, items, *, x=LEFT, y=2.05, w=CONTENT_W, size=18, gap=0.62, color=BODY):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(H - y - 0.9))
    frame = box.text_frame
    frame.word_wrap = True
    frame.margin_left = 0
    frame.margin_right = 0
    frame.margin_top = 0
    frame.margin_bottom = 0

    for index, item in enumerate(items):
        para = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        para.space_after = Pt(gap * 12)
        para.line_spacing = 1.05
        run = para.add_run()
        run.text = "- " + item
        run.font.size = Pt(size)
        run.font.color.rgb = color
        run.font.name = SANS
    return box


def footer(slide, number):
    text(slide, LEFT, H - 0.55, 6.0, 0.3, "CineSense  |  Problem Statement 8", size=10, color=MUTED)
    text(slide, W - LEFT - 1.0, H - 0.55, 1.0, 0.3, str(number), size=10,
         color=MUTED, align=PP_ALIGN.RIGHT)


# --------------------------------------------------------------------------- #
# 1. Title
# --------------------------------------------------------------------------- #

slide = new_slide()
band = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(W), Inches(0.14))
band.fill.solid()
band.fill.fore_color.rgb = ACCENT
band.line.fill.background()
band.shadow.inherit = False

text(slide, LEFT, 1.25, CONTENT_W, 0.4, "PROBLEM STATEMENT 8", size=13, color=ACCENT, bold=True)
text(slide, LEFT, 1.75, CONTENT_W, 1.2, "CineSense", size=64, color=INK, font=HEAD)
text(slide, LEFT, 2.95, CONTENT_W, 0.6, "AI-Powered Movie Recommendation System",
     size=24, color=BODY)
text(slide, LEFT, 3.72, CONTENT_W, 0.4, "cinesense-mu.vercel.app", size=17, color=ACCENT)

rule = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(LEFT), Inches(4.45), Inches(CONTENT_W), Inches(0.012))
rule.fill.solid()
rule.fill.fore_color.rgb = BORDER
rule.line.fill.background()
rule.shadow.inherit = False

text(slide, LEFT, 4.78, 4.0, 0.3, "PRESENTED BY", size=11, color=MUTED, bold=True)
names = [
    ("Sivakumar Sudhakar", "Registration No. ____________"),
    ("Yeshwant Naidu", "Registration No. ____________"),
    ("Siddhardha", "Registration No. ____________"),
]
for index, (name, reg) in enumerate(names):
    col = index % 3
    x = LEFT + col * 3.95
    text(slide, x, 5.18, 3.7, 0.32, name, size=16, color=INK, bold=True)
    text(slide, x, 5.5, 3.7, 0.3, reg, size=12, color=MUTED)

text(slide, LEFT, 6.35, CONTENT_W, 0.3, "Department of Computer Science and Engineering",
     size=12, color=MUTED)

# --------------------------------------------------------------------------- #
# 2. Introduction and problem statement
# --------------------------------------------------------------------------- #

slide = content_slide("Introduction and Problem Statement", "Section 2")
bullets(slide, [
    "Streaming services hold tens of thousands of films, and people browse only a small part of them.",
    "Sites collect ratings, but rarely explain what they do with those ratings.",
    "A new user has no history at all, so a ratings-only system cannot help them on day one.",
    "The result is the same every time: users cannot find films they would enjoy, and stop looking.",
], w=7.1)

for index, (value, label) in enumerate([("87,585", "films in our catalogue"), ("32M", "ratings behind it")]):
    y = 2.9 + index * 1.5
    text(slide, 8.55, y, 4.0, 0.7, value, size=40, color=ACCENT, font=HEAD)
    text(slide, 8.55, y + 0.68, 4.0, 0.3, label, size=12, color=MUTED)
footer(slide, 2)

# --------------------------------------------------------------------------- #
# 3. Objectives and need
# --------------------------------------------------------------------------- #

slide = content_slide("Project Objectives and Need", "Section 3")
text(slide, LEFT, 2.05, CONTENT_W, 0.3, "OBJECTIVES", size=11, color=ACCENT, bold=True)
bullets(slide, [
    "Recommend films from a user's own ratings.",
    "Explain every suggestion in plain language.",
    "Work for a first-time user with no rating history.",
    "Deliver a live, working product rather than a notebook.",
], y=2.45, w=6.1, gap=0.5)

text(slide, 7.4, 2.05, 5.1, 0.3, "WHY IT MATTERS", size=11, color=ACCENT, bold=True)
text(slide, 7.4, 2.45, 5.1, 3.4,
     "Catalogues outgrew browsing. A user cannot scan 87,000 films, "
     "and a recommendation nobody understands is one nobody acts on.\n\n"
     "A system that shows its reasoning earns trust, and one that works "
     "before you rate anything keeps the user long enough to be useful.",
     size=15, color=BODY, spacing=1.15)
footer(slide, 3)

# --------------------------------------------------------------------------- #
# 4. Overview and key features
# --------------------------------------------------------------------------- #

slide = content_slide("Project Overview and Key Features", "Section 4")
text(slide, LEFT, 2.1, 5.3, 3.6,
     "CineSense is a web application that recommends films from a user's ratings. "
     "It combines three signals: what similar users liked, what a film is like, and "
     "how widely it is rated.\n\n"
     "Each suggestion arrives with the reason behind it, so the user can see what "
     "the system acted on.",
     size=15, color=BODY, spacing=1.15)

features = [
    "Rate a few films, get a ranked list",
    "Every film card shows why it was picked",
    "Filter by genre, language and region",
    "A taste profile you can edit any time",
    "Search across all 87,585 films",
    "Works before you rate anything",
]
for index, item in enumerate(features):
    y = 2.1 + index * 0.62
    dot = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(7.3), Inches(y + 0.11), Inches(0.09), Inches(0.09))
    dot.fill.solid()
    dot.fill.fore_color.rgb = ACCENT
    dot.line.fill.background()
    dot.shadow.inherit = False
    text(slide, 7.55, y, 4.9, 0.4, item, size=15, color=BODY)
footer(slide, 4)

# --------------------------------------------------------------------------- #
# 5. How it works
# --------------------------------------------------------------------------- #

slide = content_slide("Working", "Section 4")

sources = [
    ("Your ratings", "what similar users liked"),
    ("Film similarity", "genre, tags and themes"),
    ("Popularity", "how widely a film is rated"),
]
for index, (name, note) in enumerate(sources):
    x = 0.85 + index * 4.15
    rect(slide, x, 2.15, 3.75, 1.05, "", fill=SURFACE)
    text(slide, x + 0.18, 2.32, 3.4, 0.32, name, size=15, color=INK, bold=True)
    text(slide, x + 0.18, 2.68, 3.4, 0.3, note, size=11.5, color=MUTED)
    arrow(slide, x + 1.78, 3.32, 0.19, 0.34)

rect(slide, 2.85, 3.78, 7.6, 0.85, "One score per film, weighted and combined",
     fill=RGBColor(0x1C, 0x18, 0x10), edge=RGBColor(0x5A, 0x48, 0x22), size=15,
     color=ACCENT, bold=True)
arrow(slide, W / 2 - 0.095, 4.72, 0.19, 0.34)
rect(slide, 2.85, 5.18, 7.6, 0.85, "Ranked list, each film with its reason",
     fill=SURFACE, size=15, color=INK, bold=True)

text(slide, LEFT, 6.3, CONTENT_W, 0.4,
     "Scores are scaled to a common range before combining, so no single signal dominates.",
     size=12, color=MUTED)
footer(slide, 5)

# --------------------------------------------------------------------------- #
# 6. Tools and technologies
# --------------------------------------------------------------------------- #

slide = content_slide("Tools, Technologies and Frameworks", "Section 5")

groups = [
    ("LANGUAGE AND LIBRARIES", ["Python", "pandas", "numpy", "scikit-learn", "duckDB"]),
    ("WEB AND HOSTING", ["FastAPI", "HTML", "CSS", "JavaScript", "Vercel"]),
    ("DATA", ["MovieLens 32M", "TMDB API"]),
]
for index, (heading, items) in enumerate(groups):
    x = LEFT + index * 3.95
    text(slide, x, 2.15, 3.7, 0.3, heading, size=10.5, color=ACCENT, bold=True)
    for row, item in enumerate(items):
        text(slide, x, 2.62 + row * 0.62, 3.7, 0.4, item, size=17, color=BODY)

text(slide, LEFT, 5.85, CONTENT_W, 0.6,
     "Python for the data work and the models, FastAPI to serve them, plain HTML, CSS and "
     "JavaScript for the interface, and Vercel to host it.",
     size=13, color=MUTED, spacing=1.2)
footer(slide, 6)

# --------------------------------------------------------------------------- #
# 7. Methodology
# --------------------------------------------------------------------------- #

slide = content_slide("Methodology and Implementation", "Section 6")

steps = [
    ("1. Prepare the data", "32 million ratings and film details cleaned and stored locally for fast reuse."),
    ("2. Describe each film", "Genre, tags and theme scores turned into a compact numerical profile."),
    ("3. Learn the patterns", "Similarities learned from how users rate films together."),
    ("4. Build the application", "A web app that scores a request instantly and explains the result."),
]
for index, (heading, note) in enumerate(steps):
    y = 2.05 + index * 1.14
    text(slide, LEFT, y, 0.55, 0.4, heading[0], size=20, color=ACCENT, font=HEAD)
    text(slide, LEFT + 0.55, y + 0.02, 5.0, 0.35, heading[3:], size=17, color=INK, bold=True)
    text(slide, LEFT + 0.55, y + 0.38, 10.4, 0.5, note, size=13, color=MUTED, spacing=1.15)
footer(slide, 7)

# --------------------------------------------------------------------------- #
# 8. Results
# --------------------------------------------------------------------------- #

slide = content_slide("Results", "Section 7")

stats = [
    ("87,585", "films covered"),
    ("0.82", "average error, in stars out of 5"),
    ("11%", "better ranking than one signal alone"),
    ("43,129", "films with posters and language"),
    ("1,141", "Indian films made reachable"),
    ("2.1 s", "first load, under 1 s after"),
]
for index, (value, label) in enumerate(stats):
    col, row = index % 3, index // 3
    x = LEFT + col * 3.95
    y = 2.05 + row * 1.75
    text(slide, x, y, 3.7, 0.72, value, size=38, color=ACCENT, font=HEAD)
    text(slide, x, y + 0.76, 3.5, 0.55, label, size=12, color=MUTED, spacing=1.1)
text(slide, LEFT, 5.95, CONTENT_W, 0.5,
     "Measured on 3.2 million ratings the model never saw while it was being built.",
     size=12, color=MUTED)
footer(slide, 8)

# --------------------------------------------------------------------------- #
# 9. Testing
# --------------------------------------------------------------------------- #

slide = content_slide("Testing", "Section 7")

text(slide, LEFT, 2.15, 2.6, 0.8, "192", size=44, color=ACCENT, font=HEAD)
text(slide, LEFT, 2.95, 2.8, 0.4, "automated tests", size=13, color=MUTED)
text(slide, 3.6, 2.15, 2.6, 0.8, "29", size=44, color=ACCENT, font=HEAD)
text(slide, 3.6, 2.95, 2.8, 0.4, "browser checks on the live site", size=13, color=MUTED)

text(slide, LEFT, 3.85, CONTENT_W, 0.3, "WHAT IS COVERED", size=11, color=ACCENT, bold=True)
bullets(slide, [
    "The data pipeline produces the expected counts and fails loudly when it does not.",
    "The recommendation models reproduce known results on held-out ratings.",
    "The API rejects bad input and returns valid responses.",
    "In a real browser: posters load, ratings save, filters change the list, no errors.",
], y=4.25, gap=0.42, size=15)
footer(slide, 9)

# --------------------------------------------------------------------------- #
# 10-11. Screenshots
# --------------------------------------------------------------------------- #

for number, (title, kicker, image, notes) in enumerate([
    ("The interface", "Section 7", "docs/discover.png", [
        "Recommendations with a genre and language filter row.",
        "A fingerprint of the genres your ratings concentrate in.",
    ]),
    ("Explaining a recommendation", "Section 7", "docs/movie-detail.png", [
        "The reason, the shared ground, and how much each signal contributed.",
        "Content neighbours and collaborative neighbours on the same page.",
    ]),
], start=10):
    slide = content_slide(title, kicker)
    path = ROOT / image
    pic = slide.shapes.add_picture(str(path), Inches(LEFT), Inches(2.15), width=Inches(8.2))
    if pic.height > Inches(4.55):
        pic.height = Inches(4.55)
        pic.width = Emu(int(pic.height * 8.2 / 4.55))
    for index, note in enumerate(notes):
        y = 2.35 + index * 1.15
        text(slide, 9.35, y, 3.3, 0.9, note, size=14, color=BODY, spacing=1.2)
    footer(slide, number)

# --------------------------------------------------------------------------- #
# 12. Individual contributions
# --------------------------------------------------------------------------- #

slide = content_slide("Individual Contributions", "Section 8")

people = [
    ("Sivakumar Sudhakar", [
        "Data preparation and cleaning",
        "Content and collaborative models",
        "Combining the signals, evaluation",
    ]),
    ("Yeshwant Naidu", [
        "Backend API and request handling",
        "Web interface and interaction",
        "Deployment and browser testing",
    ]),
    ("Siddhardha", [
        "Research on recommendation methods",
        "Literature and dataset study",
        "Report and presentation",
    ]),
]
for index, (name, items) in enumerate(people):
    x = LEFT + index * 3.95
    rect(slide, x, 2.1, 3.6, 3.6, "", fill=SURFACE)
    text(slide, x + 0.28, 2.35, 3.1, 0.4, name, size=17, color=INK, bold=True)
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x + 0.28), Inches(2.82), Inches(0.9), Inches(0.03))
    bar.fill.solid()
    bar.fill.fore_color.rgb = ACCENT
    bar.line.fill.background()
    bar.shadow.inherit = False
    for row, item in enumerate(items):
        text(slide, x + 0.28, 3.02 + row * 0.8, 3.05, 0.75, item, size=13, color=MUTED, spacing=1.15)
footer(slide, 12)

# --------------------------------------------------------------------------- #
# 13. Limitations and future scope
# --------------------------------------------------------------------------- #

slide = content_slide("Limitations and Future Scope", "Section 9")

text(slide, LEFT, 2.1, 5.6, 0.3, "LIMITATIONS", size=11, color=ACCENT, bold=True)
bullets(slide, [
    "No understanding of plot, only genre and tags.",
    "The catalogue is Western-heavy.",
    "Films with few ratings are weakly modelled.",
    "No user accounts, so nothing syncs.",
], y=2.5, w=5.6, size=14, gap=0.5)

text(slide, 7.2, 2.1, 5.3, 0.3, "FUTURE SCOPE", size=11, color=ACCENT, bold=True)
bullets(slide, [
    "Read plot summaries for real story similarity.",
    "Add films missing from the dataset.",
    "User accounts with saved profiles.",
    "Tune the mix of signals on live feedback.",
], x=7.2, y=2.5, w=5.3, size=14, gap=0.5)
footer(slide, 13)

# --------------------------------------------------------------------------- #
# 14. References
# --------------------------------------------------------------------------- #

slide = content_slide("References and Resources", "Section 10")
bullets(slide, [
    "MovieLens 32M and 25M datasets, GroupLens Research, University of Minnesota.",
    "MovieLens tag genome, for theme scores per film.",
    "TMDB API, for poster images and film language and country.",
    "FastAPI and scikit-learn documentation.",
], y=2.3, gap=0.65, size=16)
footer(slide, 14)

OUT.parent.mkdir(parents=True, exist_ok=True)
prs.save(OUT)
print(f"saved {OUT.name}  ({OUT.stat().st_size / 1024:.0f} KB, {len(prs.slides)} slides)")
