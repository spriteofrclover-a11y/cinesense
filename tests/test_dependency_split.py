"""The deployed bundle must stay small.

This is the guardrail that keeps the Vercel deployment viable. If someone adds
`import pandas` to the serving path and also lists it in requirements.txt, the
function bundle balloons past what a cold start can tolerate. Fail here instead
of discovering it in production.
"""

from __future__ import annotations

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: Packages that must never appear in the deployed dependency set.
HEAVY = {
    "pandas",
    "scikit-learn",
    "sklearn",
    "scipy",
    "duckdb",
    "torch",
    "sentence-transformers",
    "transformers",
    "lightfm",
    "implicit",
    "surprise",
    "nltk",
    "faiss-cpu",
    "chromadb",
    "streamlit",
    "pyarrow",
}

REQUIREMENT_RE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")


def _parse_requirements(path: Path, _seen: set[Path] | None = None) -> list[str]:
    """Return normalised package names, following ``-r`` includes.

    Resolving includes matters: ``requirements-dev.txt`` starts with
    ``-r requirements.txt``, so skipping those lines used to make the dev set
    look like it was missing every runtime dependency.
    """
    path = path.resolve()
    seen = _seen if _seen is not None else set()
    if path in seen:
        return []
    seen.add(path)

    names: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        include = re.match(r"^(?:-r|--requirement[=\s])\s*(.+)$", stripped)
        if include:
            target = (path.parent / include.group(1).strip()).resolve()
            names.extend(_parse_requirements(target, seen))
            continue

        if stripped.startswith("-"):
            continue  # other pip flags (--index-url, -e, ...)

        match = REQUIREMENT_RE.match(stripped)
        if match:
            names.append(match.group(1).lower().replace("_", "-"))
    return names


def test_runtime_requirements_exclude_heavy_packages():
    runtime = _parse_requirements(PROJECT_ROOT / "requirements.txt")
    assert runtime, "requirements.txt should not be empty"

    offenders = sorted(HEAVY & set(runtime))
    assert not offenders, (
        f"{offenders} found in requirements.txt. The serving bundle must stay "
        "numpy + fastapi + pydantic; heavier deps belong in requirements-dev.txt."
    )


def test_dev_requirements_include_runtime():
    dev = _parse_requirements(PROJECT_ROOT / "requirements-dev.txt")
    runtime = _parse_requirements(PROJECT_ROOT / "requirements.txt")

    assert set(runtime) <= set(dev), (
        "requirements-dev.txt must include runtime deps (use -r requirements.txt)"
    )
    assert "pandas" in dev


def test_pyproject_runtime_dependencies_match_requirements():
    text = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    dependencies_block = text.split("dependencies = [", 1)[1].split("]", 1)[0]
    declared = set(re.findall(r'"([A-Za-z0-9][A-Za-z0-9._-]*)', dependencies_block))

    offenders = sorted(HEAVY & {d.lower().replace("_", "-") for d in declared})
    assert not offenders, f"{offenders} declared as runtime deps in pyproject.toml"


def test_data_package_is_not_imported_by_serving():
    """Static check: the serving layer must not reach into offline modules."""
    guarded = [
        PROJECT_ROOT / "src" / "cinesense" / "serving",
        PROJECT_ROOT / "src" / "cinesense" / "recommend",
    ]

    for directory in guarded:
        if not directory.exists():
            continue
        for path in directory.rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            for forbidden in ("pandas", "sklearn", "scipy", "duckdb"):
                assert f"import {forbidden}" not in source, (
                    f"{path.relative_to(PROJECT_ROOT)} imports {forbidden}; "
                    "the runtime must stay numpy + stdlib"
                )
            assert "from cinesense.data" not in source, (
                f"{path.name} imports offline data code"
            )
            assert "from cinesense.features" not in source, (
                f"{path.name} imports offline feature code"
            )
