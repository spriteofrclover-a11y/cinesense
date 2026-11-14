"""Shared test setup.

Two jobs, both about failing loudly instead of obscurely:

1. Make ``cinesense`` importable without requiring an editable install.
2. Refuse to run at all if the interpreter is the wrong one. The ``python`` on
   PATH here is 3.11 with no pandas, and a bare ImportError buried in a test
   report wastes far more time than this message.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

REQUIRED_MODULES = ("pandas", "pyarrow", "duckdb", "sklearn", "scipy", "requests")


def pytest_report_header(config: pytest.Config) -> str:
    return f"cinesense tests | python {sys.version.split()[0]} | {Path(sys.executable).name}"


def pytest_configure(config: pytest.Config) -> None:
    missing = []
    for name in REQUIRED_MODULES:
        try:
            __import__(name)
        except ImportError:
            missing.append(name)

    if missing:
        raise pytest.UsageError(
            "missing test dependencies: "
            + ", ".join(missing)
            + f"\n  interpreter: {sys.executable}\n"
            "  Run tests through .venv, e.g.\n"
            "    .\\.venv\\Scripts\\python.exe -m pytest"
        )


@pytest.fixture
def movies_frame():
    """A tiny catalogue covering the awkward MovieLens cases."""
    import pandas as pd

    return pd.DataFrame(
        {
            "movie_id": [1, 2, 3, 4],
            "title": [
                "Toy Story (1995)",
                "Matrix, The (1999)",
                "Band of Brothers (2001-2001)",
                "No Year Here",
            ],
            "genres_raw": [
                "Adventure|Animation|Children|Comedy|Fantasy",
                "(no genres listed)",
                "Action|Drama|War",
                "Sci-Fi",
            ],
        }
    )


@pytest.fixture
def tags_frame():
    import pandas as pd

    return pd.DataFrame(
        {
            "movie_id": [1, 12],
            "tag_text": ["pixar toys childhood", "time travel"],
            "n_tags": [3, 2],
        }
    )
