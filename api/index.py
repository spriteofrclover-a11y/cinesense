"""Vercel entrypoint.

Vercel installs ``requirements.txt`` but never installs *this* project, so the
package has to be importable from the repository's ``src/`` directory. Doing it
here also keeps ``PROJECT_ROOT`` correct: ``cinesense/config.py`` resolves the
repo root from its own path, which only works while the package runs from
``src/cinesense/`` rather than from site-packages.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cinesense.serving.app import create_app  # noqa: E402

app = create_app()
