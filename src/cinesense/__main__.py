"""Allow ``python -m cinesense``."""

from __future__ import annotations

import sys

from cinesense.cli import main

if __name__ == "__main__":
    sys.exit(main())
