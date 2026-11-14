"""Report peak memory and wall time for a cold catalogue build.

The deployment budget assumes the offline build stays well under ~2 GB, because
it must never page on a 8 GB machine while other work is running. Run this after
clearing data/processed:

    Remove-Item -Recurse -Force data\\processed\\ml-32m
    .\\.venv\\Scripts\\python.exe scripts/measure_build.py

Baseline at Phase 1 (ml-32m, cold ingest, archive already extracted):
    peak RSS 871 MB, wall 7.9 s
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import psutil  # noqa: E402

from cinesense.config import load_settings  # noqa: E402
from cinesense.data.build import build_catalog  # noqa: E402

BUDGET_MB = 2048


def main() -> int:
    process = psutil.Process()
    baseline = process.memory_info().rss / (1024 * 1024)

    started = time.perf_counter()
    built = build_catalog(load_settings(), force=False)
    elapsed = time.perf_counter() - started

    peak = process.memory_info().peak_wset / (1024 * 1024)

    print()
    print(f"  baseline RSS   {baseline:8.0f} MB")
    print(f"  peak RSS       {peak:8.0f} MB   (budget {BUDGET_MB} MB)")
    print(f"  wall clock     {elapsed:8.1f} s")
    print(f"  ratings        {built.stats['ratings']:>12,}")
    print(f"  movies         {built.stats['rows']['movies']:>12,}")
    print(f"  sparsity       {built.stats['sparsity']:>12.4%}")

    if peak > BUDGET_MB:
        print(f"\n  OVER BUDGET by {peak - BUDGET_MB:.0f} MB")
        return 1

    print("\n  within budget\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
