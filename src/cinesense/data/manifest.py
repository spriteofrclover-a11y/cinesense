"""Build provenance: what was built, from what, and can we prove it.

Every artifact gets a size, row count and sha256. Two reasons:

* Reproducibility. If a model behaves differently a week later, the manifest
  says whether the data changed or the code did.
* Deployment. The serving process loads ``manifest.json`` to verify the arrays
  it finds are the ones that were built, rather than trusting filenames.
"""

from __future__ import annotations

import json
import logging
import platform
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from cinesense.data.download import sha256_file

__all__ = ["artifact_entry", "read_manifest", "write_manifest"]

log = logging.getLogger(__name__)

MANIFEST_VERSION = 1


def artifact_entry(path: Path | str, *, rows: int | None = None) -> dict[str, Any]:
    """Describe one output file for the manifest."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"cannot describe missing artifact {path}")

    entry: dict[str, Any] = {
        "name": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if rows is not None:
        entry["rows"] = int(rows)
    return entry


def write_manifest(
    path: Path | str,
    *,
    dataset: Mapping[str, Any],
    settings: Any,
    artifacts: list[Mapping[str, Any]],
    stats: Mapping[str, Any],
    config_path: Path | str | None = None,
) -> dict[str, Any]:
    """Assemble and persist the manifest, returning it for reporting."""
    path = Path(path)
    payload: dict[str, Any] = {
        "manifest_version": MANIFEST_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generator": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "executable": sys.executable,
            "cinesense_version": _package_version(),
        },
        "dataset": dict(dataset),
        "config": {"path": str(config_path) if config_path else None, "values": asdict(settings)},
        "stats": dict(stats),
        "artifacts": list(artifacts),
        "total_bytes": sum(int(a["bytes"]) for a in artifacts),
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    log.info("manifest written: %s", path)
    return payload


def read_manifest(path: Path | str) -> dict[str, Any]:
    """Load a manifest, rejecting anything written by a newer schema."""
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    version = payload.get("manifest_version")
    if version != MANIFEST_VERSION:
        raise ValueError(
            f"{path} has manifest_version={version!r}, expected {MANIFEST_VERSION}. "
            "Re-run `cinesense build-catalog --force`."
        )
    return payload


def _package_version() -> str:
    from cinesense import __version__

    return __version__
