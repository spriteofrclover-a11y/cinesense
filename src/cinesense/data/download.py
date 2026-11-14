"""Resumable, checksum-verified downloads.

Deliberately uses ``requests`` rather than ``urllib``: the stdlib CA bundle on
this machine is broken, and ``requests`` ships certifi.
"""

from __future__ import annotations

import hashlib
import logging
import time
import zipfile
from pathlib import Path

import requests

__all__ = [
    "DownloadError",
    "download",
    "extract_zip",
    "sha256_file",
]

log = logging.getLogger(__name__)

CHUNK_SIZE = 1 << 20  # 1 MiB
REPORT_EVERY = 0.10  # log progress at 10% intervals


class DownloadError(RuntimeError):
    """A remote file could not be fetched, extracted, or failed verification."""


def sha256_file(path: Path | str, chunk_size: int = CHUNK_SIZE) -> str:
    """Hex digest of a file, streamed so 240 MB never lands in memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _human_bytes(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def download(
    url: str,
    dest: Path | str,
    *,
    expected_sha256: str | None = None,
    retries: int = 3,
    timeout: int = 60,
    force: bool = False,
    session: requests.Session | None = None,
) -> Path:
    """Fetch ``url`` to ``dest``, reusing an existing verified copy.

    Downloading is the slowest, flakiest step in the pipeline, so this function
    is written to be interrupted and re-run: bytes go to ``*.part`` and are only
    promoted once the checksum matches.

    Returns the path to the completed file.
    """
    dest = Path(dest)

    if dest.exists() and not force:
        if not expected_sha256:
            log.info("cached: %s (%s, unverified)", dest.name, _human_bytes(dest.stat().st_size))
            return dest
        actual = sha256_file(dest)
        if actual == expected_sha256:
            log.info("cached: %s (sha256 verified)", dest.name)
            return dest
        log.warning("checksum mismatch on cached %s; re-downloading", dest.name)
        dest.unlink()

    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    http = session or requests.Session()
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            log.info("downloading %s (attempt %d/%d)", url, attempt, retries)
            with http.get(url, stream=True, timeout=timeout) as response:
                response.raise_for_status()
                total = int(response.headers.get("content-length") or 0)
                written = 0
                next_mark = REPORT_EVERY

                with part.open("wb") as out:
                    for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                        if not chunk:
                            continue
                        out.write(chunk)
                        written += len(chunk)
                        if total:
                            fraction = written / total
                            if fraction >= next_mark:
                                log.info(
                                    "  %3d%%  %s / %s",
                                    int(fraction * 100),
                                    _human_bytes(written),
                                    _human_bytes(total),
                                )
                                next_mark += REPORT_EVERY

            if expected_sha256:
                actual = sha256_file(part)
                if actual != expected_sha256:
                    raise DownloadError(
                        f"sha256 mismatch for {url}\n  expected {expected_sha256}\n  actual   {actual}"
                    )
            else:
                log.info("sha256 of %s = %s", dest.name, sha256_file(part))

            part.replace(dest)
            log.info("saved %s (%s)", dest.name, _human_bytes(dest.stat().st_size))
            return dest

        except (requests.RequestException, DownloadError, OSError) as exc:
            last_error = exc
            part.unlink(missing_ok=True)
            if attempt < retries:
                backoff = 2 ** attempt
                log.warning("attempt %d failed (%s); retrying in %ds", attempt, exc, backoff)
                time.sleep(backoff)

    raise DownloadError(f"could not download {url} after {retries} attempts: {last_error}")


def extract_zip(
    archive: Path | str,
    dest_dir: Path | str,
    *,
    force: bool = False,
) -> Path:
    """Extract ``archive`` into ``dest_dir``, skipping work already done.

    A ``.extracted`` marker records the archive's size so a re-run is a no-op
    instead of re-unpacking 1.2 GB.
    """
    archive = Path(archive)
    dest_dir = Path(dest_dir)
    marker = dest_dir / ".extracted"

    if marker.exists() and not force:
        log.info("cached: %s already extracted", dest_dir.name)
        return dest_dir

    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(archive) as bundle:
            members = bundle.infolist()
            log.info("extracting %d files from %s", len(members), archive.name)
            bundle.extractall(dest_dir)
    except zipfile.BadZipFile as exc:
        raise DownloadError(f"{archive} is not a valid zip archive") from exc

    marker.write_text(f"{archive.name} {archive.stat().st_size}\n", encoding="utf-8")
    return dest_dir
