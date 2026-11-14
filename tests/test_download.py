"""Download and extraction behaviour, tested without touching the network.

The cache path matters: re-running the pipeline must never re-download 240 MB,
and a corrupted cache must never be silently accepted.
"""

from __future__ import annotations

import hashlib
import zipfile

import pytest

from cinesense.data.download import DownloadError, download, extract_zip, sha256_file

#: Unroutable on purpose. Any real network attempt here would hang or fail, so a
#: passing test proves the cache short-circuited before dialling out.
NOWHERE = "http://127.0.0.1:9/never.zip"


def test_sha256_file_matches_hashlib(tmp_path):
    path = tmp_path / "x.bin"
    payload = b"cinesense" * 1000
    path.write_bytes(payload)
    assert sha256_file(path) == hashlib.sha256(payload).hexdigest()


def test_cached_file_with_matching_checksum_is_reused(tmp_path):
    destination = tmp_path / "cache.zip"
    destination.write_bytes(b"pretend dataset")
    expected = sha256_file(destination)

    result = download(NOWHERE, destination, expected_sha256=expected, retries=1)

    assert result == destination
    assert destination.read_bytes() == b"pretend dataset"


def test_cached_file_without_checksum_is_reused(tmp_path):
    destination = tmp_path / "cache.zip"
    destination.write_bytes(b"whatever")

    assert download(NOWHERE, destination, retries=1) == destination


def test_corrupt_cache_is_rejected_and_fails_cleanly(tmp_path):
    destination = tmp_path / "cache.zip"
    destination.write_bytes(b"stale bytes")

    with pytest.raises(DownloadError, match="could not download"):
        download(NOWHERE, destination, expected_sha256="0" * 64, retries=1, timeout=2)

    # A failed attempt must not leave a poisoned cache or a stray .part file.
    assert not destination.exists()
    assert not list(tmp_path.glob("*.part"))


def test_extract_zip_is_idempotent(tmp_path):
    archive = tmp_path / "data.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("ml-test/movies.csv", "movieId,title\n1,Toy Story\n")

    target = tmp_path / "out"
    extract_zip(archive, target)

    assert (target / "ml-test" / "movies.csv").exists()
    assert (target / ".extracted").exists()

    # Second call must short-circuit; deleting the payload proves it did not
    # re-extract (the marker alone is what it checks).
    (target / "ml-test" / "movies.csv").unlink()
    extract_zip(archive, target)
    assert not (target / "ml-test" / "movies.csv").exists()


def test_extract_zip_rejects_non_zip(tmp_path):
    archive = tmp_path / "not.zip"
    archive.write_bytes(b"definitely not a zip")

    with pytest.raises(DownloadError, match="not a valid zip"):
        extract_zip(archive, tmp_path / "out")
