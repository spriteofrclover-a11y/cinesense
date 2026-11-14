"""The manifest is the contract between offline training and serving.

Serving trusts it to know which arrays it is holding, so it must round-trip
exactly and must refuse anything it does not understand.
"""

from __future__ import annotations

import json

import pytest

from cinesense.config import load_settings
from cinesense.data.manifest import (
    MANIFEST_VERSION,
    artifact_entry,
    read_manifest,
    write_manifest,
)


def _build(path, artifact_path):
    return write_manifest(
        path,
        dataset={"name": "test", "slug": "tiny", "url": "https://example.test/x.zip"},
        settings=load_settings(),
        artifacts=[artifact_entry(artifact_path, rows=3)],
        stats={"ratings": 3, "sparsity": 0.9},
    )


def test_artifact_entry_records_size_and_hash(tmp_path):
    artifact = tmp_path / "movies.parquet"
    artifact.write_bytes(b"abc123")

    entry = artifact_entry(artifact, rows=7)

    assert entry["name"] == "movies.parquet"
    assert entry["bytes"] == 6
    assert entry["rows"] == 7
    assert len(entry["sha256"]) == 64


def test_artifact_entry_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        artifact_entry(tmp_path / "absent.parquet")


def test_manifest_roundtrip(tmp_path):
    artifact = tmp_path / "ratings.parquet"
    artifact.write_bytes(b"rows")
    manifest_path = tmp_path / "manifest.json"

    written = _build(manifest_path, artifact)
    loaded = read_manifest(manifest_path)

    assert loaded == written
    assert loaded["manifest_version"] == MANIFEST_VERSION
    assert loaded["total_bytes"] == 4
    assert loaded["stats"]["ratings"] == 3
    assert loaded["config"]["values"]["dataset"]["slug"] == "ml-32m"
    assert loaded["generator"]["cinesense_version"] == "0.1.0"


def test_manifest_is_valid_json_on_disk(tmp_path):
    artifact = tmp_path / "a.parquet"
    artifact.write_bytes(b"x")
    manifest_path = tmp_path / "manifest.json"
    _build(manifest_path, artifact)

    json.loads(manifest_path.read_text(encoding="utf-8"))


def test_unknown_manifest_version_is_rejected(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps({"manifest_version": MANIFEST_VERSION + 99}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="manifest_version"):
        read_manifest(manifest_path)
