"""Offline checks for the small optional source-download recipe."""

import hashlib
import json

import pytest

from scripts import download_expanded_history as download


@pytest.fixture
def source(monkeypatch):
    rows = [{"query_id": 1, "text": "café"}, {"query_id": 2, "text": "cat"}]
    payload = b"".join(
        (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8")
        for row in rows
    )
    monkeypatch.setattr(download, "ROWS_PER_SHARD", 2)
    monkeypatch.setattr(download, "EXPECTED_SHA256", {0: hashlib.sha256(payload).hexdigest()})
    monkeypatch.setattr(download, "stream_one_shard", lambda _url: iter(rows))
    return payload


def test_download_matches_expected_bytes_and_reuses_without_network(tmp_path, monkeypatch, source):
    path = download.download_shard(0, tmp_path)
    assert path.read_bytes() == source
    metadata_path = path.with_suffix(".metadata.json")
    metadata = json.loads(metadata_path.read_bytes())
    assert metadata["complete"] is True
    assert metadata["cached_rows"] == 2
    assert metadata["sha256"] == hashlib.sha256(source).hexdigest()
    before = (path.stat().st_mtime_ns, metadata_path.stat().st_mtime_ns)

    def no_network(_url):
        pytest.fail("Verified local inputs must not open a remote stream")

    monkeypatch.setattr(download, "stream_one_shard", no_network)
    assert download.download_shard(0, tmp_path) == path
    assert (path.stat().st_mtime_ns, metadata_path.stat().st_mtime_ns) == before


def test_modified_existing_file_is_rejected_without_overwriting(tmp_path, source):
    path = download.download_shard(0, tmp_path)
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        download.download_shard(0, tmp_path)
    assert path.read_bytes() == b"changed"


@pytest.mark.parametrize("problem", ["short", "wrong_bytes", "network_error"])
def test_failed_download_does_not_publish_complete_files(tmp_path, monkeypatch, source, problem):
    def stream(_url):
        yield {"query_id": 1, "text": "café"}
        if problem == "network_error":
            raise OSError("fixture network failure")
        if problem == "wrong_bytes":
            yield {"query_id": 99, "text": "different"}

    monkeypatch.setattr(download, "stream_one_shard", stream)
    with pytest.raises((ValueError, OSError)):
        download.download_shard(0, tmp_path)
    assert not list(tmp_path.glob("*.jsonl"))
    assert not list(tmp_path.glob("*.metadata.json"))
    assert len(list(tmp_path.glob("*.part"))) == 1


def test_partial_existing_pair_is_preserved(tmp_path, monkeypatch, source):
    path = tmp_path / "shard_00000_first_2.jsonl"
    path.write_bytes(b"existing input without metadata")
    with pytest.raises(FileExistsError, match="Incomplete existing"):
        download.download_shard(0, tmp_path)
    assert path.read_bytes() == b"existing input without metadata"
