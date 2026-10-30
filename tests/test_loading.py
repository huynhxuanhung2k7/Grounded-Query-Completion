import json

import pytest

import datasets
from autocomplete.artifacts import file_sha256
from autocomplete.config import REVISION
from autocomplete.data.loading import (
    load_partition,
    normalized_terms,
    read_jsonl,
    read_parquet,
    read_source,
)
from autocomplete.data.normalization import NORMALIZATION_VERSION
from datasets import Dataset


def test_parquet_loading_and_normalized_terms(tmp_path, monkeypatch):
    monkeypatch.setattr(datasets.config, "HF_DATASETS_CACHE", tmp_path / "cache")
    path = tmp_path / "records.parquet"
    Dataset.from_list([{"final_search_term": " CAT Food ", "extra": 1}]).to_parquet(
        str(path)
    )

    rows = read_parquet(path, columns=["final_search_term"])

    assert rows == [{"final_search_term": " CAT Food "}]
    assert normalized_terms(rows) == ["cat food"]
    assert normalized_terms([]) == []


def test_jsonl_receipt_detects_changed_source(tmp_path):
    path = tmp_path / "records.jsonl"
    path.write_text('{"final_search_term": "cat"}\n')
    receipt = {
        "dataset_revision": REVISION,
        "sha256": file_sha256(path),
        "cached_rows": 1,
    }
    path.with_suffix(".metadata.json").write_text(json.dumps(receipt))

    assert read_source(path) == [{"final_search_term": "cat"}]

    path.write_text('{"final_search_term": "dog"}\n')

    with pytest.raises(ValueError, match="Mismatch hash"):
        read_source(path)


def test_malformed_jsonl_reports_the_line_number(tmp_path):
    path = tmp_path / "records.jsonl"
    path.write_text("{}\nnot json\n")

    with pytest.raises(ValueError, match=":2"):
        read_jsonl(path)


def test_partition_checks_the_manifest_before_splitting(tmp_path, monkeypatch):
    monkeypatch.setattr(datasets.config, "HF_DATASETS_CACHE", tmp_path / "cache")
    path = tmp_path / "records.parquet"
    record = {
        "query_id": 1,
        "session_id": "toy",
        "prefixes": ["ca"],
        "final_search_term": "cat",
        "popularity": 1,
        "first_prefix_typed_time": "2023-09-24 12:00:00",
        "search_time": "2023-09-24 12:00:01",
    }
    Dataset.from_list([record]).to_parquet(str(path))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "normalization_version": NORMALIZATION_VERSION,
                "hashes": {"raw_file SHA-256": file_sha256(path)},
            }
        )
    )

    partition = load_partition(path, manifest)

    assert partition.train == [record]
    assert partition.development == []
