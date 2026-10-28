"""Synthetic offline checks; never download, train or open official-test data."""

import hashlib
import json

import pytest

from autocomplete import predictors
from autocomplete.artifacts import file_sha256, save_evaluation_artifact
from autocomplete.data import CUTOFF
from autocomplete.data.normalization import NORMALIZATION_VERSION
from datasets import Dataset
from scripts import process_data as build


def record(query_id, term="cat", *, first="2023-09-01", final="2023-09-01"):
    return {
        "query_id": query_id,
        "session_id": str(query_id),
        "prefixes": ["c"],
        "first_prefix_typed_time": f"{first} 00:00:00",
        "final_search_term": term,
        "search_time": f"{final} 00:00:01",
        "popularity": 9999,
    }


def source(tmp_path, rows, *, index=0, payload=None):
    path = tmp_path / f"shard_{index}.jsonl"
    payload = (
        payload
        if payload is not None
        else b"".join((json.dumps(row) + "\n").encode() for row in rows)
    )
    path.write_bytes(payload)
    metadata = {
        "dataset_revision": build.REVISION,
        "cached_rows": len(rows),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    path.with_suffix(".metadata.json").write_text(json.dumps(metadata))
    return path


def original(tmp_path, rows):
    path = tmp_path / "original.parquet"
    Dataset.from_list(rows).to_parquet(str(path))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "dataset": {"revision": build.REVISION},
                "normalization_version": NORMALIZATION_VERSION,
                "cutoff": CUTOFF.isoformat(),
                "hashes": {"raw_file SHA-256": file_sha256(path)},
            }
        )
    )
    return {"original_path": path, "original_manifest": manifest}


def test_catalog_loader_defaults_to_processed_history(tmp_path, monkeypatch):
    path = tmp_path / "catalog.json"
    path.write_text(
        json.dumps(
            {
                "normalization_version": NORMALIZATION_VERSION,
                "cutoff": CUTOFF.isoformat(),
                "term_counts": {"cat": 3, "cart": 2},
            }
        )
    )
    monkeypatch.setattr(predictors, "HISTORY_CATALOG", path)

    assert predictors.load_catalog() == ({"cat", "cart"}, {"cat": 3, "cart": 2})


@pytest.mark.parametrize("field", ["normalization_version", "cutoff"])
def test_catalog_loader_rejects_incompatible_history(tmp_path, monkeypatch, field):
    path = tmp_path / "catalog.json"
    bundle = {
        "normalization_version": NORMALIZATION_VERSION,
        "cutoff": CUTOFF.isoformat(),
        "term_counts": {"cat": 1},
    }
    bundle[field] = "different"
    path.write_text(json.dumps(bundle))
    monkeypatch.setattr(predictors, "HISTORY_CATALOG", path)
    with pytest.raises(ValueError, match="does not match"):
        predictors.load_catalog()


def test_catalog_loader_missing_history_does_not_fall_back(tmp_path, monkeypatch):
    monkeypatch.setattr(predictors, "HISTORY_CATALOG", tmp_path / "missing.json")
    with pytest.raises(FileNotFoundError):
        predictors.load_catalog()


def test_catalog_loader_explicit_raw_path_keeps_original_split(tmp_path):
    options = original(
        tmp_path,
        [
            record(1, "cat"),
            record(2, "later-only", first="2023-09-26", final="2023-09-26"),
        ],
    )
    assert predictors.load_catalog(
        options["original_path"], options["original_manifest"]
    ) == ({"cat"}, {"cat": 1})


def test_every_line_is_loaded_and_chronology_is_applied(tmp_path):
    path = source(
        tmp_path,
        [
            record(1),
            record(2, first="2023-09-26", final="2023-09-26"),
            record(3, first="2023-09-24", final="2023-09-25"),
            record(4),
        ],
    )
    result = build.load_history_sources([path])
    assert [r["query_id"] for r in result.train] == [1, 4]
    assert [r["query_id"] for r in result.development] == [2]
    assert [(r["query_id"], reason) for r, reason in result.quarantine] == [
        (3, "crosses_cutoff")
    ]


@pytest.mark.parametrize(
    "field,value",
    [
        ("dataset_revision", "different"),
        ("cached_rows", -1),
        ("cached_rows", 2),
        ("sha256", "wrong"),
    ],
)
def test_bad_metadata_or_hash_is_rejected(tmp_path, field, value):
    path = source(tmp_path, [record(1)])
    metadata_path = path.with_suffix(".metadata.json")
    metadata = json.loads(metadata_path.read_bytes())
    metadata[field] = value
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        build.load_history_sources([path])


@pytest.mark.parametrize(
    "payload,error", [(b"broken\n", ValueError), (b"[]\n", TypeError)]
)
def test_invalid_json_record_is_rejected(tmp_path, payload, error):
    path = source(tmp_path, [None], payload=payload)
    with pytest.raises(error):
        build.load_history_sources([path])


def test_empty_source_and_duplicate_ids(tmp_path):
    empty = source(tmp_path, [])
    assert build.load_history_sources([empty]).train == []
    first = source(tmp_path, [record(1)])
    second = source(tmp_path, [record(1)], index=6)
    with pytest.raises(ValueError, match="duplicated"):
        build.load_history_sources([first, second])


def test_bundle_preserves_h0_counts_repeated_terms_and_excludes_later_truth(tmp_path):
    early = record(1)
    later = record(2, "later-only", first="2023-09-26", final="2023-09-26")
    options = original(tmp_path, [early, later])
    path = source(
        tmp_path,
        [
            early,
            later,
            record(3, " CAT "),
            record(4, "dog"),
            record(5, "   "),
            record(6, "quarantine-only", first="2023-09-24", final="2023-09-25"),
        ],
    )
    bundle = build.build_expanded_catalog([path], **options)
    assert bundle["history_query_ids"] == [1, 3, 4, 5]
    assert bundle["term_counts"] == {"cat": 2, "dog": 1}
    assert bundle["counts"] == {
        "source_rows": 6,
        "eligible_history_interactions": 4,
        "later_records": 1,
        "quarantined_records": 1,
        "h0_eligible_interactions": 1,
        "additional_eligible_interactions": 3,
        "unique_terms": 2,
        "h0_unique_terms": 1,
        "new_unique_terms": 1,
        "counted_nonempty_interactions": 3,
    }
    assert bundle["quarantine_reasons"] == {"crosses_cutoff": 1}
    output = tmp_path / "catalog.json"
    assert save_evaluation_artifact(output, bundle)
    before = output.read_bytes(), output.stat().st_mtime_ns
    assert not save_evaluation_artifact(
        output, build.build_expanded_catalog([path], **options)
    )
    assert (output.read_bytes(), output.stat().st_mtime_ns) == before
    bundle["term_counts"]["new"] = 1
    with pytest.raises(FileExistsError):
        save_evaluation_artifact(output, bundle)


@pytest.mark.parametrize("changed", [record(99), record(1, "changed")])
def test_missing_or_changed_h0_record_is_rejected(tmp_path, changed):
    options = original(tmp_path, [record(1)])
    path = source(tmp_path, [changed])
    with pytest.raises(ValueError, match="H_0"):
        build.build_expanded_catalog([path], **options)


def test_original_hash_must_match_before_building(tmp_path):
    options = original(tmp_path, [record(1)])
    options["original_path"].write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash"):
        build.build_expanded_catalog([], **options)
