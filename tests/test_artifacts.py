import copy
import hashlib
import json
from dataclasses import asdict

import pytest

from autocomplete.artifacts import (
    canonical_json_bytes,
    file_sha256,
    load_development_inputs,
    prediction_content,
    save_evaluation_artifact,
    verify_file_hash,
)
from autocomplete.data import DevelopmentCase
from autocomplete.data.normalization import NORMALIZATION_VERSION


@pytest.fixture
def artifact():
    return {
        "run_id": "example",
        "serving_policy": {"k": 10},
        "results": [{"query_id": 1, "suggestions": ["cat"], "generation_seconds": 0.1}],
        "summary": {"request_count": 1, "success_at_10": 1.0},
        "timing_summary": {"evaluation_seconds": 0.2},
    }


def test_save_creates_canonical_artifact_and_reuses_identical(tmp_path, artifact):
    path = tmp_path / "run" / "results.json"
    assert save_evaluation_artifact(path, artifact) is True
    before = path.stat().st_mtime_ns
    assert path.read_bytes() == canonical_json_bytes(artifact)
    assert save_evaluation_artifact(path, copy.deepcopy(artifact)) is False
    assert path.stat().st_mtime_ns == before


def test_changed_timings_preserve_original_file_and_inputs(tmp_path, artifact):
    path = tmp_path / "result.json"
    save_evaluation_artifact(path, artifact)
    original = path.read_bytes()
    newer = copy.deepcopy(artifact)
    newer["results"][0]["generation_seconds"] = 100
    newer["timing_summary"]["evaluation_seconds"] = 200
    snapshot = copy.deepcopy(newer)
    assert save_evaluation_artifact(path, newer) is False
    assert path.read_bytes() == original
    assert newer == snapshot
    assert "generation_seconds" in artifact["results"][0]
    assert prediction_content(newer) == prediction_content(artifact)


@pytest.mark.parametrize("change", ["prediction", "policy", "summary", "case", "pool"])
def test_semantic_changes_refuse_overwrite(tmp_path, artifact, change):
    path = tmp_path / "result.json"
    save_evaluation_artifact(path, artifact)
    original = path.read_bytes()
    changed = copy.deepcopy(artifact)
    if change == "prediction":
        changed["results"][0]["suggestions"] = ["car"]
    elif change == "policy":
        changed["serving_policy"]["k"] = 1
    elif change == "summary":
        changed["summary"]["success_at_10"] = 0
    elif change == "case":
        changed["results"][0]["query_id"] = 2
    else:
        changed["results"][0]["pool_sha256"] = "different"
    with pytest.raises(FileExistsError):
        save_evaluation_artifact(path, changed)
    assert path.read_bytes() == original


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_json_is_rejected_before_creation(tmp_path, invalid):
    path = tmp_path / "new" / "result.json"
    with pytest.raises(ValueError):
        save_evaluation_artifact(path, {"score": invalid})
    assert not path.exists()
    assert not path.parent.exists()


def test_unreadable_existing_json_is_not_overwritten(tmp_path):
    path = tmp_path / "result.json"
    path.write_bytes(b"unfinished json")
    with pytest.raises(json.JSONDecodeError):
        save_evaluation_artifact(path, {"results": []})
    assert path.read_bytes() == b"unfinished json"


def test_hash_helpers(tmp_path):
    path = tmp_path / "source"
    path.write_bytes(b"sample")
    expected = hashlib.sha256(b"sample").hexdigest()
    assert file_sha256(path) == expected
    verify_file_hash(path, expected)
    with pytest.raises(ValueError, match="Mismatch hash"):
        verify_file_hash(path, "different")


@pytest.fixture
def local_inputs(tmp_path):
    raw = tmp_path / "raw.parquet"
    raw.write_bytes(b"verified input bytes; loader does not parse parquet")
    cases = tmp_path / "cases.json"
    case = DevelopmentCase(1, 0, "ca", "ca", "Cat", "cat")
    cases.write_bytes(canonical_json_bytes([asdict(case)]))
    manifest = {
        "normalization_version": NORMALIZATION_VERSION,
        "counts": {"development_cases": 1},
        "hashes": {
            "raw_file SHA-256": file_sha256(raw),
            "development_cases SHA-256": file_sha256(cases),
        },
    }
    path = tmp_path / "manifest.json"
    path.write_bytes(canonical_json_bytes(manifest))
    return path, raw, cases, manifest, case


def test_load_development_inputs_verifies_and_restores_cases(local_inputs):
    path, raw, cases, manifest, case = local_inputs
    assert load_development_inputs(path, raw, cases) == (manifest, [case])


@pytest.mark.parametrize("changed", ["raw", "cases", "normalization", "count"])
def test_load_development_inputs_rejects_drift(local_inputs, changed):
    path, raw, cases, manifest, _ = local_inputs
    if changed == "raw":
        raw.write_bytes(b"changed")
    elif changed == "cases":
        cases.write_bytes(b"[]")
    elif changed == "normalization":
        manifest["normalization_version"] = "different"
        path.write_bytes(canonical_json_bytes(manifest))
    else:
        manifest["counts"]["development_cases"] = 2
        path.write_bytes(canonical_json_bytes(manifest))
    with pytest.raises(ValueError):
        load_development_inputs(path, raw, cases)
