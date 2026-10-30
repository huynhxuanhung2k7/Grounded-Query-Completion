import json
from dataclasses import asdict

import pytest

from autocomplete.artifacts import (
    canonical_json_bytes,
    file_sha256,
    load_development_inputs,
    save_evaluation_artifact,
    save_json,
    verify_file_hash,
)
from autocomplete.data import DevelopmentCase
from autocomplete.data.normalization import NORMALIZATION_VERSION


def test_json_save_reuses_identical_content_and_refuses_replacement(tmp_path):
    path = tmp_path / "summary.json"
    value = {"request_count": 2}

    assert save_json(path, value) is True
    original = path.read_bytes()
    modified = path.stat().st_mtime_ns

    assert save_json(path, value) is False
    assert path.stat().st_mtime_ns == modified

    with pytest.raises(FileExistsError):
        save_json(path, {"request_count": 3})

    assert path.read_bytes() == original


def test_identical_evaluation_preserves_the_original_timing(tmp_path):
    path = tmp_path / "summary.json"
    result = {"request_count": 2, "success_at_10": 0.5, "p95_ms": 10}
    save_evaluation_artifact(path, result)
    original = path.read_bytes()

    assert save_evaluation_artifact(path, {**result, "p95_ms": 99}) is False
    assert path.read_bytes() == original


def test_changed_evaluation_scores_cannot_overwrite_evidence(tmp_path):
    path = tmp_path / "summary.json"
    save_evaluation_artifact(path, {"success_at_10": 0.5})

    with pytest.raises(FileExistsError):
        save_evaluation_artifact(path, {"success_at_10": 0.6})

    assert json.loads(path.read_text()) == {"success_at_10": 0.5}


def test_development_inputs_are_restored_only_when_hashes_match(tmp_path):
    raw = tmp_path / "raw.parquet"
    raw.write_bytes(b"tiny source fingerprint fixture")
    case = DevelopmentCase(1, 0, "ca", "ca", "cat", "cat")
    cases = tmp_path / "cases.json"
    cases.write_bytes(canonical_json_bytes([asdict(case)]))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "normalization_version": NORMALIZATION_VERSION,
                "counts": {"development_cases": 1},
                "hashes": {
                    "raw_file SHA-256": file_sha256(raw),
                    "development_cases SHA-256": file_sha256(cases),
                },
            }
        )
    )

    assert load_development_inputs(manifest, raw, cases)[1] == [case]

    raw.write_bytes(b"changed source")

    with pytest.raises(ValueError, match="Mismatch hash"):
        load_development_inputs(manifest, raw, cases)


def test_canonical_json_and_hash_verification(tmp_path):
    assert canonical_json_bytes({"b": 2, "a": 1}) == canonical_json_bytes(
        {"a": 1, "b": 2}
    )
    path = tmp_path / "value.json"
    path.write_bytes(canonical_json_bytes({"value": 1}))

    assert verify_file_hash(path, file_sha256(path)) is None

    with pytest.raises(ValueError):
        canonical_json_bytes({"value": float("nan")})
