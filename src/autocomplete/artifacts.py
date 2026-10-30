"""Verified evaluation inputs and non-overwriting JSON artifact storage."""

import hashlib
import json
from pathlib import Path

from autocomplete.data import DevelopmentCase
from autocomplete.data.normalization import NORMALIZATION_VERSION


def file_sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def verify_file_hash(path: Path, expected_hash: str) -> None:
    if file_sha256(path) != expected_hash:
        raise ValueError(f"Mismatch hash on {path}")


def load_development_inputs(
    manifest_path: Path,
    raw_data_path: Path,
    cases_path: Path,
) -> tuple[dict, list[DevelopmentCase]]:
    """Verify local source/case fingerprints before constructing frozen cases."""
    manifest = json.loads(manifest_path.read_bytes())
    if manifest["normalization_version"] != NORMALIZATION_VERSION:
        raise ValueError("Manifest normalization version differs")

    verify_file_hash(raw_data_path, manifest["hashes"]["raw_file SHA-256"])
    verify_file_hash(cases_path, manifest["hashes"]["development_cases SHA-256"])

    cases = [DevelopmentCase(**row) for row in json.loads(cases_path.read_bytes())]

    if len(cases) != manifest["counts"]["development_cases"]:
        raise ValueError("Development-case count differs from manifest")

    return manifest, cases


def canonical_json_bytes(value: object) -> bytes:
    """Use one strict, stable encoding for persisted JSON and content hashes."""
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False
    ).encode("utf-8")


def save_json(path: Path, value: object) -> bool:
    """Write a small JSON file; reuse equal content, never replace different data."""
    payload = canonical_json_bytes(value)
    if path.exists():
        if json.loads(path.read_bytes()) != value:
            raise FileExistsError(f"Refusing to overwrite different content: {path}")
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return True


def prediction_content(artifact: dict) -> dict:
    """Exclude only documented variable timing fields, without mutating input."""
    content = dict(artifact)
    content.pop("timing_summary", None)
    content.pop("p95_ms", None)

    if "results" in content:
        content["results"] = [
            {key: value for key, value in row.items() if key != "generation_seconds"}
            for row in content["results"]
        ]

    return content


def save_evaluation_artifact(path: Path, artifact: dict) -> bool:
    """Create exclusively; reuse identical predictions without replacing old timings.

    Returns True for a new file and False for identical reuse. Differences other
    than the documented timing fields raise FileExistsError. Never overwrites.
    """
    payload = canonical_json_bytes(artifact)
    if path.exists():
        existing = json.loads(path.read_bytes())
        if prediction_content(existing) != prediction_content(artifact):
            raise FileExistsError(f"Refusing to overwrite different results: {path}")

        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as output:
        output.write(payload)

    return True


if __name__ == "__main__":
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as directory:
        path = Path(directory) / "summary.json"
        result = {"request_count": 2, "success_at_10": 0.5}
        print("First save creates a file:", save_json(path, result))
        print("Identical save reuses it:", save_json(path, result))
        digest = file_sha256(path)
        verify_file_hash(path, digest)
        print("Verified SHA-256:", digest)
        try:
            save_json(path, {"request_count": 3})
        except FileExistsError:
            print("Different content: overwrite refused")
        print("Saved content remains:", json.loads(path.read_text()))
