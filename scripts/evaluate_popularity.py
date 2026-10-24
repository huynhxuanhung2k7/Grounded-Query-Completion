"""Evaluate the H_0 popularity baseline on saved September development cases.

Run from the project directory with
``uv run python -m scripts.evaluate_popularity``. Verify source-file fingerprints
before loading data, build historical terms/counts from the earlier partition,
and save predictions plus normalized exact-match scores under ``artifacts/``.
This command performs no model training or official October evaluation.
"""

import hashlib
import json
from pathlib import Path

from autocomplete.catalog import build_catalog, count_history_terms
from autocomplete.contract import MAX_ADDED_LENGTH, MAX_INPUT_LENGTH
from autocomplete.data import DevelopmentCase, partition_record
from autocomplete.evaluation import evaluate_popularity
from datasets import Dataset
from scripts.prepare_data import MANIFEST, OUTPUT
from scripts.probe_amazonqac import RAW_DATA


def verify_file_hash(path: Path, expected_hash: str) -> None:
    """Require a file's exact bytes to match its manifest SHA-256 fingerprint.

    Args:
        path: Raw Parquet or saved development-case JSON file to verify.
        expected_hash: Expected hexadecimal SHA-256 digest from the data manifest.

    Raises:
        ValueError: The calculated digest differs from the expected digest.
        FileNotFoundError: The input file does not exist.

    Other file-reading errors propagate. This verifies file identity, not schema
    validity; chronological validation is performed by ``partition_record``.
    """
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
        raise ValueError(f"Mismatch hash on {path}")


def main() -> None:
    """Verify inputs, evaluate popularity and safely persist development results.

    Read the versioned manifest and verify both cached input hashes. Reconstruct
    H_0 from ``partition.train`` only and build its normalized catalog and
    occurrence counts; supplied popularity metadata is not used. Load the fixed
    JSON cases as ``DevelopmentCase`` objects, retaining misses in both metric
    denominators. Recorded answers are passed only to the evaluator.

    Write ``artifacts/popularity_h0/development_results.json`` with the run ID,
    manifest SHA-256, per-request rows and Success@10/MRR@10 summary. Identical
    output is reused without writing; conflicting output raises
    ``FileExistsError`` and remains untouched. Print counts and summary metrics.

    File, JSON and data-validation errors propagate rather than producing a
    partial success result. The command assumes project-relative input paths.
    """
    manifest = json.loads(MANIFEST.read_bytes())
    verify_file_hash(RAW_DATA, manifest["hashes"]["raw_file SHA-256"])
    verify_file_hash(OUTPUT, manifest["hashes"]["development_cases SHA-256"])

    raw_data = Dataset.from_parquet(str(RAW_DATA))
    partition = partition_record(raw_data.to_list())

    catalog = build_catalog(partition.train)
    count_history = count_history_terms(partition.train)

    saved_rows = json.loads(OUTPUT.read_bytes())
    cases = [DevelopmentCase(**row) for row in saved_rows]

    print(f"{'Historical interactions:':<25}{len(partition.train)}")
    print(f"{'Catalog terms:':<25}{len(catalog)}")
    print(f"{'Counted terms:':<25}{len(count_history)}")
    print(f"{'Total occurrences:':<25}{sum(count_history.values())}")
    print(f"{'Development cases:':<25}: {len(cases)}")
    print(isinstance(cases[0], DevelopmentCase))

    evaluation = evaluate_popularity(catalog, count_history, cases)
    
    run_id = "popularity_h0_contract_v1"
    artifact = {
        "run_id": run_id,
        "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        "serving_policy": {
            "version": "contract-v1",
            "max_input_length": MAX_INPUT_LENGTH,
            "max_added_length": MAX_ADDED_LENGTH,
            "k": 10,
        },
        **evaluation,
    }

    result_path = Path("artifacts") / run_id / "development_results.json"
    payload = json.dumps(
        artifact,
        sort_keys=True,
        ensure_ascii=False,
        indent=2,
    ).encode("utf-8")

    result_path.parent.mkdir(parents=True, exist_ok=True)
    if result_path.exists():
        if result_path.read_bytes() != payload:
            raise FileExistsError(
                f"Refusing to overwrite different results: {result_path}"
            )
    else:
        result_path.write_bytes(payload)

    print(json.dumps(evaluation["summary"], indent=2))


if __name__ == "__main__":
    main()
