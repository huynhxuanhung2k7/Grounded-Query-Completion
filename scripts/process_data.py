"""Process manually downloaded AmazonQAC files; never download or train.

Run from project/: uv run python -m scripts.process_data --dataset development
"""

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from autocomplete.artifacts import (
    canonical_json_bytes,
    file_sha256,
    save_json,
    verify_file_hash,
)
from autocomplete.baseline.catalog import count_history_terms
from autocomplete.config import (
    HISTORY_CATALOG,
    HISTORY_ROOT,
    HISTORY_ROWS_PER_SHARD,
    INPUT,
    MANIFEST,
    OFFICIAL_CASES,
    OFFICIAL_MANIFEST,
    OFFICIAL_RAW,
    OUTPUT,
    REVISION,
    SAMPLE_ROOT,
    SAMPLE_ROWS_PER_SHARD,
    SHARD_INDICES,
)
from autocomplete.data import (
    CUTOFF,
    DevelopmentCase,
    PartitionResult,
    make_development_cases,
    partition_record,
    write_development_cases,
)
from autocomplete.data.loading import load_partition, read_parquet, read_source
from autocomplete.data.normalization import (
    NORMALIZATION_VERSION,
    normalize_prefix,
    normalize_term,
)
from datasets import Dataset

SEED = 42
SOURCE_FILE = "test/part-00000-ba5fc3b2-05ee-452d-bf3d-84a2fd6a463a-c000.snappy.parquet"
EXPECTED_SHA256 = "0cc23fb9d7d63e2681bddbc47d57b44774f7fc90d5da448e91aea17f0b78b3d4"
EXPECTED_ROWS = 20_000


def shard_path(root: Path, index: int, count: int) -> Path:
    return root / f"shard_{index:05d}_first_{count}.jsonl"


def combine_sample() -> None:
    """Create the raw Parquet once; never replace the original training source."""
    if INPUT.exists():
        return

    records = []
    for index in SHARD_INDICES:
        path = shard_path(SAMPLE_ROOT, index, SAMPLE_ROWS_PER_SHARD)
        rows = read_source(path)

        if len(rows) != SAMPLE_ROWS_PER_SHARD:
            raise ValueError(f"Expected {SAMPLE_ROWS_PER_SHARD} records: {path}")

        records.extend(rows)
    partition_record(records)  # Reject invalid schema/duplicate IDs before saving.
    INPUT.parent.mkdir(parents=True, exist_ok=True)
    # Keep the original writer so cached checkpoints still match the raw-file hash.
    Dataset.from_list(records).to_parquet(str(INPUT))


def prepare_development() -> None:
    combine_sample()
    # Existing manifests remain byte-for-byte unchanged for old checkpoints.
    partition = load_partition(INPUT, MANIFEST if MANIFEST.exists() else None)
    cases = make_development_cases(partition, SEED)
    case_hash = write_development_cases(OUTPUT, cases)
    counts = {
        "source": len(partition.train)
        + len(partition.development)
        + len(partition.quarantine),
        "train": len(partition.train),
        "development_cases": len(cases),
        "quarantine": len(partition.quarantine),
    }
    if MANIFEST.exists():
        existing = json.loads(MANIFEST.read_text())
        if (
            existing["hashes"]["development_cases SHA-256"] != case_hash
            or existing["counts"] != counts
            or existing["development_prefix_seed"] != SEED
            or existing["cutoff"] != CUTOFF.isoformat()
        ):
            raise ValueError("Prepared data differs from the saved manifest")
    else:
        save_json(
            MANIFEST,
            {
                "dataset": {"name": "amazon/AmazonQAC", "revision": REVISION},
                "shard": {
                    "shard_index": list(SHARD_INDICES),
                    "per_shard capacity": SAMPLE_ROWS_PER_SHARD,
                },
                "development_prefix_seed": SEED,
                "cutoff": CUTOFF.isoformat(),
                "normalization_version": NORMALIZATION_VERSION,
                "counts": counts,
                "quarantine_reasons": dict(
                    Counter(reason for _, reason in partition.quarantine)
                ),
                "hashes": {
                    "raw_file SHA-256": file_sha256(INPUT),
                    "development_cases SHA-256": case_hash,
                },
            },
        )
    print(json.dumps(counts, indent=2))
    print(f"Cases: {OUTPUT}\nManifest: {MANIFEST}")


def load_history_sources(paths: list[Path]) -> PartitionResult:
    records = []
    for path in paths:
        rows = read_source(path)
        records.extend(rows)
        print(f"Loaded {len(rows)} rows: {path.name}")
    return partition_record(records)  # Reject duplicate IDs; apply the same cutoff.


def build_expanded_catalog(
    paths: list[Path],
    *,
    original_path: Path = INPUT,
    original_manifest: Path = MANIFEST,
) -> dict:
    manifest = json.loads(original_manifest.read_text())
    if (
        manifest["dataset"]["revision"] != REVISION
        or manifest["normalization_version"] != NORMALIZATION_VERSION
        or manifest["cutoff"] != CUTOFF.isoformat()
    ):
        raise ValueError("Original sample uses different data/split settings")
    verify_file_hash(original_path, manifest["hashes"]["raw_file SHA-256"])
    original_rows = read_parquet(original_path)
    original = partition_record(original_rows)
    partition = load_history_sources(paths)
    all_rows = [
        *partition.train,
        *partition.development,
        *(row for row, _ in partition.quarantine),
    ]
    by_id = {row["query_id"]: row for row in all_rows}
    # The 100K selection includes the 20K sample; never concatenate it a second time.
    for row in original_rows:
        if by_id.get(row["query_id"]) != row:
            raise ValueError(
                f"Expanded source omits or changes original H_0/sample ID {row['query_id']}"
            )
    counts = count_history_terms(partition.train)
    old_counts = count_history_terms(original.train)
    return {
        "history_id": f"history_source_{len(all_rows)}",
        "dataset": {"name": "amazon/AmazonQAC", "revision": REVISION},
        "cutoff": CUTOFF.isoformat(),
        "normalization_version": NORMALIZATION_VERSION,
        "sources": [{"path": str(path), "sha256": file_sha256(path)} for path in paths],
        "counts": {
            "source_rows": len(all_rows),
            "eligible_history_interactions": len(partition.train),
            "later_records": len(partition.development),
            "quarantined_records": len(partition.quarantine),
            "h0_eligible_interactions": len(original.train),
            "additional_eligible_interactions": len(partition.train)
            - len(original.train),
            "unique_terms": len(counts),
            "h0_unique_terms": len(old_counts),
            "new_unique_terms": len(counts.keys() - old_counts.keys()),
            "counted_nonempty_interactions": sum(counts.values()),
        },
        "quarantine_reasons": dict(
            Counter(reason for _, reason in partition.quarantine)
        ),
        "history_query_ids": sorted(row["query_id"] for row in partition.train),
        "term_counts": counts,
    }


def prepare_history() -> None:
    paths = [shard_path(HISTORY_ROOT, i, HISTORY_ROWS_PER_SHARD) for i in SHARD_INDICES]
    bundle = build_expanded_catalog(
        paths, original_path=INPUT, original_manifest=MANIFEST
    )
    expected = len(SHARD_INDICES) * HISTORY_ROWS_PER_SHARD
    if bundle["counts"]["source_rows"] != expected:
        raise ValueError(f"Expected {expected} source rows; check the four files")
    created = save_json(HISTORY_CATALOG, bundle)
    print(json.dumps(bundle["counts"], indent=2))
    print(f"{'Saved' if created else 'Reusing'}: {HISTORY_CATALOG}")
    print("Original training/development data and model weights are unchanged.")


def prepare_official(
    raw_path: Path,
    cases_path: Path = OFFICIAL_CASES,
    manifest_path: Path = OFFICIAL_MANIFEST,
) -> Path:
    if file_sha256(raw_path) != EXPECTED_SHA256:
        raise ValueError(f"Raw file checksum mismatch: {raw_path}")
    rows = read_parquet(raw_path, columns=["query_id", "prefix", "final_search_term"])
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} rows, found {len(rows)}")
    cases, seen = [], set()
    for row in rows:
        query_id, prefix, truth = (
            row["query_id"],
            row["prefix"],
            row["final_search_term"],
        )
        if (
            type(query_id) is not int
            or not isinstance(prefix, str)
            or not isinstance(truth, str)
        ):
            raise TypeError(
                "Expected integer query_id and string prefix/final_search_term"
            )
        if query_id in seen:
            raise ValueError(f"Duplicate query_id: {query_id}")
        seen.add(query_id)
        # Official requests already have a prefix: do not sample or cut the answer.
        cases.append(
            asdict(
                DevelopmentCase(
                    query_id,
                    0,
                    prefix,
                    normalize_prefix(prefix),
                    truth,
                    normalize_term(truth),
                )
            )
        )
    # Keep earlier prepared versions unchanged, including their formatting/metadata.
    if cases_path.exists() and manifest_path.exists():
        info = json.loads(manifest_path.read_text())
        if (
            json.loads(cases_path.read_text()) != cases
            or info["raw_sha256"] != EXPECTED_SHA256
            or info["cases_sha256"] != file_sha256(cases_path)
            or info["request_count"] != len(cases)
            or info["normalization_version"] != NORMALIZATION_VERSION
        ):
            raise ValueError(
                "Existing official preparation differs; choose new output paths"
            )
        return cases_path
    manifest = {
        "dataset": "amazon/AmazonQAC",
        "revision": REVISION,
        "source_file": SOURCE_FILE,
        "raw_sha256": EXPECTED_SHA256,
        "request_count": len(cases),
        "cases_sha256": (
            file_sha256(cases_path)
            if cases_path.exists()
            else hashlib.sha256(canonical_json_bytes(cases)).hexdigest()
        ),
        "normalization_version": NORMALIZATION_VERSION,
        "case_order": "source-file order; prefix_index=0 is only an adapter placeholder",
        "prior_exploratory_probe": {
            "date": "2026-10-25",
            "exposed_query_ids": [case["query_id"] for case in cases[:100]],
            "note": "First 100 requests are retained, but are not untouched test evidence",
        },
    }
    for path, value in ((cases_path, cases), (manifest_path, manifest)):
        if path.exists() and json.loads(path.read_text()) != value:
            raise FileExistsError(f"Different prepared data already exists: {path}")
    save_json(cases_path, cases)
    save_json(manifest_path, manifest)
    return cases_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=("development", "history", "official"), required=True
    )
    args = parser.parse_args(argv)
    if args.dataset == "development":
        prepare_development()
    elif args.dataset == "history":
        prepare_history()
    else:
        print(f"Prepared requests: {prepare_official(OFFICIAL_RAW)}")
        print("No model evaluated.")


if __name__ == "__main__":
    main()
