"""Load and validate the downloaded source for an expanded catalog."""

import hashlib
import json
from collections import Counter
from pathlib import Path

from autocomplete.data import PartitionResult, partition_record
from scripts.probe_amazonqac import (
    REVISION,
    SHARD_INDICES,
    make_shard_url,
)

SOURCE_ROOT = (
    Path("datasets/amazonqac/raw/history_profile") / REVISION
)


def load_history_sources(paths: list[Path]) -> PartitionResult:
    """Verify downloaded JSONL files and partition their combined records.

    Requires completed metadata with the expected revision, source URL,
    row count and SHA-256. Duplicate interaction IDs are rejected by
    partition_record().
    """
    records: list[dict[str, object]] = []

    for path in paths:
        metadata_path = path.with_suffix(".metadata.json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        if metadata.get("dataset_revision") != REVISION:
            raise ValueError(f"Unexpected dataset revision: {metadata_path}")

        if metadata.get("complete") is not True:
            raise ValueError(f"Source cache is incomplete: {path}")

        if metadata.get("cache_format") != "jsonl-v1":
            raise ValueError(f"Unsupported cache format: {metadata_path}")

        shard_index = metadata.get("shard_index")
        if type(shard_index) is not int or shard_index not in SHARD_INDICES:
            raise ValueError(f"Unexpected shard index: {metadata_path}")

        if metadata.get("source_url") != make_shard_url(shard_index):
            raise ValueError(f"Unexpected source URL: {metadata_path}")

        expected_rows = metadata.get("cached_rows")
        if type(expected_rows) is not int or expected_rows < 0:
            raise ValueError(f"Invalid cached row count: {metadata_path}")

        with path.open("rb") as handle:
            actual_hash = hashlib.file_digest(handle, "sha256").hexdigest()

        if actual_hash != metadata.get("sha256"):
            raise ValueError(f"Source SHA-256 mismatch: {path}")

        file_records: list[dict[str, object]] = []

        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSON at {path}:{line_number}"
                    ) from error

                if not isinstance(record, dict):
                    raise ValueError(
                        f"Expected a record object at {path}:{line_number}"
                    )

                file_records.append(record)

        if len(file_records) != expected_rows:
            raise ValueError(
                f"Row count mismatch for {path}: "
                f"expected {expected_rows}, found {len(file_records)}"
            )

        records.extend(file_records)
        print(f"Verified shard {shard_index}: {len(file_records)} rows")

    # Existing validation checks schema, duplicate IDs and chronology.
    return partition_record(records)


def main() -> None:
    paths = [
        SOURCE_ROOT / f"shard_{index:05d}_first_10000.jsonl"
        for index in SHARD_INDICES
    ]

    partition = load_history_sources(paths)

    total_rows = (
        len(partition.train)
        + len(partition.development)
        + len(partition.quarantine)
    )
    quarantine_reasons = Counter(
        reason for _, reason in partition.quarantine
    )

    print(f"Source rows:       {total_rows}")
    # Successful partitioning guarantees unique interaction IDs.
    print(f"Unique source IDs: {total_rows}")
    print(f"Eligible history:  {len(partition.train)}")
    print(f"Later records:     {len(partition.development)}")
    print(f"Quarantined:       {len(partition.quarantine)}")
    print(f"Quarantine reasons: {dict(quarantine_reasons)}")


if __name__ == "__main__":
    main()