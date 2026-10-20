"""Prepare fixed development cases from the cached AmazonQAC sample."""

import hashlib
import json
from collections import Counter
from pathlib import Path

from autocomplete.data import (
    CUTOFF,
    make_development_cases,
    partition_record,
    write_development_cases,
)
from autocomplete.normalization import NORMALIZATION_VERSION
from datasets import Dataset
from scripts.probe_amazonqac import REVISION, ROW_PER_SHARD, SHARD_INDICES

INPUT = Path("datasets/amazonqac/raw/train_sample_20k.parquet")
OUTPUT = Path("datasets/amazonqac/processed/development_cases.json")
MANIFEST = Path("datasets/amazonqac/manifest_20k.v2.json")
SEED = 42

def main() -> None:
    """Load the local sample, partition interactions, and save development cases.

    Use SEED to select one observed prefix per development interaction, then
    write canonical JSON and print partition counts, case count, and SHA-256.
    The writer reuses an identical existing file and rejects different content.
    """
    if not INPUT.is_file():
        raise FileNotFoundError(
            f"Cached AmazonQAC sample not found: {INPUT.resolve()}. "
            "Run from the project directory and ensure the sample exists."
        ) 
    
    raw_data = Dataset.from_parquet(str(INPUT))
    partition = partition_record(raw_data.to_list())
    
    development_cases = make_development_cases(partition, SEED)
    development_cases_digest = write_development_cases(OUTPUT, development_cases)
    raw_file_digest = hashlib.sha256(INPUT.read_bytes()).hexdigest()
    
    manifest = {
        "dataset": {
            "name": "amazon/AmazonQAC",
            "revision": REVISION
        },
        "shard": {
            "shard_index": SHARD_INDICES,
            "per_shard capacity": ROW_PER_SHARD,
        },
        "development_prefix_seed": SEED,
        "cutoff": CUTOFF.isoformat(),
        "normalization_version" : NORMALIZATION_VERSION,
        "counts": {
            "source": len(raw_data),
            "train": len(partition.train),
            "development_cases": len(partition.development),
            "quarantine": len(partition.quarantine)
        },
        "quarantine_reasons": dict(
            Counter(reason for _, reason in partition.quarantine)
        ),
        "hashes": {
            "raw_file SHA-256":  raw_file_digest,
            "development_cases SHA-256": development_cases_digest
        },
        "source_schema": raw_data.features.to_dict(),
        "selected_query_ids": list(raw_data["query_id"]),
        "selection_rule": {
            "method": "first_n_rows_per_shard",
            "shard_order": list(SHARD_INDICES),
            "rows_per_shard": ROW_PER_SHARD,
        },
    }
    manifest_bytes = json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8")
    if MANIFEST.exists():
        if MANIFEST.read_bytes() != manifest_bytes:
            raise FileExistsError(
            f"Refusing to overwrite different manifest: {MANIFEST}"
        )
        else:
            print(f"Reusing identical manifest: {MANIFEST}")
    else:
        MANIFEST.write_bytes(manifest_bytes)
    
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"{'Source:':<13}{len(raw_data)}")
    print(f"{'Train:':<13}{len(partition.train)}")
    print(f"{'Development:':<13}{len(partition.development)}")
    print(f"{'Quarantine:':<13}{len(partition.quarantine)}")
    print(f"{'Cases:':<13}{len(development_cases)}")    
    print(f"{'Digest:':<13}{development_cases_digest}")

if __name__ == "__main__":
    main()    
