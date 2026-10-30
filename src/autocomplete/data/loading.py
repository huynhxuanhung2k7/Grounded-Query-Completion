"""Small local-file helpers shared by prepare, train and evaluate commands."""

import json
from pathlib import Path

from autocomplete.artifacts import verify_file_hash
from autocomplete.config import REVISION
from autocomplete.data import partition_record
from autocomplete.data.normalization import NORMALIZATION_VERSION, normalize_term
from datasets import load_dataset


def read_parquet(path: Path, columns: list[str] | None = None) -> list[dict]:
    """Read a local file through the same Datasets API used for acquisition.

    'train' is the file loader's default split name, not our chronology decision.
    """
    dataset = load_dataset(
        "parquet", data_files=str(path), split="train", columns=columns
    )
    return dataset.to_list()


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as source:
        for number, line in enumerate(source, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{number}") from error
            if not isinstance(record, dict):
                raise TypeError(f"Expected a record at {path}:{number}")
            records.append(record)
    return records


def read_source(path: Path) -> list[dict]:
    """Accept manual JSONL files too; check an adjacent receipt when present."""
    receipt = path.with_suffix(".metadata.json")
    if receipt.exists():
        info = json.loads(receipt.read_text())
        if info["dataset_revision"] != REVISION:
            raise ValueError(f"Different dataset revision: {path}")
        verify_file_hash(path, info["sha256"])
    records = read_jsonl(path)
    if receipt.exists() and len(records) != info["cached_rows"]:
        raise ValueError(f"Row count mismatch: {path}")
    return records


def load_partition(raw_path: Path, manifest_path: Path | None = None):
    """Read the September source and split whole interactions, never windows."""
    if manifest_path is not None:
        manifest = json.loads(manifest_path.read_text())
        
        if manifest["normalization_version"] != NORMALIZATION_VERSION:
            raise ValueError("Different normalization version")
        
        verify_file_hash(raw_path, manifest["hashes"]["raw_file SHA-256"])
    
    return partition_record(read_parquet(raw_path))


def normalized_terms(records: list[dict]) -> list[str]:
    return [normalize_term(record["final_search_term"]) for record in records]


if __name__ == "__main__":
    from tempfile import TemporaryDirectory

    from datasets import Dataset

    rows = [{"final_search_term": "  CAT Food "}, {"final_search_term": "cat toy"}]
    with TemporaryDirectory() as directory:
        path = Path(directory) / "queries.parquet"
        Dataset.from_list(rows).to_parquet(str(path))
        loaded = read_parquet(path)
        print("Loaded records:", loaded)
        print("Normalized terms:", normalized_terms(loaded))
        print("Empty record list:", normalized_terms([]))
