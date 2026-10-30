"""Small local-file helpers shared by prepare, train and evaluate commands."""

import json
from pathlib import Path

from autocomplete.artifacts import verify_file_hash
from autocomplete.config import REVISION
from autocomplete.data import partition_record
from autocomplete.data.normalization import NORMALIZATION_VERSION, normalize_term
from datasets import load_dataset


def read_parquet(path: Path, columns: list[str] | None = None) -> list[dict]:
    """Read local Parquet through Hugging Face Datasets.

    Args:
        path: Local Parquet file.
        columns: Optional fields to load.
    Returns:
        Record dictionaries; the loader does not split by time.
    """
    dataset = load_dataset(
        "parquet", data_files=str(path), split="train", columns=columns
    )

    return dataset.to_list()


def read_jsonl(path: Path) -> list[dict]:
    """Read one JSON record per line.

    Args:
        path: Local JSONL file.
    Returns:
        Record dictionaries in file order.
    """
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
    """Read local JSONL and check its download receipt if present.

    Args:
        path: Local JSONL file.
    Returns:
        Source record dictionaries.
    """
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
    """Split complete September interactions before building windows.

    Args:
        raw_path: Local Parquet file.
        manifest_path: Optional saved input fingerprint.
    Returns:
        Training, development and quarantined records.
    """
    if manifest_path is not None:
        manifest = json.loads(manifest_path.read_text())

        if manifest["normalization_version"] != NORMALIZATION_VERSION:
            raise ValueError("Different normalization version")

        verify_file_hash(raw_path, manifest["hashes"]["raw_file SHA-256"])

    return partition_record(read_parquet(raw_path))


def normalized_terms(records: list[dict]) -> list[str]:
    """Extract normalized completed queries from interactions.

    Args:
        records: Source record dictionaries.
    Returns:
        One normalized term per interaction.
    """
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
