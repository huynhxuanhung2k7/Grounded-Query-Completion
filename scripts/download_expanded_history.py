"""Optionally reproduce the four downloaded 10K-row history source files.

Run from project/: uv run python -m scripts.download_expanded_history
If the submission ZIP already provides verified files, they are reused locally.
This downloads source records and build_expanded_catalog applies history.
"""

import hashlib
import json
from itertools import islice
from pathlib import Path

from scripts.probe_amazonqac import (
    REVISION,
    SHARD_INDICES,
    make_shard_url,
    stream_one_shard,
)

ROWS_PER_SHARD = 10_000
OUTPUT_ROOT = Path("datasets/amazonqac/raw/history_profile") / REVISION

# Fingerprints of the exact JSONL files retained for this project.
EXPECTED_SHA256 = {
    0: "84953d3d3c7bad4aec85623288e495f37031993879abf69485da35374fd8a539",
    6: "e6420d470b0690a6660c04a86b8e4b0a34592729f1eac878f0eaf0501a03d0f4",
    13: "43262955a4a3731ae88518da0d6cd19ff252e3061ae31de4995151b1e825160c",
    19: "3f9d04c58042d024bdc8b9facb2ec4edfed387ca900cabc0a13bcf320a983ffc",
}


def download_shard(shard_index: int, output_root: Path = OUTPUT_ROOT) -> Path:
    """Reuse a verified file or download a fixed source prefix without overwriting.

    Interrupted downloads leave a .part file for inspection. Move that partial
    file aside before retrying; this simple script does not implement resume.
    """
    path = output_root / f"shard_{shard_index:05d}_first_{ROWS_PER_SHARD}.jsonl"
    metadata_path = path.with_suffix(".metadata.json")
    metadata = {
        "cache_format": "jsonl-v1",
        "dataset_revision": REVISION,
        "shard_index": shard_index,
        "source_url": make_shard_url(shard_index),
        "max_scanned_rows": ROWS_PER_SHARD,
        "cached_rows": ROWS_PER_SHARD,
        "complete": True,
        "sha256": EXPECTED_SHA256[shard_index],
        "stop_reason": "row_limit",
    }

    if path.exists() or metadata_path.exists():
        if not path.is_file() or not metadata_path.is_file():
            raise FileExistsError(f"Incomplete existing file pair; preserved: {path}")
        existing = json.loads(metadata_path.read_bytes())
        if any(existing.get(key) != value for key, value in metadata.items()):
            raise ValueError(f"Source metadata mismatch; preserved: {metadata_path}")
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        if digest != metadata["sha256"]:
            raise ValueError(f"SHA-256 mismatch; preserved: {path}")
        print(f"Reusing verified shard {shard_index}: {path}", flush=True)
        return path

    output_root.mkdir(parents=True, exist_ok=True)
    partial_path = path.with_suffix(".jsonl.part")
    print(f"Downloading first {ROWS_PER_SHARD} rows: {metadata['source_url']}", flush=True)
    digest = hashlib.sha256()
    row_count = 0
    with partial_path.open("xb") as handle:
        source = stream_one_shard(metadata["source_url"])
        for record in islice(source, ROWS_PER_SHARD):
            line = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
            handle.write(line)
            digest.update(line)
            row_count += 1

    if row_count != ROWS_PER_SHARD:
        raise ValueError(f"Expected {ROWS_PER_SHARD} rows, received {row_count}: {partial_path}")
    if digest.hexdigest() != metadata["sha256"]:
        raise ValueError(f"Downloaded bytes differ from the retained source: {partial_path}")

    partial_path.rename(path)
    with metadata_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    print(f"Saved and verified {row_count} rows: {path}", flush=True)
    return path


def main() -> None:
    for shard_index in SHARD_INDICES:
        download_shard(shard_index)


if __name__ == "__main__":
    main()
