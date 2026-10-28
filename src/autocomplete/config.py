"""Shared local paths and reference decoder settings; run from project/."""

from pathlib import Path

INPUT = Path("datasets/amazonqac/raw/train_sample_20k.parquet")
OUTPUT = Path("datasets/amazonqac/processed/development_cases.json")
MANIFEST = Path("datasets/amazonqac/manifest_20k.v2.json")
CHECKPOINT = Path("artifacts/neural_v1/best.pt")

REVISION = "975cbfac4623ffb054c59631e06e39d0831e47fb"
SHARD_INDICES = (0, 6, 13, 19)
SAMPLE_ROWS_PER_SHARD = 5_000  # Four files: 20,000 source rows total.
HISTORY_ROWS_PER_SHARD = 25_000  # Includes the sample: 100,000 total, not extra.
SAMPLE_ROOT = Path("datasets/amazonqac/raw/sample_20k") / REVISION
HISTORY_ROOT = Path("datasets/amazonqac/raw/history_100k") / REVISION
HISTORY_CATALOG = Path("datasets/amazonqac/processed/history_source_100k/catalog.json")
OFFICIAL_RAW = Path("datasets/amazonqac/raw/official") / REVISION / "october.parquet"
OFFICIAL_CASES = Path("datasets/amazonqac/processed/official_cases.json")
OFFICIAL_MANIFEST = Path("datasets/amazonqac/processed/official_manifest.json")

BEAM_WIDTH = 20
CANDIDATE_BUDGET = 50
MAX_SUGGESTIONS = 10
