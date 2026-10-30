"""Shared local paths and reference decoder settings; run from project/."""

from pathlib import Path

INPUT = Path("datasets/amazonqac/raw/train_sample_20k.parquet")
OUTPUT = Path("datasets/amazonqac/processed/development_cases.json")
MANIFEST = Path("datasets/amazonqac/manifest_20k.v2.json")
CHECKPOINT = Path("artifacts/neural_v1/best.pt")

REVISION = "975cbfac4623ffb054c59631e06e39d0831e47fb"
SHARD_INDICES = (0, 6, 13, 19)
SAMPLE_ROWS_PER_SHARD = 5_000
HISTORY_ROWS_PER_SHARD = 25_000
SAMPLE_ROOT = Path("datasets/amazonqac/raw/sample_20k") / REVISION
HISTORY_ROOT = Path("datasets/amazonqac/raw/history_100k") / REVISION
HISTORY_CATALOG = Path("datasets/amazonqac/processed/history_source_100k/catalog.json")
OFFICIAL_RAW = Path("datasets/amazonqac/raw/official") / REVISION / "october.parquet"
OFFICIAL_CASES = Path("datasets/amazonqac/processed/official_cases.json")
OFFICIAL_MANIFEST = Path("datasets/amazonqac/processed/official_manifest.json")

BEAM_WIDTH = 20
CANDIDATE_BUDGET = 50
MAX_SUGGESTIONS = 10


if __name__ == "__main__":
    print("Working directory:", Path.cwd())
    print("Training source:", INPUT)
    print("Historical catalog:", HISTORY_CATALOG)
    print("Neural checkpoint:", CHECKPOINT)
    print("Live beam width:", BEAM_WIDTH)
    print("Finished candidate budget:", CANDIDATE_BUDGET)
    print("Maximum displayed suggestions:", MAX_SUGGESTIONS)
