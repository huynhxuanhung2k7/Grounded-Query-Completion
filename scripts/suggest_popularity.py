import argparse
import json

from autocomplete.baselines import suggest_popular
from autocomplete.catalog import build_catalog, count_history_terms
from autocomplete.contract import MAX_INPUT_LENGTH, prepare_prefix
from autocomplete.data import partition_record
from datasets import Dataset
from scripts.evaluate_popularity import verify_file_hash
from scripts.prepare_data import MANIFEST
from scripts.probe_amazonqac import RAW_DATA


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Suggest searches using historical popularity."
    )
    
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--k", type=int, default=10)
    
    args = parser.parse_args()
    if args.k <= 0:
        parser.error("--k must be greater than zero")   
    
    normalized_prefix, reason = prepare_prefix(args.prefix)

    if reason == "empty_prefix":
        print("Please enter a nonempty prefix.")
        return

    if reason == "input_too_long":
        print(
            f"Input too long: maximum {MAX_INPUT_LENGTH} "
            "characters after normalization."
        )
        return
    
    print(f"prefix={args.prefix!r}, k={args.k}")
    
    manifest = json.loads(MANIFEST.read_bytes())
    verify_file_hash(RAW_DATA, manifest["hashes"]["raw_file SHA-256"])
    
    raw_data = Dataset.from_parquet(str(RAW_DATA))
    partition = partition_record(raw_data.to_list())
    
    catalog = build_catalog(partition.train)
    counts = count_history_terms(partition.train)
    
    suggestions = suggest_popular(catalog, counts, normalized_prefix, args.k)
    if not suggestions:
        print("No matching historical searches.")
        return

    for rank, suggestion in enumerate(suggestions, start=1):
        print(f"{rank}. {suggestion}")

if __name__ == "__main__":
    main()