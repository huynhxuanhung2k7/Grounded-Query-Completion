"""Prepare fixed development cases from the cached AmazonQAC sample."""

from pathlib import Path

from datasets import Dataset

from autocomplete.data import (
    partition_record,
    make_development_cases,
    write_development_cases
)

INPUT = Path("datasets/amazonqac/raw/train_sample_20k.parquet")
OUTPUT = Path("datasets/amazonqac/processed/development_cases.json")
SEED = 42

def main() -> None:
    """Load the local sample, partition interactions, and save development cases.

    Run from the project directory so INPUT and OUTPUT resolve correctly.
    Use SEED to select one observed prefix per development interaction, then
    write canonical JSON and print partition counts, case count, and SHA-256.
    The writer reuses an identical existing file and rejects different content.
    """
    raw_data = Dataset.from_parquet(str(INPUT))
    partition = partition_record(raw_data.to_list())
    
    development_cases = make_development_cases(partition, SEED)
    digest = write_development_cases(OUTPUT, development_cases)
    
    print(f"{'Source:':<13}{len(raw_data)}")
    print(f"{'Train:':<13}{len(partition.train)}")
    print(f"{'Development:':<13}{len(partition.development)}")
    print(f"{'Quarantine:':<13}{len(partition.quarantine)}")
    print(f"{'Cases:':<13}{len(development_cases)}")    
    print(f"{'Digest:':<13}{digest}")

if __name__ == "__main__":
    main()    
