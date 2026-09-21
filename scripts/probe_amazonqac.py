from itertools import islice
from pathlib import Path

from datasets import Dataset, load_dataset

REVISION = "975cbfac4623ffb054c59631e06e39d0831e47fb"
SHARD_INDICES = (0, 6, 13, 19)
ROW_PER_SHARD = 5_000

OUTPUT = Path("datasets/amazonqac/raw/train_sample_20k.parquet")
TEMP_OUTPUT = Path("datasets/amazonqac/raw/train_sample_20k.tmp.parquet")

SHARD_SUFFIX = "1d4931ef-16a0-4dc1-8034-ddaa3f7ddec6-c000.snappy.parquet"


def make_shard_url(shard_idx: int) -> str:
    shard_path = f"train/part-{shard_idx:05d}-{SHARD_SUFFIX}"

    return (
        "https://huggingface.co/datasets/amazon/AmazonQAC/"
        f"resolve/{REVISION}/{shard_path}"
    )


def stream_one_shard(shard_url: str):
    return load_dataset(
        "parquet",
        data_files=shard_url,
        split="train",
        streaming=True,
    )


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing sample: {OUTPUT}")

    all_rows = []

    # download 4 shards
    for shard_idx in SHARD_INDICES:
        shard = stream_one_shard(shard_url=make_shard_url(shard_idx))
        rows = list(islice(shard, ROW_PER_SHARD))

        if len(rows) != ROW_PER_SHARD:
            raise RuntimeError(
                f"Shard {shard_idx} returned {len(rows)}, expected {ROW_PER_SHARD}"
            )

        print(f"Shard {shard_idx}: {len(rows)}")
        all_rows.extend(rows)

    print(f"Total: {len(all_rows)}")

    # save them
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    dataset = Dataset.from_list(all_rows)
    dataset.to_parquet(OUTPUT)
    print(f"Saved {len(dataset)} rows to {OUTPUT}")


if __name__ == "__main__":
    main()
