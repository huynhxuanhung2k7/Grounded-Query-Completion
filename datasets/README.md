# Local AmazonQAC data

Download manually from [AmazonQAC](https://huggingface.co/datasets/amazon/AmazonQAC).
The project does not require a reproducible downloading program.

Source revision used by the existing files:
`975cbfac4623ffb054c59631e06e39d0831e47fb`.
[September files](https://huggingface.co/datasets/amazon/AmazonQAC/tree/975cbfac4623ffb054c59631e06e39d0831e47fb/train)
and [official October file](https://huggingface.co/datasets/amazon/AmazonQAC/tree/975cbfac4623ffb054c59631e06e39d0831e47fb/test).

All paths below are relative to `project/datasets/amazonqac/`; keep data out of Git.
Paths and row counts are constants in `src/autocomplete/config.py`.

| Choice | Local input | Prepared output |
|---|---|---|
| development | `raw/train_sample_20k.parquet` | `processed/development_cases.json`, `manifest_20k.v2.json` |
| history | `raw/history_100k/REVISION/shard_NNNNN_first_25000.jsonl` | `processed/history_source_100k/catalog.json` |
| official | `raw/official/REVISION/october.parquet` | `processed/official_cases.json`, `processed/official_manifest.json` |

Use the actual revision above in place of REVISION. Shard indices are 0, 6, 13 and 19.
The sample selects the first 5,000 rows of each shard. If the combined sample Parquet
already exists, processing uses it unchanged; otherwise it combines the four local
`raw/sample_20k/REVISION/shard_NNNNN_first_5000.jsonl` files.
The existing JSONL selections are cached local inputs, not the format downloaded by
clicking a full Parquet shard. There is no need to download them again.
JSONL is one record dictionary per line; files without old download receipts are accepted.

Official filename on Hugging Face:
`part-00000-ba5fc3b2-05ee-452d-bf3d-84a2fd6a463a-c000.snappy.parquet`.
Save it locally as `october.parquet` at the configured path; the processor checks its known
checksum. It preserves the supplied prefix, including trailing spaces and rewrites.

Run the appropriate choice from `project/`:

```bash
uv run python -m scripts.process_data --dataset development
uv run python -m scripts.process_data --dataset history
uv run python -m scripts.process_data --dataset official
```

No processing choice downloads, trains or evaluates. Existing identical outputs are reused;
different output is refused. Existing datasets, checkpoint identities and earlier results
are not replaced by this command cleanup.

The original 20K source splits into 16,072 training interactions, 3,918 development cases
and 10 quarantines. Split whole interactions before expanding character windows/prefixes.
All catalog counts use complete interactions before September 25, 2023.
Later and quarantined rows do not add catalog terms; supplied popularity is not used as
our observed historical occurrence count.

The 100K source includes the original sample once: it is not 100K extra rows or 100K
eligible interactions. Its processed catalog has 80,136 eligible interactions / 69,707
terms and is the current lookup default. It does not enlarge neural/n-gram training.
The older 40K source/catalog remains unchanged (32,005 eligible interactions / 29,516 terms).

October is reserved for frozen final evaluation. A prior exploratory probe exposed the
first 100 requests; preparation retains those IDs rather than treating them as untouched.
For the full split rules see [PROJECT_CONTEXT.md](../../PROJECT_CONTEXT.md#3-data-contract-and-acquisition).
