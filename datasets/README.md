# Dataset location

The maintained acquisition, schema, normalization, split, and popularity contracts are
in [PROJECT_CONTEXT.md](../../PROJECT_CONTEXT.md#3-data-contract-and-acquisition).
Use this existing `project/datasets/` directory consistently.

AmazonQAC is selected. One streamed record was inspected on 2026-09-08. A bounded 20K-row
raw sample across four fixed recorded shards was saved and verified on 2026-09-15. Its
current schema and simple descriptive statistics are explored in
[amazonqac_eda.ipynb](../amazonqac_eda.ipynb).
Keep raw/processed data out of version control; commit acquisition instructions, manifests,
hashes, and small synthetic fixtures.

The previously downloaded `MIMICS-Click.tsv` was removed from the current branch on
2026-09-05 and remains recoverable from earlier Git history. It is not an input to the
selected AmazonQAC training, catalog, or evaluation pipeline.
