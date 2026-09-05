# Dataset location

The maintained acquisition, schema, normalization, split, and popularity contracts are
in [PROJECT_CONTEXT.md](../../PROJECT_CONTEXT.md#3-data-contract-and-acquisition).
Use this existing `project/datasets/` directory consistently.

AmazonQAC is selected; acquisition is not recorded yet. Start with a ten-record schema
probe, then a bounded cached sample. Keep raw/processed data out of version control;
commit acquisition instructions, manifests, hashes, and small synthetic fixtures.

`MIMICS-Click.tsv` is retained as the previously downloaded alternative. It is not an
input to the selected AmazonQAC training, catalog, or evaluation pipeline.
