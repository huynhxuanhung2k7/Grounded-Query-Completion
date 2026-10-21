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

The 2026-10-20 design separates the model-training sample T from historical sources H_j.
T builds neural/n-gram examples and the character vocabulary. H_j builds distinct normalized
catalog phrases K_j and occurrence counts f_j and may contain many more earlier interactions.
Start with H_0=T (16,072 interactions); larger history targets of 100,000 and 1,000,000
eligible interactions are future cost-dependent acquisitions, not files already present.
Do not expand the model vocabulary or retrain weights during the catalog-size study.

Both sources use valid complete interactions before 25 September 2023 at 00:00 while
the existing 3,918 later-September development cases remain fixed. Ten reversed-time source
records are quarantined. The separate official test covers 1–14 October 2023 and stays
untouched until configurations are frozen. Neither held-out answers nor supplied
full-September popularity aggregates construct the historical catalogs/counts.

Keep model-training, historical-catalog/count and development artifacts separately identified
by provenance and digest. Existing preparation scripts implement the small-sample path;
the larger projected-column history reader/catalog-size experiment is planned work.

The previously downloaded `MIMICS-Click.tsv` was removed from the current branch on
2026-09-05 and remains recoverable from earlier Git history. It is not an input to the
selected AmazonQAC training, catalog, or evaluation pipeline.
