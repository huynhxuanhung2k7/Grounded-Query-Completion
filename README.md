# Grounded Query Autocomplete

A FIT1061 learning project: compare historical popularity, character n-grams, static-catalog
neural ranking, free neural generation and catalog-filtered neural generation.
Grounding checks historical membership, not factual correctness or product availability.

## Five entry points

| Script | Purpose |
|---|---|
| `process_data.py` | Prepare development cases, historical counts or official cases from local files |
| `train_model.py` | Train the baseline and save its best character-validation checkpoint |
| `tune_neural.py` | Harry's unfinished Optuna learning exercise; not a ready-to-run tuner yet |
| `evaluate.py` | Evaluate one method on development or official cases |
| `suggest.py` | Try a method once or through an interactive terminal prompt |

Run commands from this directory, using Python 3.11:

```bash
uv sync
uv run python -m pytest -q
```

## Data: download manually, then process locally

Source: [AmazonQAC](https://huggingface.co/datasets/amazon/AmazonQAC).
[Local data paths and selection notes](datasets/README.md) describe where files belong.
No download script is required or included.

```bash
uv run python -m scripts.process_data --dataset development
uv run python -m scripts.process_data --dataset history
uv run python -m scripts.process_data --dataset official
```

Run only the processing choice you need. Existing prepared data do not need rebuilding.
The original 20K September source contains 16,072 training interactions, 3,918 development
cases and 10 quarantines. The current 100K-source historical catalog has 80,136 eligible
interactions and 69,707 terms. It expands lookup history, not model training.
Development and October answers must never be added to the catalog.

## Evaluate

```bash
uv run python -m scripts.evaluate --model popularity
uv run python -m scripts.evaluate --model ngram
uv run python -m scripts.evaluate --model catalog_neural
uv run python -m scripts.evaluate --model neural
uv run python -m scripts.evaluate --model neural --grounded
```

Default split is development. Generative methods default to beam search (width 20, pool 50);
`--decoder greedy` is available for the earlier single-completion comparison.
Neural methods load `artifacts/neural_v1/best.pt`, without training. N-gram counts still come
only from the September training partition, even when evaluating October.

Once models/settings are frozen and official cases are prepared, the same command supports:

```bash
uv run python -m scripts.evaluate --model neural --split official
```

This documents a capability, not a completed official evaluation. The first 100 October
requests were already exposed in an exploratory probe: do not call the whole set untouched
or repeatedly tune using its results.

New files go to `artifacts/results/<split>/<method-and-settings>/summary.json`.
Each compact summary contains dataset/split, relevant file paths and model/decoder settings,
request count, Success@10, MRR@10, response coverage and `p95_ms`.
P95 is the nearest-rank 95th percentile of **individual prediction times in milliseconds**,
including generation/ranking/filtering but excluding model loading, file loading and metrics.
It is a local run measurement, not a production latency guarantee.

New summaries do not save full candidate pools or every request. Historical prediction
artifacts remain available in the recovery archive for report examples. Identical reruns keep the original
summary and timing; conflicting results require a different `--output-root`, never overwrite
saved evidence. Other options: `--checkpoint PATH`, `--beam-width N`, `--candidate-budget N`.

## Try it in the terminal

```bash
uv run python -m scripts.suggest --model popularity --prefix "cat "
uv run python -m scripts.suggest --model neural --grounded
```

Without `--prefix`, enter prefixes repeatedly; an empty line quits. The model loads once.
The CLI and evaluator use the same prediction function, so the demonstration does not
silently substitute a different algorithm.

## Fresh baseline run (requested 2026-10-28)

The old active artifacts have been moved, including checkpoints and results, to
`../.context-backups/catalog-reset-20261028.QSD0Jp/artifacts/`. Raw/processed data remain.
The active output folder is empty; training must run before neural evaluation.

```bash
uv run python -m scripts.process_data --dataset development
uv run python -m scripts.process_data --dataset history
uv run python -m scripts.train_model
uv run python -m scripts.evaluate --model ngram --decoder greedy
uv run python -m scripts.evaluate --model neural --decoder greedy
uv run python -m scripts.evaluate --model popularity
uv run python -m scripts.evaluate --model ngram
uv run python -m scripts.evaluate --model neural
uv run python -m scripts.evaluate --model neural --grounded
uv run python -m scripts.evaluate --model catalog_neural
```

Processing reuses identical existing outputs; it does not redownload data. Training writes
`best.pt`, `history.json` and `summary.json` under `artifacts/neural_v1/`. The best file is
updated only within this new training run; rerunning training refuses an existing directory.
The current training defaults are at most 10 epochs, early-stopping patience 2, Adam,
learning rate 0.001, batch size 256 and the original 32/64/256 ReLU network.

Catalog-neural scoring uses the same network as generation: it exhaustively ranks existing
matching phrases rather than generating them. Batches of 256 character/EOS windows avoid
one forward call per character; a single scorer is used. Very short prefixes can still be
expensive because they match many phrases. Batching changes computation, not predictions;
do not call it a better model without measured quality.

No real processing/training/evaluation was started by the reset. The official raw file and
prepared files are now local; the temporary setup helper was removed on 2026-10-29.
The permanent official processing command can reuse identical existing outputs.
Reserve official evaluation for the final frozen settings, after any development tuning.

## Code and remaining work

- `src/autocomplete/data/`: normalization, local loading, chronological split and cases.
- `src/autocomplete/baseline/`: historical counts, catalog lookup and character n-grams.
- `src/autocomplete/model/`: vocabulary, windows, neural model, training and saved weights.
- `predictors.py`: model setup and the shared suggestion function.
- `decoding.py`, `evaluation.py`, `artifacts.py`: decoding, metrics and small result files.

See [PROGRESS.md](../PROGRESS.md) for verified status and
[PROJECT_CONTEXT.md](../PROJECT_CONTEXT.md) for the maintained design.
Synthetic command tests do not complete the real five-system development comparison.
Remaining emphasis is model understanding, development experiments and explaining the
results—not restoring download infrastructure.

The shard-prefix sample is bounded rather than uniform. Exact-match scores measure one
recorded answer, not all useful suggestions; catalog absence and incompatible rewrites
limit exact grounding. Development-selected tuning gains are not proven test superiority.
