# Query Autocomplete

A FIT1061 prototype comparing popularity, character n-grams, catalog-neural ranking,
free neural generation and catalog-filtered neural generation on AmazonQAC.

## Setup

Use Python 3.11 and run commands from this repository directory:

```bash
uv sync
```

Place the supplied `datasets/amazonqac/` and `artifacts/` folders here. Data and trained
models are excluded from Git. For data paths and preparation, see
[datasets/README.md](datasets/README.md).

Neural methods default to `artifacts/neural_v1/best.pt`. The selected tuned checkpoint
is `artifacts/neural_tuning/trial_031/best.pt`.

## Evaluate

```bash
uv run python -m scripts.evaluate --model popularity
uv run python -m scripts.evaluate --model ngram
uv run python -m scripts.evaluate --model catalog_neural
uv run python -m scripts.evaluate --model neural
uv run python -m scripts.evaluate --model neural --grounded
```

Defaults: September development cases, baseline checkpoint, beam width 20 and candidate
budget 50. Useful options:

| Option | Purpose |
|---|---|
| `--split official` | Evaluate October requests |
| `--checkpoint PATH` | Select another neural checkpoint |
| `--output-root PATH` | Save a separate evaluation run |
| `--decoder greedy` | Use greedy generation instead of beam search |

For example, evaluate the tuned neural model separately:

```bash
uv run python -m scripts.evaluate --model neural --checkpoint artifacts/neural_tuning/trial_031/best.pt --output-root artifacts/results_tuned
```

Results are saved under `artifacts/results/<split>/<method-and-settings>/summary.json`
unless another output root is given. Each summary reports Success@10, MRR@10, response
coverage and prediction p95 in milliseconds. Existing identical runs retain their original
results and timing; changed results require another output root.

October is an already-used benchmark. Choose settings using development data.
Grounding checks historical query membership, not product availability or factual correctness.

## Try suggestions and run tests

```bash
uv run python -m scripts.suggest --model neural --grounded --prefix "cat "
uv run python -m pytest -q
```

Omit `--prefix` for interactive suggestions. Tests use small fixtures and need no real
dataset or saved model. Source modules also contain runnable examples, such as
`uv run python src/autocomplete/model/windows.py`.

For a new training experiment, use `uv run python -m scripts.train_model` or
`uv run python -m scripts.tune_neural`. Set a separate run directory before starting
another experiment; an existing tuning study adds more trials when rerun.
