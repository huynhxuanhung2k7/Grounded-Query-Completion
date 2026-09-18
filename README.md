# Grounded Query Autocomplete

**FIT1061 — Introduction to Artificial Intelligence | HD Project**

## Abstract

This project investigates grounded query autocomplete for e-commerce search. Given a
partially typed product query, the planned system will generate and rank up to ten completed
searches. Its central experiment compares the same character-level neural model in two modes:
a free mode that may generate previously unseen queries, and a grounded mode that only shows
queries supported by a historical search catalog. Popularity and character n-gram systems
will provide simpler comparison baselines.

The project uses the [AmazonQAC dataset](https://huggingface.co/datasets/amazon/AmazonQAC),
which contains real query prefixes, final search terms, timestamps, sessions, and popularity
metadata. The work is being developed as an HD project for FIT1061, Introduction to
Artificial Intelligence, while keeping the codebase suitable for continued personal research
and extension after the unit.

## Project purpose

Autocomplete systems must balance useful ranking, response availability, novelty, and speed.
Restricting suggestions to historically observed searches guarantees catalog support, but it
may also remove useful new completions or return too few suggestions. This project asks:

> How does exact historical grounding change the ranking quality, availability, novelty, and
> response time of character-based query autocomplete?

Here, *grounded* means that a displayed suggestion is present in a catalog built only from
the training partition. It does not mean that a suggestion is factually correct, safe, or
linked to an available product.

## Current status

The project is under active development. The current implementation focuses on repository
setup and data preparation:

- a Python 3.11 package and test environment are configured with `uv`;
- a bounded 20,000-record AmazonQAC sample has been acquired from four pinned source shards;
- the local sample's schema, identifiers, date coverage, and content hash have been checked;
- a notebook provides initial exploratory data analysis; and
- schema validation and text normalization are in progress.

The baseline recommenders, neural model, decoding, shared evaluation, and command-line
autocomplete demo described below are planned work. The repository does not yet provide a
complete suggestion system or experimental results.

## Dataset

[AmazonQAC](https://huggingface.co/datasets/amazon/AmazonQAC) is an Amazon query-autocomplete
dataset containing real typed prefixes and submitted product searches. The project uses a
bounded local subset so that the experiment remains reproducible and practical on student
hardware.

The current local sample contains 20,000 September 2023 training interactions selected as
5,000 records from each of four fixed shards. Raw data is deliberately excluded from version
control. Acquisition settings and source revision are recorded in
[`scripts/probe_amazonqac.py`](scripts/probe_amazonqac.py), while dataset handling notes are in
[`datasets/README.md`](datasets/README.md).

The project will use chronological partitions:

- earlier September interactions for training, catalog construction, and popularity counts;
- later September interactions for development and improvement decisions; and
- the separate October test data only for the final frozen evaluation.

Splitting complete query records before generating model examples helps prevent prefixes from
the same interaction leaking across partitions.

## Methodology

The planned experiment has five main stages:

1. **Prepare the data.** Validate the AmazonQAC schema, normalize Unicode and whitespace while
   preserving meaningful punctuation, create chronological partitions, and record a
   reproducible data manifest.
2. **Build simple baselines.** Rank compatible historical queries by training-partition
   popularity and implement a character n-gram model.
3. **Train a character model.** Train a small fixed-context neural network to predict the next
   character, including the end of a query.
4. **Generate suggestions.** Decode completions from the same neural candidate pool in free
   and exact-grounded modes so that the effect of grounding can be isolated.
5. **Evaluate the systems.** Compare Success@10, MRR@10, response coverage, fill rate,
   out-of-catalog rate, candidate recall, and request latency. Results will also be examined by
   catalog reachability, prefix compatibility, prefix length, and other relevant slices.

The first goal is a small, reproducible end-to-end prototype. Any later improvement will be
motivated by observed development failures and tested as a controlled change before the final
evaluation is run.

## Planned final system

The completed prototype will provide a terminal interface that accepts a partial product query
and returns up to ten ranked suggestions. It will expose four systems through a shared
evaluation contract:

- popularity lookup;
- character n-gram completion;
- neural free generation; and
- neural generation with exact catalog grounding.

The finished project will also save reloadable model and data artifacts, reproducible
per-request evaluation results, latency measurements, and representative failure cases.
Typo correction, semantic retrieval, session-aware recommendations, and a browser interface
are possible future extensions rather than requirements of the initial prototype.

## Setup

### Requirements

- Python 3.11
- [`uv`](https://docs.astral.sh/uv/)
- Git

Clone the repository, enter the implementation directory, and install the locked dependencies:

```bash
git clone https://github.com/huynhxuanhung2k7/Grounded-Query-Completion.git
cd Grounded-Query-Completion/project
uv sync
```

### Run the tests

```bash
uv run pytest
```

The suite currently covers package import, training-record validation, and normalization.
Because normalization is active work, the current branch may contain failing tests that define
the next behavior to implement.

### Run the current entry point

```bash
uv run python main.py
```

This is currently a setup smoke check. It will become the command-line autocomplete demo as the
prototype is completed.

### Acquire the bounded dataset sample

Data acquisition is optional for basic package tests and requires network access. From the
`project` directory, run:

```bash
uv run python scripts/probe_amazonqac.py
```

The script downloads the pinned 20,000-record sample to
`datasets/amazonqac/raw/train_sample_20k.parquet`. It refuses to overwrite an existing sample.
The raw-data directory is ignored by Git and should remain local.

## Repository structure

```text
project/
├── datasets/                 # Local data location and handling notes
├── scripts/                  # Dataset acquisition utilities
├── src/autocomplete/         # Python package source
├── tests/                    # Automated behavioral tests
├── amazonqac_eda.ipynb       # Exploratory analysis of the local sample
├── main.py                   # Current smoke entry point; planned CLI demo
├── pyproject.toml            # Package and dependency configuration
└── uv.lock                   # Locked dependency versions
```

## Limitations

- The 20,000-record subset is bounded and is not assumed to represent all Amazon search
  traffic.
- Exact grounding cannot return a correct query that is absent from the training catalog, and
  it may reduce the number of available suggestions.
- The initial character model only extends a prefix; it is not designed to correct spelling or
  recover arbitrary query rewrites.
- A fixed context window loses information from the beginning of sufficiently long prefixes.
- Text normalization can merge distinct surface forms, so collisions must be measured and
  retained raw text must remain available for audits.
- The recorded final search is one observed user choice, not a complete list of all useful
  suggestions. Exact-match metrics therefore do not measure overall human satisfaction.
- Offline evaluation on sampled historical data cannot by itself justify deployment.
- The project is a learning and research prototype, not a production search service.

## Future development

After the core prototype is complete, possible extensions include larger data experiments,
retrieval-first grounded ranking, typo correction, richer neural architectures, session
context, content-policy checks, and a user-facing interface. These will be treated as separate
experiments so that their costs and effects can be measured rather than assumed.

## Acknowledgements

This project uses the AmazonQAC dataset introduced in
[AmazonQAC: A Large-Scale, Naturalistic Query Autocomplete Dataset](https://aclanthology.org/2024.emnlp-industry.78/).
It is being developed for FIT1061, Introduction to Artificial Intelligence.
