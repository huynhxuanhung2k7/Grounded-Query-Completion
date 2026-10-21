# Grounded Query Autocomplete

**FIT1061 — Introduction to Artificial Intelligence | HD Project**

## Abstract

This project investigates grounded query autocomplete for e-commerce search. Given a
partially typed product query, the planned system will generate and rank up to ten completed
searches. Its central experiment compares the same character-level neural model in two modes:
a free mode that may generate previously unseen queries, and a grounded mode that only shows
queries supported by a historical search catalog. Historical popularity, character n-gram
generation and static-catalog neural ranking provide distinct comparison baselines.
Catalog size will be varied independently of the neural-training sample.

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
eligible earlier interactions. That historical source can be larger than the sample used
to train the neural model. It does not mean that a suggestion is factually correct, safe, or
linked to an available product.

## Current status

The project is under active development. Data preparation and the first popularity
development evaluation are implemented:

- a bounded 20,000-record AmazonQAC sample has been acquired from four pinned source shards;
- whole-record partitioning and fixed development-case preparation exist;
- the verified local split is 16,072 training, 3,918 development and 10 quarantined records.
- historical catalog construction, occurrence counts, prefix lookup and popularity ranking work;
- per-request scoring, batch Success@10/MRR@10 and a hash-verified evaluation runner work.

The n-gram, neural model, static neural ranking, decoding, complete serving contract and
interactive autocomplete demo remain planned work. The measured popularity rehearsal below
is development evidence, not a completed five-system comparison or an official-test result.

### Verified H_0 popularity rehearsal

On 21 October 2026, an independent reconstruction checked all 3,918 saved prediction rows,
their recorded scores and the source/manifest hashes. H_0 contains the 16,072 eligible
earlier interactions and its catalog has 15,266 distinct normalized terms.

| Measure | Result |
|---|---|
| Development requests | 3,918 |
| Top-ten hits | 187 |
| Success@10 | 4.772843% |
| MRR@10 | 0.03410014 |
| Catalog-member answers | 303 / 3,918 (7.7335%) |
| Requests receiving suggestions (independent output audit) | 1,928 / 3,918 (49.2088%) |
| Fill rate (independent output audit) | 13,996 / 39,180 (35.7223%) |

The ranking and metric calculations are correct for this bounded sample. The low catalog
membership limits historical lookup even before prefix compatibility and ranking are
considered. This motivates the planned larger eligible-history pilot while keeping model
training and development cases fixed; it does not establish that expansion will improve
results. Popularity ignores the supplied full-source popularity values and never uses
development answers to construct its catalog or counts.

Response coverage and fill rate above were calculated from saved predictions during review;
the runner currently saves only request count, Success@10 and MRR@10 in its summary.
Latency, uncertainty, shared length-limit/error contracts and the other systems still need
implementation or verification before the final comparison.

## Dataset

[AmazonQAC](https://huggingface.co/datasets/amazon/AmazonQAC) is an Amazon query-autocomplete
dataset containing real typed prefixes and submitted product searches. The project uses a
bounded local subset so that the experiment remains reproducible and practical on student
hardware.

The current local sample contains 20,000 September 2023 interactions selected as 5,000 records
from each of four fixed shards. This sample is intended for building, debugging, and evaluating
the first end-to-end development prototype; it is not necessarily the final training scale.
Two data budgets are planned separately. The neural/n-gram training sample stays manageable;
a later model-data experiment may expand its source toward 300,000 interactions after
profiling. A historical-catalog experiment instead keeps the neural model fixed and streams
more eligible earlier interactions to collect distinct phrases and occurrence counts.
Working history targets are the existing 16,072 interactions, then 100,000 and 1,000,000
eligible interactions, subject to acquisition/memory limits. The larger targets have not
been acquired and are not requirements for the first prototype.

The chronological boundary is 25 September 2023 at 00:00. Both model training and historical
catalog sources must precede it while the fixed 25–30 September development cases guide
choices. The released [official test](https://huggingface.co/datasets/amazon/AmazonQAC/tree/main/test)
covers 1–14 October 2023 and is reserved for final evaluation of frozen configurations.
No later-October dataset is assumed.

The current small catalog has 15,266 normalized phrases. Only 303 of the 3,918 development
answers are members (7.73%). This is a data-coverage diagnostic, not model accuracy. Report
it again for every catalog size, along with prefix compatibility and generation/ranking
failures; keep overall and seen/unseen results visible.

Raw data is deliberately excluded from version control. Acquisition settings and source
revision are recorded in
[`scripts/probe_amazonqac.py`](scripts/probe_amazonqac.py), while dataset handling notes are in
[`datasets/README.md`](datasets/README.md).

Splitting complete query records before generating model examples helps prevent prefixes from
the same interaction leaking across partitions.

## Methodology

The planned experiment has five main stages:

1. **Prepare the data.** Validate the AmazonQAC schema, normalize Unicode and whitespace while
   preserving meaningful punctuation, create chronological partitions, and record a
   reproducible data manifest.
2. **Build comparison baselines.** Rank compatible catalog queries by historical-source
   frequency and implement a character n-gram on the model-training sample. Once the neural
   model exists, add static-catalog ranking with its suffix/EOS probabilities.
3. **Train a character model.** Train a small fixed-context neural network to predict the next
   character, including the end of a query.
4. **Generate and rank suggestions.** Decode completions from the same neural candidate pool
   in free and exact-grounded modes. The static neural baseline scores existing compatible
   catalog phrases directly, without beam generation.
5. **Evaluate the systems.** Compare Success@10, MRR@10, response coverage, fill rate,
   out-of-catalog rate, candidate recall, and request latency. Results will also be examined by
   catalog reachability, prefix compatibility, prefix length, and other relevant slices.

The first goal is a small, reproducible end-to-end prototype using the current 20,000-record
sample. The planned studies compare systems/filtering, historical-catalog sizes at fixed
model and generated pools, and candidate budgets at fixed model/catalog. These are future
experiments, not results. Any later model-training expansion is a separate change. Choose
configurations using September development data and freeze them before official evaluation.

## Planned final system

The completed prototype will provide a terminal interface that accepts a partial product query
and returns up to ten ranked suggestions. It will expose five systems through a shared
evaluation contract:

- popularity lookup;
- character n-gram completion;
- static-catalog neural ranking of existing phrases;
- neural free generation; and
- neural generation with exact catalog grounding.

The finished project will also save reloadable model and data artifacts, reproducible
per-request evaluation results, latency measurements, and representative failure cases.
The additional static baseline is our working interpretation of the TA's wording:
popularity already ranks a fixed catalog by frequency, so this baseline instead ranks
that catalog with the same neural model. Its exact meaning should be confirmed at the next
tutor discussion. Reference scoring is exhaustive over eligible matches; any shortlist
must be labelled separately and report its candidate-recall limitation.

The working quality aim is a five-percentage-point absolute Success@10 gain over popularity,
supported by paired uncertainty and subgroup analysis. Also compare neural generation with
n-gram generation. Exact membership is an engineering check; recommendation usefulness
requires quality, availability and latency evidence. These are planning aims, not scores.

H2 may submit the modular project through the coordinator-approved GitHub ZIP/link route.
The notebook/report can explain and reproduce its evidence with two or more meaningful
experiments and referenced visualisations. Template placeholders do not cap project scope.
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

The existing suite covers package import, record validation, normalization, chronological
partitioning, development-case preparation/persistence, catalog/count construction, prefix
matching, popularity ranking and per-request/batch evaluation. Neural model and decoder
behavior remain to be implemented and verified.

### Run the popularity development evaluation

With the cached sample and prepared cases available, run from the project directory:

```bash
uv run python -m scripts.evaluate_popularity
```

The runner verifies the raw Parquet and saved cases against
`datasets/amazonqac/manifest_20k.v2.json`, builds catalog/counts from the earlier partition,
and saves `artifacts/popularity_h0/development_results.json`. Results contain the run ID,
manifest fingerprint, every original case with predictions/scores, and a summary. An
identical second run reuses the file; conflicting content is rejected without overwrite.
Raw/processed data, generated artifacts, caches and machine-specific files stay out of Git.
Manifests and the unexecuted EDA notebook can be committed for reproducibility.

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

- The acquired 20,000-record subset is bounded and is not assumed to represent all Amazon
  search traffic. Larger model/history sizes are plans subject to profiling.
- Exact grounding cannot return a correct query that is absent from the historical catalog, and
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

After the core prototype is complete, planned studies examine catalog size and candidate
budgets. Further possible extensions include neural-training scale, explicitly shortlisted
catalog ranking, typo correction, richer neural architectures, session
context, content-policy checks, and a user-facing interface. These will be treated as separate
experiments so that their costs and effects can be measured rather than assumed.

## Acknowledgements

This project uses the AmazonQAC dataset introduced in
[AmazonQAC: A Large-Scale, Naturalistic Query Autocomplete Dataset](https://aclanthology.org/2024.emnlp-industry.78/).
It is being developed for FIT1061, Introduction to Artificial Intelligence.
