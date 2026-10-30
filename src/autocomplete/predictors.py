"""Prepare a prediction function without putting model setup in the runner."""

import json
from pathlib import Path

from autocomplete.artifacts import file_sha256
from autocomplete.baseline.baselines import suggest_popular
from autocomplete.baseline.catalog import (
    build_catalog,
    count_history_terms,
)
from autocomplete.baseline.ngram import NgramModel
from autocomplete.config import (
    BEAM_WIDTH,
    CANDIDATE_BUDGET,
    CHECKPOINT,
    HISTORY_CATALOG,
    INPUT,
    MANIFEST,
)
from autocomplete.contract import prepare_prefix
from autocomplete.data import CUTOFF
from autocomplete.data.loading import load_partition, normalized_terms
from autocomplete.data.normalization import NORMALIZATION_VERSION
from autocomplete.decoding import (
    NeuralPredictor,
    beam_candidates,
    greedy_complete,
)
from autocomplete.evaluation import beam_prediction, completion_prediction
from autocomplete.model.checkpoint import load_checkpoint


def load_predictor(
    method: str, raw_path: Path, manifest_path: Path, checkpoint_path: Path
):
    """Set up a predictor after the command has verified its development inputs."""
    if method in ("neural", "catalog_neural"):
        model, vocabulary, checkpoint = load_checkpoint(checkpoint_path)

        if checkpoint["manifest_sha256"] != file_sha256(manifest_path):
            raise ValueError("Checkpoint references a different manifest")

        details = {
            "checkpoint": {
                "path": checkpoint_path.as_posix(),
                "sha256": file_sha256(checkpoint_path),
                "epoch": checkpoint["epoch"],
                "model_config": checkpoint["model_config"],
            }
        }
        predictor = NeuralPredictor(model, vocabulary)
        if method == "catalog_neural":
            return predictor.score_catalog_terms, details
        return predictor.next_log_probs, details

    if method != "ngram":
        raise ValueError(f"Unknown predictor: {method}")

    partition = load_partition(raw_path)
    model = NgramModel()
    model.fit(normalized_terms(partition.train))
    return model.next_log_probs, {
        "model": {
            "type": "character_ngram",
            "order": model.order,
            "alpha": model.alpha,
            "training_interactions": len(partition.train),
        }
    }


def load_catalog(raw_path: Path | None = None, manifest_path: Path | None = None):
    """Load processed 100K history by default"""
    if raw_path is None:
        bundle = json.loads(HISTORY_CATALOG.read_text(encoding="utf-8"))

        if bundle["normalization_version"] != NORMALIZATION_VERSION:
            raise ValueError("Catalog normalization does not match")

        if bundle["cutoff"] != CUTOFF.isoformat():
            raise ValueError("Catalog cutoff does not match")

        counts = bundle["term_counts"]

        return set(counts), counts

    records = load_partition(raw_path, manifest_path).train
    return build_catalog(records), count_history_terms(records)


def make_suggester(
    method: str,
    *,
    decoder: str = "beam",
    checkpoint_path: Path = CHECKPOINT,
    raw_path: Path = INPUT,
    manifest_path: Path = MANIFEST,
    beam_width: int = BEAM_WIDTH,
    candidate_budget: int = CANDIDATE_BUDGET,
    grounded: bool = False,
):
    """Load once, then return prefix -> predictions for both evaluation and the CLI."""
    if method != "popularity":
        predict, _ = load_predictor(
            method,
            raw_path,
            manifest_path,
            checkpoint_path,
        )
    catalog = counts = None
    if method in ("popularity", "catalog_neural") or grounded:
        catalog, counts = load_catalog()

    def predict_prefix(prefix):
        if method == "popularity":
            _, reason = prepare_prefix(prefix)
            suggestions = [] if reason else suggest_popular(catalog, counts, prefix)

            return {
                "suggestions": suggestions,
                "scores": [counts[term] for term in suggestions],
                "stop_reason": reason or (None if suggestions else "no_candidates"),
            }

        if method == "catalog_neural":
            normalized_prefix, reason = prepare_prefix(prefix)
            ranked = [] if reason else predict(normalized_prefix, catalog)
            top_ten = ranked[:10]
            return {
                "suggestions": [row["completion"] for row in top_ten],
                "scores": [row["log_score"] for row in top_ten],
                "stop_reason": reason or (None if top_ten else "no_candidates"),
            }

        if decoder == "beam":
            pool = beam_candidates(prefix, predict, beam_width, candidate_budget)

            return beam_prediction(pool, catalog=catalog if grounded else None)

        return completion_prediction(greedy_complete(prefix, predict))

    return predict_prefix


if __name__ == "__main__":
    print("Reading existing historical catalog:", HISTORY_CATALOG)
    suggest = make_suggester("popularity")
    for prefix in ("cat ", ""):
        prediction = suggest(prefix)
        print(f"Prefix {prefix!r}: {len(prediction['suggestions'])} suggestions")
        ranked = list(zip(
            prediction["suggestions"], prediction["scores"], strict=True
        ))
        print("First three (term, count):", ranked[:3])
        print("Stop reason:", prediction["stop_reason"])
