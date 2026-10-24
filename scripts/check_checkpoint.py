"""Reload a saved neural checkpoint and reproduce its development loss.

Run from project/: uv run python -m scripts.check_checkpoint [checkpoint_path]
This reads local artifacts only; it does not retrain or overwrite results.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from autocomplete.data import partition_record
from autocomplete.model.model import NeuralCharModel
from autocomplete.model.validation import evaluate
from autocomplete.model.vocabulary import EOS, PAD, UNK, CharVocabulary
from autocomplete.model.windows import NextCharacterDataset
from autocomplete.normalization import NORMALIZATION_VERSION, normalize_term
from datasets import Dataset
from scripts.evaluate_popularity import verify_file_hash
from scripts.prepare_data import INPUT, MANIFEST


def from_tokens(tokens: list[str]) -> CharVocabulary:
    """Restore the exact saved ordering, rather than rebuilding sorted IDs."""
    if not isinstance(tokens, list):
        raise TypeError("Saved tokens must be a list")

    if any(not isinstance(token, str) for token in tokens):
        raise TypeError("Every token must be a string")

    if tokens[:3] != [PAD, UNK, EOS]:
        raise ValueError("Expected PAD, UNK, EOS at IDs 0, 1, 2")

    if len(set(tokens)) != len(tokens):
        raise ValueError("Saved tokens must be unique")

    characters = tokens[3:]
    if not characters or any(len(token) != 1 for token in characters):
        raise ValueError("Expected nonempty single-character vocabulary")

    vocabulary = CharVocabulary.__new__(CharVocabulary)
    vocabulary.id_to_token = list(tokens)
    vocabulary.token_to_id = {token: token_id for token_id, token in enumerate(tokens)}

    return vocabulary


def load_checkpoint(
    checkpoint_path: Path,
) -> tuple[NeuralCharModel, CharVocabulary, dict]:
    """Restore a compatible finite predictor on CPU, ready for inference."""
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format_version")
    if checkpoint["normalization_version"] != NORMALIZATION_VERSION:
        raise ValueError("Checkpoint normalization version differs from current code")

    vocabulary = from_tokens(checkpoint["id_to_token"])
    if len(vocabulary) != checkpoint["model_config"]["vocab_size"]:
        raise ValueError("Saved vocabulary size differs from model configuration")

    model = NeuralCharModel(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    if any(
        not torch.isfinite(value).all().item() for value in model.state_dict().values()
    ):
        raise ValueError("Checkpoint model contains nonfinite weights")
    model.eval()

    return model, vocabulary, checkpoint


def verify_checkpoint(
    checkpoint_path: Path,
    *,
    manifest_path: Path = MANIFEST,
    raw_data_path: Path = INPUT,
) -> dict[str, int | float | None]:
    """Re-evaluate the original development terms and compare saved statistics."""
    model, vocabulary, checkpoint = load_checkpoint(checkpoint_path)
    manifest_bytes = manifest_path.read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != checkpoint["manifest_sha256"]:
        raise ValueError("Data manifest differs from checkpoint provenance")
    manifest = json.loads(manifest_bytes)
    verify_file_hash(raw_data_path, manifest["hashes"]["raw_file SHA-256"])

    partition = partition_record(Dataset.from_parquet(str(raw_data_path)).to_list())
    validation_terms = [
        normalize_term(record["final_search_term"]) for record in partition.development
    ]
    validation_dataset = NextCharacterDataset(
        validation_terms,
        vocabulary,
        context_length=checkpoint["model_config"]["context_length"],
        allow_unknown=True,
    )
    validation_loader = DataLoader(validation_dataset, batch_size=256, shuffle=False)
    report = evaluate(model, validation_loader)
    saved = checkpoint["validation_report"]

    if report["loss"] is None or saved["loss"] is None:
        raise ValueError("Reload verification requires scored validation targets")
    if not math.isfinite(report["loss"]) or not math.isfinite(saved["loss"]):
        raise ValueError("Reload verification requires finite validation losses")
    torch.testing.assert_close(
        report["loss"],
        saved["loss"],
        rtol=1e-6,
        atol=1e-7,
    )
    for key in ("total_targets", "scored_targets", "unknown_targets"):
        if report[key] != saved[key]:
            raise ValueError(f"Validation count differs: {key}")
    if report["unknown_target_fraction"] != saved["unknown_target_fraction"]:
        raise ValueError("Validation unknown-target fraction differs")

    print("Loaded epoch:", checkpoint["epoch"])
    print("Saved validation:", checkpoint["validation_report"])
    print("Reloaded validation:", report)
    print("Checkpoint reload verification passed.")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoint",
        nargs="?",
        type=Path,
        default=Path("artifacts/neural_v1/checkpoints/epoch_04.pt"),
    )
    args = parser.parse_args()
    verify_checkpoint(args.checkpoint)


if __name__ == "__main__":
    main()
