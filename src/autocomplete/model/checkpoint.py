"""Reload saved neural weights and vocabulary without importing command scripts."""

from pathlib import Path

import torch

from autocomplete.data.normalization import NORMALIZATION_VERSION
from autocomplete.model.model import NeuralCharModel
from autocomplete.model.vocabulary import CharVocabulary


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

    vocabulary = CharVocabulary.from_tokens(checkpoint["id_to_token"])
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
