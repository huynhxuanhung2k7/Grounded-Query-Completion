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


if __name__ == "__main__":
    from tempfile import TemporaryDirectory

    torch.manual_seed(42)
    vocabulary = CharVocabulary(["cat"])
    config = {
        "vocab_size": len(vocabulary), "context_length": 2,
        "embedding_dim": 4, "hidden_dim": 8,
    }
    original = NeuralCharModel(**config).eval()
    with TemporaryDirectory() as directory:
        path = Path(directory) / "toy.pt"
        torch.save({
            "format_version": 1, "normalization_version": NORMALIZATION_VERSION,
            "model_config": config, "model_state": original.state_dict(),
            "id_to_token": vocabulary.id_to_token,
        }, path)
        restored, restored_vocabulary, _ = load_checkpoint(path)
        print("Restored token order:", restored_vocabulary.id_to_token)
        print("Inference mode:", not restored.training)
        context = torch.tensor([vocabulary.encode("ca")], dtype=torch.long)
        with torch.no_grad():
            same_logits = torch.equal(original(context), restored(context))
            print("Same logits after reload:", same_logits)
    print("Toy checkpoint removed; experiment checkpoints were not used")
