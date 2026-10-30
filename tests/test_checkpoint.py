import pytest
import torch

from autocomplete.data.normalization import NORMALIZATION_VERSION
from autocomplete.model.checkpoint import load_checkpoint
from autocomplete.model.model import NeuralCharModel
from autocomplete.model.vocabulary import EOS, PAD, UNK


@pytest.fixture
def checkpoint_source(tmp_path):
    torch.manual_seed(42)
    config = {"vocab_size": 6, "context_length": 2, "embedding_dim": 4, "hidden_dim": 8}
    model = NeuralCharModel(**config).eval()
    checkpoint = {
        "format_version": 1,
        "normalization_version": NORMALIZATION_VERSION,
        "model_config": config,
        "model_state": model.state_dict(),
        "id_to_token": [PAD, UNK, EOS, "t", "a", "c"],
    }

    return tmp_path / "model.pt", model, checkpoint


def test_restore_preserves_saved_ids_and_logits(checkpoint_source):
    path, original, checkpoint = checkpoint_source
    torch.save(checkpoint, path)

    restored, vocabulary, _ = load_checkpoint(path)
    contexts = torch.tensor([[0, 5], [5, 4]])

    assert vocabulary.encode("cat") == [5, 4, 3]
    assert not restored.training
    assert next(restored.parameters()).device.type == "cpu"

    with torch.no_grad():
        torch.testing.assert_close(
            original(contexts), restored(contexts), rtol=0, atol=0
        )


def test_different_normalization_is_rejected(checkpoint_source):
    path, _, checkpoint = checkpoint_source
    checkpoint["normalization_version"] = "different"
    torch.save(checkpoint, path)

    with pytest.raises(ValueError, match="normalization"):
        load_checkpoint(path)


def test_unsupported_format_is_rejected(checkpoint_source):
    path, _, checkpoint = checkpoint_source
    checkpoint["format_version"] = 999
    torch.save(checkpoint, path)

    with pytest.raises(ValueError, match="format_version"):
        load_checkpoint(path)


def test_vocabulary_size_must_match_model(checkpoint_source):
    path, _, checkpoint = checkpoint_source
    checkpoint["id_to_token"] = [PAD, UNK, EOS, "a"]
    torch.save(checkpoint, path)

    with pytest.raises(ValueError, match="vocabulary size"):
        load_checkpoint(path)


def test_nonfinite_weights_are_rejected(checkpoint_source):
    path, _, checkpoint = checkpoint_source
    next(iter(checkpoint["model_state"].values())).view(-1)[0] = float("nan")
    torch.save(checkpoint, path)

    with pytest.raises(ValueError, match="nonfinite"):
        load_checkpoint(path)
