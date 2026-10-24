"""Checkpoint restoration checks using only tiny, locally created fixtures."""

import hashlib
from pathlib import Path

import pytest
import torch

from autocomplete.model.model import NeuralCharModel
from autocomplete.model.vocabulary import EOS, PAD, UNK
from autocomplete.normalization import NORMALIZATION_VERSION
from scripts.check_checkpoint import from_tokens, load_checkpoint, verify_checkpoint


def test_from_tokens_preserves_saved_order_and_copies_input() -> None:
    tokens = [PAD, UNK, EOS, "z", "a", " "]
    vocabulary = from_tokens(tokens)
    assert vocabulary.encode("za ?") == [3, 4, 5, 1]
    assert vocabulary.decode([3, 4, 5]) == "za "
    tokens.append("q")
    assert len(vocabulary) == 6


@pytest.mark.parametrize(
    ("tokens", "error"),
    [
        ((PAD, UNK, EOS, "a"), TypeError),
        ([PAD, UNK, EOS, 1], TypeError),
        ([UNK, PAD, EOS, "a"], ValueError),
        ([PAD, UNK, EOS, "a", "a"], ValueError),
        ([PAD, UNK, EOS], ValueError),
        ([PAD, UNK, EOS, "cat"], ValueError),
    ],
)
def test_from_tokens_rejects_invalid_saved_mapping(tokens, error) -> None:
    with pytest.raises(error):
        from_tokens(tokens)


@pytest.fixture
def checkpoint_source():
    torch.manual_seed(42)
    config = {
        "vocab_size": 6,
        "context_length": 3,
        "embedding_dim": 4,
        "hidden_dim": 8,
        "activation": "relu",
    }
    model = NeuralCharModel(**config).eval()
    checkpoint = {
        "format_version": 1,
        "model_config": config,
        "model_state": model.state_dict(),
        "id_to_token": [PAD, UNK, EOS, "t", "a", "c"],
        "normalization_version": NORMALIZATION_VERSION,
        "manifest_sha256": hashlib.sha256(b"original manifest").hexdigest(),
    }
    return model, checkpoint


def test_checkpoint_restores_fresh_model_with_matching_logits(
    tmp_path: Path, checkpoint_source
) -> None:
    original, checkpoint = checkpoint_source
    path = tmp_path / "checkpoint.pt"
    torch.save(checkpoint, path)
    restored, vocabulary, _ = load_checkpoint(path)

    assert restored is not original
    assert not restored.training
    assert next(restored.parameters()).device.type == "cpu"
    assert vocabulary.encode("cat") == [5, 4, 3]
    contexts = torch.tensor([[0, 0, 5], [5, 4, 3]], dtype=torch.long)
    with torch.no_grad():
        torch.testing.assert_close(original(contexts), restored(contexts))


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("format_version", 99, "format_version"),
        ("normalization_version", "different", "normalization"),
        ("id_to_token", [PAD, UNK, EOS, "a"], "vocabulary size"),
    ],
)
def test_checkpoint_rejects_incompatible_metadata(
    tmp_path: Path, checkpoint_source, field, replacement, message
) -> None:
    _, checkpoint = checkpoint_source
    checkpoint[field] = replacement
    path = tmp_path / "checkpoint.pt"
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match=message):
        load_checkpoint(path)


def test_checkpoint_rejects_nonfinite_weights(
    tmp_path: Path, checkpoint_source
) -> None:
    _, checkpoint = checkpoint_source
    next(iter(checkpoint["model_state"].values())).view(-1)[0] = float("nan")
    path = tmp_path / "checkpoint.pt"
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="nonfinite"):
        load_checkpoint(path)


def test_verification_rejects_changed_manifest_before_loading_data(
    tmp_path: Path, checkpoint_source
) -> None:
    _, checkpoint = checkpoint_source
    path = tmp_path / "checkpoint.pt"
    torch.save(checkpoint, path)
    manifest = tmp_path / "manifest.json"
    manifest.write_bytes(b"changed manifest")
    with pytest.raises(ValueError, match="manifest differs"):
        verify_checkpoint(
            path, manifest_path=manifest, raw_data_path=tmp_path / "missing.parquet"
        )
