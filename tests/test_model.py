import pytest
import torch

from autocomplete.model.model import NeuralCharModel


@pytest.mark.parametrize("activation", ["relu", "tanh", "gelu"])
@pytest.mark.parametrize("batch_size", [1, 4])
def test_model_shape_and_finite_logits(
    activation: str,
    batch_size: int,
) -> None:
    model = NeuralCharModel(
        vocab_size=20,
        activation=activation,
    )

    contexts = torch.randint(
        0, 20, (batch_size, 32),
        dtype=torch.long,
    )
    logits = model(contexts)

    assert logits.shape == (batch_size, 20)
    assert torch.isfinite(logits).all()


def test_model_rejects_unknown_activation() -> None:
    with pytest.raises(ValueError, match="Unknown activation"):
        NeuralCharModel(vocab_size=20, activation="unknown")