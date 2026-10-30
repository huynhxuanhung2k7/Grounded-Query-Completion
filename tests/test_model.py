import pytest
import torch
from torch import nn

from autocomplete.model.model import NeuralCharModel


def test_default_model_preserves_baseline_weights_and_output_shape():
    torch.manual_seed(42)
    baseline = nn.Sequential(
        nn.Embedding(6, 4),
        nn.Flatten(start_dim=1),
        nn.Linear(12, 8),
        nn.ReLU(),
        nn.Linear(8, 6),
    )

    torch.manual_seed(42)
    model = NeuralCharModel(6, context_length=3, embedding_dim=4, hidden_dim=8)
    contexts = torch.tensor([[0, 3, 4], [3, 4, 5]])

    assert model(contexts).shape == (2, 6)
    assert list(model.network.state_dict()) == list(baseline.state_dict())
    torch.testing.assert_close(model(contexts), baseline(contexts), rtol=0, atol=0)


def test_configurable_model_has_finite_gradients_and_single_request_inference():
    model = NeuralCharModel(
        6,
        context_length=3,
        embedding_dim=4,
        hidden_dim=8,
        activation="silu",
        num_hidden_layers=2,
        normalization="batch_norm",
        dropout=0.2,
    )
    contexts = torch.tensor([[0, 3, 4], [3, 4, 5]])

    nn.CrossEntropyLoss()(model(contexts), torch.tensor([4, 2])).backward()

    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())

    model.eval()
    with torch.no_grad():
        first = model(contexts[:1])
        second = model(contexts[:1])

    assert first.shape == (1, 6)
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_unknown_activation_is_rejected():
    with pytest.raises(ValueError, match="activation"):
        NeuralCharModel(6, activation="unknown")


def test_nonpositive_dimensions_are_rejected():
    with pytest.raises(ValueError, match="positive integer"):
        NeuralCharModel(6, hidden_dim=0)


def test_invalid_dropout_is_rejected():
    with pytest.raises(ValueError, match="dropout"):
        NeuralCharModel(6, dropout=1.0)
