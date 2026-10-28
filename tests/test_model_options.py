"""Model-option regressions retained while the tuning runner is rebuilt."""

import math

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from autocomplete.model.model import NeuralCharModel
from autocomplete.model.training import train_one_epoch


def test_default_architecture_keeps_exact_legacy_keys_and_logits():
    torch.manual_seed(42)
    legacy = nn.Sequential(
        nn.Embedding(6, 4),
        nn.Flatten(start_dim=1),
        nn.Linear(12, 8),
        nn.ReLU(),
        nn.Linear(8, 6),
    )
    torch.manual_seed(42)
    current = NeuralCharModel(6, context_length=3, embedding_dim=4, hidden_dim=8)
    assert list(current.network.state_dict()) == list(legacy.state_dict())
    x = torch.tensor([[0, 3, 4], [3, 4, 5]])
    torch.testing.assert_close(current(x), legacy(x), rtol=0, atol=0)


@pytest.mark.parametrize("normalization", ["none", "layer_norm", "batch_norm"])
@pytest.mark.parametrize("activation", ["relu", "gelu", "tanh"])
@pytest.mark.parametrize("depth", [1, 2])
def test_variants_train_and_allow_single_request_eval(normalization, activation, depth):
    model = NeuralCharModel(
        6,
        context_length=3,
        embedding_dim=4,
        hidden_dim=8,
        num_hidden_layers=depth,
        normalization=normalization,
        activation=activation,
        dropout=0.2,
    )
    logits = model(torch.tensor([[0, 3, 4], [3, 4, 5]]))
    nn.CrossEntropyLoss()(logits, torch.tensor([4, 2])).backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )
    model.eval()
    x = torch.tensor([[0, 3, 4]])
    with torch.no_grad():
        first = model(x)
        second = model(x)
    assert first.shape == (1, 6)
    torch.testing.assert_close(first, second, rtol=0, atol=0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_hidden_layers": 0},
        {"normalization": "bad"},
        {"dropout": -0.1},
        {"dropout": 1.0},
        {"dropout": math.nan},
        {"hidden_dim": True},
    ],
)
def test_invalid_model_options_fail(kwargs):
    with pytest.raises(ValueError):
        NeuralCharModel(6, **kwargs)


def tiny_loader():
    return DataLoader(
        TensorDataset(torch.tensor([[0, 3, 4], [3, 4, 5]]), torch.tensor([4, 2])),
        batch_size=2,
    )


def test_gradient_clipping_is_applied(monkeypatch):
    model = NeuralCharModel(6, context_length=3, embedding_dim=4, hidden_dim=8)
    calls = []
    original = nn.utils.clip_grad_norm_

    def clip(parameters, maximum, **kwargs):
        calls.append((maximum, kwargs))
        return original(parameters, maximum, **kwargs)

    monkeypatch.setattr(nn.utils, "clip_grad_norm_", clip)
    report = train_one_epoch(
        model,
        tiny_loader(),
        torch.optim.Adam(model.parameters()),
        nn.CrossEntropyLoss(),
        gradient_clip_norm=1.0,
    )
    assert math.isfinite(report["loss"])
    assert calls == [(1.0, {"error_if_nonfinite": True})]
