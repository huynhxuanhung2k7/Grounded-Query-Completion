import math

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from autocomplete.model.model import NeuralCharModel
from autocomplete.model.training import train_one_epoch
from autocomplete.model.vocabulary import CharVocabulary
from autocomplete.model.windows import NextCharacterDataset


def training_example():
    torch.manual_seed(42)
    vocabulary = CharVocabulary(["cat"])
    dataset = NextCharacterDataset(["cat"], vocabulary, context_length=2)
    model = NeuralCharModel(
        len(vocabulary), context_length=2, embedding_dim=4, hidden_dim=8
    )

    return model, dataset


def test_one_epoch_updates_weights_and_counts_all_targets():
    model, dataset = training_example()
    before = model.network[-1].weight.detach().clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    report = train_one_epoch(
        model, DataLoader(dataset, batch_size=4), optimizer, nn.CrossEntropyLoss()
    )

    assert report["target_count"] == 4
    assert math.isfinite(report["loss"])
    assert not torch.equal(before, model.network[-1].weight)


def test_mean_loss_weights_uneven_batches_by_target_count():
    model, dataset = training_example()
    contexts, targets = next(iter(DataLoader(dataset, batch_size=4)))
    expected = nn.CrossEntropyLoss()(model(contexts), targets).item()
    optimizer = torch.optim.SGD(model.parameters(), lr=0)

    report = train_one_epoch(
        model, DataLoader(dataset, batch_size=3), optimizer, nn.CrossEntropyLoss()
    )

    assert report["loss"] == pytest.approx(expected)


def test_gradient_clipping_limits_the_gradient_norm():
    model, dataset = training_example()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    train_one_epoch(
        model,
        DataLoader(dataset, batch_size=4),
        optimizer,
        nn.CrossEntropyLoss(),
        gradient_clip_norm=0.1,
    )
    gradients = torch.cat(
        [parameter.grad.flatten() for parameter in model.parameters()]
    )

    assert gradients.norm().item() <= 0.10001


def test_empty_training_loader_is_rejected():
    model, _ = training_example()
    empty = TensorDataset(
        torch.empty((0, 2), dtype=torch.long), torch.empty(0, dtype=torch.long)
    )
    optimizer = torch.optim.Adam(model.parameters())

    with pytest.raises(ValueError, match="no targets"):
        train_one_epoch(model, DataLoader(empty), optimizer, nn.CrossEntropyLoss())
