import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from autocomplete.model.model import NeuralCharModel
from autocomplete.model.validation import evaluate
from autocomplete.model.vocabulary import PAD_ID, UNK_ID, CharVocabulary
from autocomplete.model.windows import NextCharacterDataset


def validation_example():
    torch.manual_seed(42)
    vocabulary = CharVocabulary(["cat"])
    model = NeuralCharModel(
        len(vocabulary), context_length=2, embedding_dim=4, hidden_dim=8
    )
    dataset = NextCharacterDataset(
        ["cap"], vocabulary, context_length=2, allow_unknown=True
    )

    return model, dataset


def test_validation_counts_unknown_targets_without_changing_weights():
    model, dataset = validation_example()
    before = [parameter.detach().clone() for parameter in model.parameters()]

    report = evaluate(model, DataLoader(dataset, batch_size=4))

    assert report["total_targets"] == 4
    assert report["scored_targets"] == 3
    assert report["unknown_targets"] == 1
    assert report["unknown_target_fraction"] == 0.25
    assert not model.training
    assert all(
        torch.equal(a, b) for a, b in zip(before, model.parameters(), strict=True)
    )


def test_validation_loss_averages_only_known_targets_across_batches():
    model, dataset = validation_example()
    contexts, targets = next(iter(DataLoader(dataset, batch_size=4)))
    known = targets != UNK_ID
    expected = nn.CrossEntropyLoss()(model(contexts[known]), targets[known]).item()

    report = evaluate(model, DataLoader(dataset, batch_size=3))

    assert report["loss"] == pytest.approx(expected)


def test_all_unknown_targets_have_no_defined_loss():
    model, _ = validation_example()
    dataset = TensorDataset(torch.tensor([[0, 0]]), torch.tensor([UNK_ID]))

    report = evaluate(model, DataLoader(dataset))

    assert report["loss"] is None
    assert report["scored_targets"] == 0
    assert report["unknown_target_fraction"] == 1


def test_empty_validation_has_no_defined_loss_or_unknown_fraction():
    model, _ = validation_example()
    empty = TensorDataset(
        torch.empty((0, 2), dtype=torch.long), torch.empty(0, dtype=torch.long)
    )

    report = evaluate(model, DataLoader(empty))

    assert report["loss"] is None
    assert report["total_targets"] == 0
    assert report["unknown_target_fraction"] is None


def test_padding_cannot_be_a_validation_target():
    model, _ = validation_example()
    dataset = TensorDataset(torch.tensor([[0, 0]]), torch.tensor([PAD_ID]))

    with pytest.raises(ValueError, match="PAD"):
        evaluate(model, DataLoader(dataset))
