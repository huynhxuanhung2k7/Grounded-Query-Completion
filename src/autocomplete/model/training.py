from time import perf_counter

from torch import nn, optim
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: optim.Optimizer,
    loss_function: _Loss,
) -> dict[str, int | float]:
    model.train()
    device = next(model.parameters()).device

    started = perf_counter()
    running_loss = 0.0
    target_count = 0

    for batch_idx, (contexts, targets) in enumerate(data_loader):
        contexts, targets = contexts.to(device), targets.to(device)

        optimizer.zero_grad()
        logits = model(contexts)
        loss = loss_function(logits, targets)

        loss.backward()
        optimizer.step()

        batch_targets = targets.numel()
        running_loss += loss.item() * batch_targets
        target_count += batch_targets

    if target_count == 0:
        raise ValueError("Training loader contains no targets")

    return {
        "loss": running_loss / target_count,
        "target_count": target_count,
        "elapsed_seconds": perf_counter() - started,
    }


def main() -> None:
    import math

    import torch

    from autocomplete.model.model import NeuralCharModel
    from autocomplete.model.vocabulary import CharVocabulary, EOS_ID, UNK_ID, PAD_ID
    from autocomplete.model.windows import NextCharacterDataset
    from autocomplete.normalization import normalize_prefix

    torch.manual_seed(42)

    terms = [
        "cat food",
        "cat treats",
        "dog food",
        "dog toys",
        "wireless mouse",
        "wireless keyboard",
        "usb cable",
        "usb charger",
        "iphone 15 case",
        "laptop stand",
        "running shoes",
        "water bottle",
    ] * 4

    vocab = CharVocabulary(terms)
    dataset = NextCharacterDataset(terms, vocab)

    data_loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = NeuralCharModel(vocab_size=len(vocab))
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    loss_function = torch.nn.CrossEntropyLoss()

    before = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }

    report = train_one_epoch(model, data_loader, optimizer, loss_function)
    for _ in range(10):
        report = train_one_epoch(model, data_loader, optimizer, loss_function)

    print("Epoch report:", report)


if __name__ == "__main__":
    main()
