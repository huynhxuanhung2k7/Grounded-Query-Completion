import torch
from torch import nn
from torch.utils.data import DataLoader

from autocomplete.model.vocabulary import PAD_ID, UNK_ID


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader) -> dict[str, int | float | None]:
    """Measure character loss without changing weights; skip UNK targets.

    Args:
        model: Character model.
        loader: Validation context and target batches.
    Returns:
        Mean loss and known/unknown target counts.
    """
    model.eval()

    device = next(model.parameters()).device
    criterion = nn.CrossEntropyLoss(reduction="sum")

    total_loss = 0.0
    total_targets = 0
    scored_targets = 0
    unknown_targets = 0

    for contexts, targets in loader:
        contexts = contexts.to(device)
        targets = targets.to(device)

        if (targets == PAD_ID).any().item():
            raise ValueError("PAD must not be a prediction target")

        total_targets += targets.numel()

        known_mask = targets != UNK_ID
        unknown_targets += (~known_mask).sum().item()
        batch_scored = known_mask.sum().item()

        if batch_scored == 0:
            continue

        logits = model(contexts[known_mask])
        loss = criterion(logits, targets[known_mask])

        total_loss += loss.item()
        scored_targets += batch_scored

    return {
        "loss": total_loss / scored_targets if scored_targets else None,
        "total_targets": total_targets,
        "scored_targets": scored_targets,
        "unknown_targets": unknown_targets,
        "unknown_target_fraction": unknown_targets / total_targets
        if total_targets
        else None,
    }


if __name__ == "__main__":
    from autocomplete.model.model import NeuralCharModel
    from autocomplete.model.vocabulary import CharVocabulary
    from autocomplete.model.windows import NextCharacterDataset

    torch.manual_seed(42)
    vocabulary = CharVocabulary(["cat"])
    model = NeuralCharModel(
        len(vocabulary), context_length=2, embedding_dim=4, hidden_dim=8
    )
    dataset = NextCharacterDataset(
        ["cap"], vocabulary, context_length=2, allow_unknown=True
    )
    before = [parameter.detach().clone() for parameter in model.parameters()]
    print("Training vocabulary: 'cat'; validation term: 'cap'")
    report = evaluate(model, DataLoader(dataset, batch_size=4))
    print("Validation report:", report)
    unchanged = all(
        torch.equal(saved, current)
        for saved, current in zip(before, model.parameters(), strict=True)
    )
    print("Weights unchanged:", unchanged)
