import torch
from torch import nn
from torch.utils.data import DataLoader

from autocomplete.model.model import NeuralCharModel
from autocomplete.model.vocabulary import CharVocabulary
from autocomplete.model.windows import NextCharacterDataset


def main() -> None:
    torch.manual_seed(42)

    terms = ["cat"] * 16
    vocabulary = CharVocabulary(terms)
    dataset = NextCharacterDataset(terms, vocabulary)

    data_loader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=True,
    )

    contexts, targets = next(iter(data_loader))

    model = NeuralCharModel(vocab_size=len(vocabulary))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        initial_loss = criterion(model(contexts), targets).item()

    for step in range(1, 501):
        model.train()
        optimizer.zero_grad()
        logits = model(contexts)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(contexts)
            final_loss = criterion(logits, targets).item()

            accuracy = (logits.argmax(dim=1) == targets).float().mean().item()

            if step == 1 or step % 20 == 0:
                print(f"Step {step}: loss={final_loss:.4f}, accuracy={accuracy:.1%}")

            if final_loss < 0.01 and accuracy == 1.0:
                break

    assert final_loss < initial_loss
    assert accuracy == 1.0

    print(f"Loss: {initial_loss:.4f} -> {final_loss:.4f}")

    examples = NextCharacterDataset(["cat"], vocabulary)
    with torch.no_grad():
        for context, target in examples:
            predicted = model(context.unsqueeze(0)).argmax(dim=1).item()

            print(
                "Expected:",
                vocabulary.id_to_token[target.item()],
                "| Predicted:",
                vocabulary.id_to_token[predicted],
            )


if __name__ == "__main__":
    main()
