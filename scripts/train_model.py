"""Train the baseline character model."""

import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from autocomplete.artifacts import canonical_json_bytes, file_sha256, save_json
from autocomplete.config import INPUT, MANIFEST
from autocomplete.data.loading import load_partition, normalized_terms
from autocomplete.data.normalization import NORMALIZATION_VERSION
from autocomplete.model.model import NeuralCharModel
from autocomplete.model.training import train_one_epoch
from autocomplete.model.validation import evaluate
from autocomplete.model.vocabulary import CharVocabulary
from autocomplete.model.windows import NextCharacterDataset

SEED = 42
MAX_EPOCHS = 10
EARLY_STOPPING_PATIENCE = 2
BATCH_SIZE = 256
LEARNING_RATE = 0.001
RUN_DIRECTORY = Path("artifacts/neural_v1")
MODEL_CONFIG = {
    "context_length": 32,
    "embedding_dim": 64,
    "hidden_dim": 256,
    "activation": "relu",
}


def main() -> None:
    if RUN_DIRECTORY.exists():
        raise FileExistsError(
            f"Run already exists: {RUN_DIRECTORY}. Choose a new RUN_DIRECTORY."
        )

    torch.manual_seed(SEED)
    partition = load_partition(INPUT, MANIFEST)
    train_terms = normalized_terms(partition.train)
    validation_terms = normalized_terms(partition.development)
    vocabulary = CharVocabulary(train_terms)
    config = {"vocab_size": len(vocabulary), **MODEL_CONFIG}
    train_data = NextCharacterDataset(
        train_terms, vocabulary, context_length=config["context_length"]
    )
    validation_data = NextCharacterDataset(
        validation_terms,
        vocabulary,
        context_length=config["context_length"],
        allow_unknown=True,
    )
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    validation_loader = DataLoader(validation_data, batch_size=BATCH_SIZE)
    model = NeuralCharModel(**config)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss()
    RUN_DIRECTORY.mkdir(parents=True)
    print(
        f"Training {len(train_terms)} interactions; validating {len(validation_terms)}",
        flush=True,
    )

    history, best_loss, stale_epochs = [], float("inf"), 0
    best_path = None

    for epoch in range(1, MAX_EPOCHS + 1):
        print(f"Epoch {epoch}/{MAX_EPOCHS}: training...", flush=True)
        training = train_one_epoch(model, train_loader, optimizer, criterion)
        validation = evaluate(model, validation_loader)
        loss = validation["loss"]

        if loss is None or not math.isfinite(loss):
            raise ValueError(f"Invalid validation loss at epoch {epoch}")

        history.append({"epoch": epoch, "training": training, "validation": validation})

        (RUN_DIRECTORY / "history.json").write_bytes(canonical_json_bytes(history))
        print(
            f"Train loss {training['loss']:.4f}; validation loss {loss:.4f}", flush=True
        )

        if loss < best_loss:
            best_loss, stale_epochs = loss, 0
            best_path = RUN_DIRECTORY / "best.pt"
            torch.save(
                {
                    "format_version": 1,
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "model_config": config,
                    "id_to_token": vocabulary.id_to_token,
                    "training_report": training,
                    "validation_report": validation,
                    "seed": SEED,
                    "normalization_version": NORMALIZATION_VERSION,
                    "manifest_sha256": file_sha256(MANIFEST),
                },
                best_path,
            )
        else:
            stale_epochs += 1

        if stale_epochs >= EARLY_STOPPING_PATIENCE:
            print("Early stopping: validation loss stopped improving.")
            break

    save_json(
        RUN_DIRECTORY / "summary.json",
        {
            "best_checkpoint": str(best_path),
            "best_validation_loss": best_loss,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "seed": SEED,
        },
    )
    print(f"Best checkpoint: {best_path}", flush=True)


if __name__ == "__main__":
    main()
