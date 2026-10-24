import math
import hashlib
import json
from pathlib import Path
from dataclasses import asdict
import platform
from time import perf_counter

import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from autocomplete.model.model import NeuralCharModel
from autocomplete.model.vocabulary import CharVocabulary
from autocomplete.model.windows import NextCharacterDataset
from autocomplete.model.training import train_one_epoch
from autocomplete.normalization import NORMALIZATION_VERSION, normalize_term
from autocomplete.data import DevelopmentCase, partition_record
from autocomplete.model.validation import evaluate

from autocomplete.evaluation import score_case, summarize_scores
from autocomplete.contract import MAX_ADDED_LENGTH, MAX_INPUT_LENGTH

from datasets import Dataset
from scripts.prepare_data import MANIFEST, OUTPUT
from scripts.probe_amazonqac import RAW_DATA
from scripts.evaluate_popularity import verify_file_hash


def main() -> None:
    torch.manual_seed(42)

    manifest = json.loads(MANIFEST.read_bytes())
    verify_file_hash(RAW_DATA, manifest["hashes"]["raw_file SHA-256"])
    verify_file_hash(OUTPUT, manifest["hashes"]["development_cases SHA-256"])

    raw_data = Dataset.from_parquet(str(RAW_DATA))
    partition = partition_record(raw_data.to_list())

    train_terms = [
        normalize_term(record["final_search_term"]) for record in partition.train
    ]

    validation_terms = [
        normalize_term(record["final_search_term"]) for record in partition.development
    ]

    vocabulary = CharVocabulary(train_terms)
    train_dataset = NextCharacterDataset(train_terms, vocabulary)
    validation_dataset = NextCharacterDataset(
        validation_terms, vocabulary, allow_unknown=True
    )

    MODEL_CONFIG = {
        "vocab_size": len(vocabulary),
        "context_length": 32,
        "embedding_dim": 64,
        "hidden_dim": 256,
        "activation": "relu",
    }

    model = NeuralCharModel(**MODEL_CONFIG)

    training_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=256)

    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    best_validation_loss = float("inf")
    epochs_without_improvement = 0
    best_epoch = None
    best_checkpoint_path = None

    run_directory = Path("artifacts/neural_v1")
    run_directory.mkdir(parents=True, exist_ok=False)

    print("Run directory:", run_directory, flush=True)
    print("Training interactions:", len(train_terms))
    print("Development interactions:", len(validation_terms))
    print("Vocabulary size:", len(vocabulary))
    print("Training targets:", len(train_dataset))
    print("Validation targets:", len(validation_dataset), flush=True)

    for epoch in range(1, 11):
        print(f"\nEpoch {epoch}/10", flush=True)
        with tqdm(
            training_loader,
            desc="Training",
            unit="batch",
            leave=False,
        ) as training_progress:
            training_report = train_one_epoch(
                model, training_progress, optimizer, loss_function
            )

        validation_started = perf_counter()
        with tqdm(
            validation_loader,
            desc="Validation",
            unit="batch",
            leave=False,
        ) as validation_progress:
            validation_report = evaluate(model, validation_progress)
        validation_seconds = perf_counter() - validation_started

        current_loss = validation_report["loss"]

        if current_loss is None:
            raise ValueError("Validation contains no scored targets")

        if not math.isfinite(current_loss):
            raise ValueError(
                f"Invalid validation loss at epoch {epoch}: {current_loss}"
            )

        print(
            f"Epoch {epoch}/10 | "
            f"train_loss={training_report['loss']:.4f} | "
            f"val_loss={current_loss:.4f} | "
            f"train_seconds={training_report['elapsed_seconds']:.2f} | "
            f"validation_seconds={validation_seconds:.2f}",
            flush=True,
        )
        print(
            f"Validation unknown targets: "
            f"{validation_report['unknown_targets']}/"
            f"{validation_report['total_targets']}"
        )

        if current_loss < best_validation_loss:
            best_validation_loss = current_loss
            best_epoch = epoch
            epochs_without_improvement = 0

            checkpoint_path = run_directory / "checkpoints" / f"epoch_{epoch:02d}.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

            checkpoint = {
                "format_version": 1,
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "id_to_token": list(vocabulary.id_to_token),
                "model_config": dict(MODEL_CONFIG),
                "training_report": dict(training_report),
                "validation_report": dict(validation_report),
                "seed": 42,
                "normalization_version": NORMALIZATION_VERSION,
                "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
                "dependency_versions": {
                    "python": platform.python_version(),
                    "torch": str(torch.__version__),
                },
            }

            with checkpoint_path.open("xb") as output:
                torch.save(checkpoint, output)

            best_checkpoint_path = checkpoint_path
            print(f"Saved best checkpoint: {best_checkpoint_path}", flush=True)

        else:
            epochs_without_improvement += 1
            print(
                f"Consecutive epochs without improvement: {epochs_without_improvement}",
                flush=True,
            )

        if epochs_without_improvement >= 2:
            print(
                "Early stopping: validation loss did not improve "
                "for two consecutive epochs.",
                flush=True,
            )
            break

    print("\nTraining finished.")
    print("Best epoch:", best_epoch)
    print(f"Best validation loss: {best_validation_loss:.4f}")
    print("Best checkpoint:", best_checkpoint_path, flush=True)


if __name__ == "__main__":
    main()
