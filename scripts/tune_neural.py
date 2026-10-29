import math
from pathlib import Path
from math import ceil
from time import perf_counter


import json
import optuna

from autocomplete.artifacts import load_development_inputs
from autocomplete.config import OUTPUT
from autocomplete.data.loading import load_partition, normalized_terms
from autocomplete.model.vocabulary import CharVocabulary

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from optuna.trial import Trial

from autocomplete.model.model import NeuralCharModel
from autocomplete.model.windows import NextCharacterDataset
from autocomplete.model.training import train_one_epoch
from autocomplete.model.validation import evaluate

from autocomplete.artifacts import save_json
from autocomplete.config import (
    INPUT,
    MANIFEST,
    BEAM_WIDTH,
    CANDIDATE_BUDGET,
)

from autocomplete.evaluation import evaluate_cases
from autocomplete.predictors import make_suggester

from autocomplete.artifacts import file_sha256
from autocomplete.data.normalization import NORMALIZATION_VERSION

N_TRIALS = 40
MAX_EPOCHS = 10
EARLY_STOPPING = 2
SEED = 42

RUN_DIRECTORY = Path("artifacts/neural_tuning")

def suggest_config(trial: Trial) -> dict[str, object]:

    model = {
        "hidden_dim": trial.suggest_categorical("hidden_dim", [256, 512]),
        "context_length": trial.suggest_categorical("context_length", [32, 64]),  
        "num_hidden_layers": trial.suggest_categorical("num_hidden_layers", [1, 2]),
        "embedding_dim": trial.suggest_categorical("embedding_dim", [64, 128]),
        "activation": trial.suggest_categorical("activation", ["relu", "gelu", "silu", "tanh"]),
        "normalization": trial.suggest_categorical("normalization", ["none", "layer_norm", "batch_norm"]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),    
    }
    
    optimizer_name = trial.suggest_categorical("optimizer", ["adam", "adamw"])
    if optimizer_name == "adamw":
        weight_decay = trial.suggest_categorical("weight_decay", [0.0, 0.001, 0.0001])
    else:
        weight_decay = 0.0
        
    training = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
        "optimizer":  optimizer_name,
        "weight_decay": weight_decay,
        "scheduler": trial.suggest_categorical("scheduler", ["none", "plateau", "cosine"]),
        "gradient_clip_norm": trial.suggest_categorical("gradient_clip_norm", [None, 1.0, 5.0]),
    }
        
    return {
        "model": model,
        "training": training
    }
    

def train_trial(
    config,
    train_terms,
    validation_terms,
    vocabulary,
    trial_directory,
) -> Path:
    model_config, training_config = config["model"], config["training"]
    
    context_length = model_config["context_length"]
    batch_size = training_config["batch_size"]
    
    torch.manual_seed(SEED)
    
    full_model_config = {
        "vocab_size": len(vocabulary),
        **model_config,
    }
    
    model = NeuralCharModel(**full_model_config)
    
    train_data = NextCharacterDataset(
        train_terms,
        vocabulary,
        context_length,
    )
    
    validation_data = NextCharacterDataset(
        validation_terms,
        vocabulary,
        context_length,
        allow_unknown=True
    )
    
    train_loader = DataLoader(
        train_data,
        batch_size,
        shuffle=True,
    )
    
    val_loader = DataLoader(
        validation_data,
        batch_size,
    )
    
    optimizer_type = {
        "adam": optim.Adam,
        "adamw": optim.AdamW,
    }[training_config["optimizer"]]
    
    optimizer = optimizer_type(
        model.parameters(),
        lr=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
    )
    
    loss_function = nn.CrossEntropyLoss()
    
    scheduler = None
    if training_config["scheduler"] == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=0
        )
    elif training_config["scheduler"] == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=MAX_EPOCHS,
            eta_min=0.0
        )
        
    best_path = trial_directory / "best.pt"
    history_path = trial_directory / "history.pt"
    
    history = []
    best_loss = float("inf")
    epochs_without_improvement = 0
    
    for epoch in range(1, MAX_EPOCHS + 1):
        training_report = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_function,
            gradient_clip_norm=training_config["gradient_clip_norm"],
        )
        validation_report = evaluate(model, val_loader)
        loss = validation_report["loss"]
        
        if loss is None or not math.isfinite(loss):
            raise ValueError("Validation loss is missing or non-finite")

        history.append({
            "epoch": epoch,
            "training": training_report,
            "validation": validation_report,
        })
        
        if loss < best_loss:
            best_loss = loss
            epochs_without_improvement = 0
            
            torch.save(
                {
                    "format_version": 1,
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "model_config": full_model_config,
                    "id_to_token": vocabulary.id_to_token,
                    "training_report": training_report,
                    "validation_report": validation_report,
                    "seed": SEED,
                    "normalization_version": NORMALIZATION_VERSION,
                    "manifest_sha256": file_sha256(MANIFEST),
                },
                best_path,
            )
        else:
            epochs_without_improvement += 1
        
        if training_config["scheduler"] == "plateau":
            scheduler.step(loss)
        elif training_config["scheduler"] == "cosine":
            scheduler.step()
        
        torch.save(history, history_path)
        
        if epochs_without_improvement >= EARLY_STOPPING:
            break
    
    return best_path
            
def evaluate_trial(checkpoint_path, development_cases) -> dict:
    suggest = make_suggester(
        "neural",
        decoder="beam",
        checkpoint_path=checkpoint_path,
        raw_path=INPUT,
        manifest_path=MANIFEST,
        beam_width=BEAM_WIDTH,
        candidate_budget=CANDIDATE_BUDGET,
        grounded=False,
    )
    
    latencies_ms = []
    
    def time_predict(prefix): 
        started = perf_counter()
        
        prediction = suggest(prefix)
        
        elapsed_ms = (perf_counter() - started) * 1000
        latencies_ms.append(elapsed_ms)
        
        return prediction

    result = evaluate_cases(
        development_cases,
        time_predict,
    )
    
    if latencies_ms:
        sorted_times = sorted(latencies_ms)
        rank = math.ceil(0.95 * len(sorted_times))
        p95_ms = sorted_times[rank - 1]
    else:
        p95_ms = None

    result["p95_ms"] = p95_ms
    
    return result

def objective(
    trial,
    train_terms,
    validation_terms,
    vocabulary,
    development_cases,
    run_directory,
) -> float:
    config = suggest_config(trial)
    
    trial_directory = run_directory / f"trial_{trial.number:03d}"
    trial_directory.mkdir(parents=True, exist_ok=True)
    
    save_json(trial_directory / "config.json", config) 
    
    checkpoint_path = train_trial(
        config,
        train_terms,
        validation_terms,
        vocabulary,
        trial_directory,
    )
    
    result = evaluate_trial(
        checkpoint_path,
        development_cases,
    )
    
    summary = {
        "trial_number": trial.number,
        "split": "development",
        "checkpoint_path": str(checkpoint_path),
        "decoder": "beam",
        "grounded": False,
        "beam_width": BEAM_WIDTH,
        "candidate_budget": CANDIDATE_BUDGET,
        **result["summary"],
        **result["prediction_summary"],
        "p95_ms": result["p95_ms"],
    }
    
    save_json(trial_directory / "summary.json", summary)
    
    return summary["success_at_10"]

def main() -> None:
    RUN_DIRECTORY.mkdir(parents=True, exist_ok=True)
    
    partition = load_partition(INPUT, MANIFEST)
    
    train_terms = normalized_terms(partition.train)
    validation_terms = normalized_terms(partition.development)
    
    vocabulary = CharVocabulary(train_terms)
    
    
    _, development_cases = load_development_inputs(MANIFEST, INPUT, OUTPUT)
    
    database_path = RUN_DIRECTORY / "study.db"
    storage_url = f"sqlite:///{database_path.resolve()}"
    
    study_name = "neural_tuning_v1"
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        direction="maximize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    
    def run_trial(trial):
        return objective(
            trial,
            train_terms,
            validation_terms,
            vocabulary,
            development_cases,
            RUN_DIRECTORY,
        )
    
    study.optimize(
        run_trial,
        n_trials=N_TRIALS,
    )
    
    if not any(
        t.state == optuna.trial.TrialState.COMPLETE
        for t in study.trials
    ):
        print("No completed trials.")
        return

    best = study.best_trial
    best_directory = RUN_DIRECTORY / f"trial_{best.number:03d}"

    save_json(
        RUN_DIRECTORY / f"best_{len(study.trials):03d}.json",
        {
            "trial_number": best.number,
            "success_at_10": best.value,
            "config": json.loads(
                (best_directory / "config.json").read_text()
            ),
            "checkpoint_path": str(best_directory / "best.pt"),
        },
    )

    print(f"Best trial {best.number}: Success@10={best.value:.4f}")

if __name__ == "__main__":
    main()