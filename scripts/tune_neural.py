"""M8.T1: rebuild hyperparameter search one small behavior at a time."""

from optuna.trial import Trial


def suggest_config(trial: Trial) -> dict[str, int]:
    """Choose hidden_dim from 256 or 512 for this trial."""
    hidden_dim = trial.suggest_int("hidden_dim", 256, 512)
    return {"hidden_dim": hidden_dim}