"""M8.T1 only: choose one configuration value without training a model."""

import pytest
from optuna.trial import FixedTrial

from scripts.tune_neural import suggest_config


@pytest.mark.parametrize("hidden_dim", [256, 512])
def test_trial_selects_the_requested_hidden_width(hidden_dim):
    trial = FixedTrial({"hidden_dim": hidden_dim})

    assert suggest_config(trial) == {"hidden_dim": hidden_dim}
    assert trial.params == {"hidden_dim": hidden_dim}
