from pathlib import Path

from autocomplete import config


def test_paths_and_decoder_budgets_are_shared():
    assert isinstance(config.CHECKPOINT, Path)
    assert not config.INPUT.is_absolute()
    assert config.HISTORY_CATALOG.parts[0] == "datasets"

    assert config.BEAM_WIDTH > 0
    assert config.CANDIDATE_BUDGET >= config.MAX_SUGGESTIONS == 10
