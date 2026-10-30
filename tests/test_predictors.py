import math
from types import SimpleNamespace

import pytest

from autocomplete import predictors
from autocomplete.model.vocabulary import EOS


def test_popularity_setup_loads_once_and_reuses_counts(monkeypatch):
    calls = []
    counts = {"cat food": 2, "cat toy": 1}

    def load_catalog():
        calls.append("loaded")

        return set(counts), counts

    monkeypatch.setattr(predictors, "load_catalog", load_catalog)
    suggest = predictors.make_suggester("popularity")

    assert suggest("cat ")["suggestions"] == ["cat food", "cat toy"]
    assert suggest("")["stop_reason"] == "empty_prefix"
    assert calls == ["loaded"]


def test_ngram_setup_uses_only_training_terms(monkeypatch, tmp_path):
    partition = SimpleNamespace(
        train=[{"final_search_term": "cat"}],
        development=[{"final_search_term": "car"}],
    )
    monkeypatch.setattr(predictors, "load_partition", lambda _: partition)

    scorer = predictors.load_predictor(
        "ngram", tmp_path / "raw", tmp_path / "manifest", tmp_path / "model"
    )

    assert "t" in scorer("ca")
    assert "r" not in scorer("ca")


def test_free_and_grounded_generation_share_the_same_candidate_pool(monkeypatch):
    probabilities = {"ca": {"t": 0.6, "r": 0.4}, "cat": {EOS: 1.0}, "car": {EOS: 1.0}}

    def score(text):
        return {token: math.log(value) for token, value in probabilities[text].items()}

    monkeypatch.setattr(predictors, "load_predictor", lambda *args: score)
    monkeypatch.setattr(predictors, "load_catalog", lambda: ({"car"}, {"car": 1}))
    free = predictors.make_suggester("neural", beam_width=2, candidate_budget=2)
    grounded = predictors.make_suggester(
        "neural", beam_width=2, candidate_budget=2, grounded=True
    )

    free_result = free("ca")
    grounded_result = grounded("ca")

    assert free_result["suggestions"] == ["cat", "car"]
    assert grounded_result["suggestions"] == ["car"]
    assert free_result["candidates"] == grounded_result["candidates"]


def test_neural_setup_rejects_a_different_training_manifest(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("original input description")
    monkeypatch.setattr(
        predictors,
        "load_checkpoint",
        lambda _: (None, None, {"manifest_sha256": "different"}),
    )

    with pytest.raises(ValueError, match="different manifest"):
        predictors.load_predictor(
            "neural", tmp_path / "raw", manifest, tmp_path / "model"
        )


def test_catalog_neural_adapter_normalizes_prefix_and_keeps_ranked_scores(monkeypatch):
    counts = {"cat food": 2, "cat toy": 1}

    def score(prefix, catalog):
        assert prefix == "cat "
        assert catalog == set(counts)

        return [
            {"completion": "cat toy", "log_score": -0.2},
            {"completion": "cat food", "log_score": -0.4},
        ]

    monkeypatch.setattr(predictors, "load_predictor", lambda *args: score)
    monkeypatch.setattr(predictors, "load_catalog", lambda: (set(counts), counts))
    suggest = predictors.make_suggester("catalog_neural")

    result = suggest(" CAT ")

    assert result["suggestions"] == ["cat toy", "cat food"]
    assert result["scores"] == [-0.2, -0.4]
