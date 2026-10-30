import math

import pytest
import torch

from autocomplete.decoding import (
    NeuralPredictor,
    beam_candidates,
    exact_filter,
    greedy_complete,
)
from autocomplete.model.model import NeuralCharModel
from autocomplete.model.vocabulary import EOS, PAD, UNK, CharVocabulary


def toy_scores(text):
    probabilities = {
        "ca": {"t": 0.6, "r": 0.4},
        "cat": {EOS: 0.5, "s": 0.5},
        "car": {EOS: 1.0},
        "cats": {EOS: 1.0},
    }

    return {token: math.log(value) for token, value in probabilities[text].items()}


def tiny_predictor():
    torch.manual_seed(42)
    vocabulary = CharVocabulary(["cat", "car"])
    model = NeuralCharModel(
        len(vocabulary), context_length=2, embedding_dim=4, hidden_dim=8
    )

    return NeuralPredictor(model, vocabulary)


def test_greedy_beam_and_exact_filter_match_hand_computed_paths():
    greedy = greedy_complete("ca", toy_scores)
    pool = beam_candidates("ca", toy_scores, beam_width=2, candidate_budget=3)
    candidates = pool["candidates"]
    before = [row.copy() for row in candidates]

    assert greedy["completion"] == "cat"
    assert greedy["log_score"] == pytest.approx(math.log(0.3))
    assert [row["completion"] for row in candidates] == ["car", "cat", "cats"]
    assert [math.exp(row["log_score"]) for row in candidates] == pytest.approx(
        [0.4, 0.3, 0.3]
    )
    assert exact_filter(candidates, {"cat", "cats"}) == before[1:]
    assert candidates == before


def test_complete_prefix_can_end_and_empty_input_is_rejected():
    result = greedy_complete("cat", lambda _: {EOS: 0.0})

    assert result["completion"] == "cat"
    assert result["log_score"] == 0
    assert (
        greedy_complete("", lambda _: pytest.fail("No prediction expected"))[
            "stop_reason"
        ]
        == "empty_prefix"
    )


def test_generation_without_eos_stops_at_the_length_limit():
    scores = lambda _: {"a": 0.0}

    greedy = greedy_complete("a", scores)
    beam = beam_candidates("a", scores, beam_width=1, candidate_budget=2)

    assert greedy["completion"] is None
    assert greedy["stop_reason"] == "length_limit"
    assert beam["candidates"] == []
    assert beam["stop_reason"] == "length_limit"


def test_neural_adapter_truncates_context_and_excludes_special_outputs():
    predictor = tiny_predictor()

    rows = predictor.next_log_probs_batch(["at", "xxcat"])
    scores = predictor.next_log_probs("ca")

    assert rows.shape == (2, len(predictor.vocabulary))
    torch.testing.assert_close(rows[0], rows[1], rtol=0, atol=0)
    assert PAD not in scores and UNK not in scores
    assert EOS in scores


def test_batched_catalog_scores_match_individual_suffix_and_eos_scores():
    predictor = tiny_predictor()
    catalog = {"ca", "cat", "car", "cap", "dog"}
    expected = {}

    for term in ("ca", "cat", "car"):
        score = 0.0

        for position in range(2, len(term) + 1):
            token = term[position] if position < len(term) else EOS
            score += predictor.next_log_probs(term[:position])[token]

        expected[term] = score

    ranked = predictor.score_catalog_terms("ca", catalog, batch_size=2)
    expected_order = sorted(expected, key=lambda term: (-expected[term], term))

    assert [row["completion"] for row in ranked] == expected_order
    assert {row["completion"]: row["log_score"] for row in ranked} == pytest.approx(
        expected
    )
