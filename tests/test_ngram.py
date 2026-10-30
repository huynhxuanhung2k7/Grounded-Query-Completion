import math

import pytest

from autocomplete.baseline.ngram import EOS, NgramModel


def test_fit_counts_repeated_terms_and_end_of_query():
    model = NgramModel(order=3)
    model.fit(["cat", "cat", "car"])

    assert model.counts["ca"] == {"t": 2, "r": 1}
    assert model.counts["at"] == {EOS: 2}


def test_smoothed_probabilities_match_a_hand_calculation():
    model = NgramModel(alpha=0.1)
    model.fit(["cat", "car"])

    probabilities = model.next_log_probs("ca")

    assert math.exp(probabilities["t"]) == pytest.approx(1.1 / 2.5)
    assert math.exp(probabilities[EOS]) == pytest.approx(0.1 / 2.5)
    assert sum(math.exp(value) for value in probabilities.values()) == pytest.approx(1)


def test_unknown_context_backs_off_to_a_known_suffix_or_unigram():
    model = NgramModel()
    model.fit(["cat", "car"])

    assert model.next_log_probs("za") == model.next_log_probs("a")
    assert model.next_log_probs("zzz") == model.next_log_probs("")


def test_greedy_completion_scores_the_suffix_and_eos():
    model = NgramModel()
    model.fit(["cat"])

    result = model.greedy_complete("ca")
    expected = model.next_log_probs("ca")["t"] + model.next_log_probs("cat")[EOS]

    assert result["completion"] == "cat"
    assert result["log_score"] == pytest.approx(expected)
    assert model.greedy_complete("cat")["completion"] == "cat"


def test_empty_training_data_is_rejected():
    with pytest.raises(ValueError, match="nonempty"):
        NgramModel().fit([])
