import math
from collections import Counter

import pytest

from autocomplete.contract import MAX_ADDED_LENGTH, MAX_INPUT_LENGTH
from autocomplete.ngram import EOS, NgramModel


@pytest.mark.parametrize("order", [0, -1, True, False, 1.5, "4"])
def test_invalid_order_is_rejected(order: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        NgramModel(order=order)


@pytest.mark.parametrize("alpha", [0, -0.1, math.inf, -math.inf, math.nan])
def test_invalid_alpha_is_rejected(alpha: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        NgramModel(alpha=alpha)


def test_fit_counts_all_suffixes_and_preserves_repeated_terms() -> None:
    model = NgramModel()
    model.fit(["cat", "cat"])

    assert model.counts == {
        "": Counter({"c": 2, "a": 2, "t": 2, EOS: 2}),
        "c": Counter({"a": 2}),
        "a": Counter({"t": 2}),
        "ca": Counter({"t": 2}),
        "t": Counter({EOS: 2}),
        "at": Counter({EOS: 2}),
        "cat": Counter({EOS: 2}),
    }
    assert model.alphabet == {"c", "a", "t", EOS}


def test_fit_accepts_generator_and_skips_empty_terms() -> None:
    model = NgramModel()
    model.fit(term for term in ["", "cat", ""])
    assert model.counts[""][EOS] == 1


@pytest.mark.parametrize("terms", [[], [""], ["", ""]])
def test_fit_requires_a_nonempty_term(terms: list[str]) -> None:
    with pytest.raises(ValueError, match="nonempty term"):
        NgramModel().fit(terms)


def test_refit_replaces_counts_and_alphabet() -> None:
    model = NgramModel()
    model.fit(["cat"])
    model.fit(["dog"])
    assert model.alphabet == {"d", "o", "g", EOS}
    assert "ca" not in model.counts
    assert model.counts[""][EOS] == 1


@pytest.mark.parametrize("order", [1, 2, 4])
def test_contexts_respect_configured_order(order: int) -> None:
    model = NgramModel(order=order)
    model.fit(["abcde"])
    assert max(map(len, model.counts)) == order - 1
    assert model.counts[""][EOS] == 1


@pytest.mark.parametrize("method", ["next_log_probs", "greedy_complete"])
def test_prediction_requires_fit(method: str) -> None:
    with pytest.raises(ValueError, match="fit"):
        getattr(NgramModel(), method)("ca")


@pytest.mark.parametrize("alpha", [0.1, 0.5])
def test_smoothing_matches_hand_calculation(alpha: float) -> None:
    model = NgramModel(alpha=alpha)
    model.fit(["cat", "car"])
    result = model.next_log_probs("ca")
    denominator = 2 + alpha * 5

    assert set(result) == {"c", "a", "t", "r", EOS}
    for token, log_prob in result.items():
        count = 1 if token in {"t", "r"} else 0
        assert log_prob == pytest.approx(math.log((count + alpha) / denominator))
    assert sum(map(math.exp, result.values())) == pytest.approx(1.0)


def test_longest_observed_suffix_wins_over_shorter_context() -> None:
    model = NgramModel()
    model.fit(["cat", "bat", "bar"])
    # "a" has r/t observations, but "ca" has only t.
    assert model.next_log_probs("xxca") == model.next_log_probs("ca")
    assert model.next_log_probs("ca") != model.next_log_probs("a")
    assert model.next_log_probs("xxa") == model.next_log_probs("a")


def test_unseen_context_falls_back_to_unigram_without_mutating_counts() -> None:
    model = NgramModel()
    model.fit(["cat", "car"])
    before = {context: counter.copy() for context, counter in model.counts.items()}
    assert model.next_log_probs("zzz") == model.next_log_probs("")
    assert model.counts == before


def test_greedy_score_covers_only_suffix_and_eos() -> None:
    model = NgramModel()
    model.fit(["cat", "cat"])
    expected = model.next_log_probs("ca")["t"] + model.next_log_probs("cat")[EOS]
    assert model.greedy_complete("  CA") == {
        "completion": "cat",
        "log_score": pytest.approx(expected),
        "stop_reason": "eos",
    }


def test_greedy_permits_immediate_eos() -> None:
    model = NgramModel()
    model.fit(["cat"])
    assert model.greedy_complete("cat") == {
        "completion": "cat",
        "log_score": pytest.approx(model.next_log_probs("cat")[EOS]),
        "stop_reason": "eos",
    }


def test_greedy_breaks_ties_lexically() -> None:
    for terms in (["cat", "car"], ["car", "cat"]):
        model = NgramModel()
        model.fit(terms)
        assert model.greedy_complete("ca")["completion"] == "car"


def test_unknown_input_characters_are_preserved() -> None:
    model = NgramModel()
    model.fit(["cat"])
    assert model.greedy_complete("xxca")["completion"] == "xxcat"


def test_trailing_word_boundary_is_preserved() -> None:
    model = NgramModel()
    model.fit(["cat food"])
    assert model.greedy_complete(" CAT  ")["completion"] == "cat food"


def test_immediate_eos_cannot_erase_trailing_space(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = NgramModel()
    model.fit(["cat"])
    monkeypatch.setattr(model, "next_log_probs", lambda context: {EOS: 0.0})
    assert model.greedy_complete("cat ") == {
        "completion": None,
        "log_score": None,
        "stop_reason": "no_candidates",
    }


@pytest.mark.parametrize(
    ("prefix", "reason"),
    [
        ("", "empty_prefix"),
        (" \t\n", "empty_prefix"),
        ("a" * (MAX_INPUT_LENGTH + 1), "input_too_long"),
    ],
)
def test_rejected_input_returns_explicit_empty_outcome(
    prefix: str, reason: str
) -> None:
    model = NgramModel()
    model.fit(["cat"])
    assert model.greedy_complete(prefix) == {
        "completion": None,
        "log_score": None,
        "stop_reason": reason,
    }


@pytest.mark.parametrize("added_length", [MAX_ADDED_LENGTH, MAX_ADDED_LENGTH + 1])
def test_eos_is_allowed_at_exact_cap_but_no_extra_character(added_length: int) -> None:
    # Distinct normalized characters make each next-character choice unambiguous.
    suffix = "".join(chr(0x4E00 + index) for index in range(added_length))
    model = NgramModel()
    model.fit(["p" + suffix])
    result = model.greedy_complete("p")
    if added_length == MAX_ADDED_LENGTH:
        assert result["completion"] == "p" + suffix
        assert result["stop_reason"] == "eos"
        assert math.isfinite(result["log_score"])
    else:
        assert result == {
            "completion": None,
            "log_score": None,
            "stop_reason": "length_limit",
        }


def test_repeating_generation_stops_without_returning_unfinished_query() -> None:
    model = NgramModel()
    model.fit(["a" * 8])
    assert model.greedy_complete("a") == {
        "completion": None,
        "log_score": None,
        "stop_reason": "length_limit",
    }
