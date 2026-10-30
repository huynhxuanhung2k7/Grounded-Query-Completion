import pytest

from autocomplete.data import DevelopmentCase
from autocomplete.evaluation import (
    beam_prediction,
    completion_prediction,
    evaluate_cases,
    prediction_summary,
    score_case,
    summarize_scores,
)


def test_exact_match_scores_rank_and_top_ten_boundary():
    assert score_case(["cat food", "cat toy"], "cat toy") == (True, 0.5)
    assert score_case(["cat food"], "cat bed") == (False, 0)
    assert score_case([f"cat{i}" for i in range(10)] + ["cat"], "cat") == (False, 0)


def test_summary_includes_misses_and_handles_no_requests():
    assert summarize_scores([(True, 0.5), (False, 0)]) == {
        "request_count": 2,
        "success_at_10": 0.5,
        "mrr_at_10": 0.25,
    }
    assert summarize_scores([]) == {
        "request_count": 0,
        "success_at_10": None,
        "mrr_at_10": None,
    }
    assert prediction_summary([]) == {"response_coverage": None}


def test_evaluation_passes_only_prefixes_and_retains_unanswered_requests():
    cases = [
        DevelopmentCase(1, 0, " CA", "ca", "cat", "cat"),
        DevelopmentCase(2, 0, "dog", "dog", "dog food", "dog food"),
    ]
    received = []

    def predict(prefix):
        received.append(prefix)

        return {"suggestions": ["car", "cat"] if prefix == " CA" else []}

    result = evaluate_cases(cases, predict)

    assert received == [" CA", "dog"]
    assert result["summary"] == {
        "request_count": 2,
        "success_at_10": 0.5,
        "mrr_at_10": 0.25,
    }
    assert result["prediction_summary"] == {"response_coverage": 0.5}
    assert result["results"][0]["normalized_final_search_term"] == "cat"


def test_duplicate_predictions_are_rejected():
    case = DevelopmentCase(1, 0, "ca", "ca", "cat", "cat")

    with pytest.raises(ValueError, match="unique"):
        evaluate_cases([case], lambda _: {"suggestions": ["cat", "cat"]})


def test_prediction_adapters_preserve_the_pool_and_empty_outputs():
    candidates = [
        {"completion": "car", "log_score": -1.0},
        {"completion": "cat", "log_score": -2.0},
    ]
    pool = {"candidates": candidates, "stop_reason": "exhausted"}

    grounded = beam_prediction(pool, catalog={"cat"})
    empty = beam_prediction(pool, catalog=set())

    assert grounded["suggestions"] == ["cat"]
    assert grounded["candidates"] == candidates
    assert empty["suggestions"] == []
    assert empty["stop_reason"] == "no_catalog_candidates"
    assert (
        completion_prediction(
            {"completion": None, "log_score": None, "stop_reason": "length_limit"}
        )["suggestions"]
        == []
    )
