import copy
from dataclasses import asdict

import pytest

from autocomplete.data import DevelopmentCase
from autocomplete.evaluation import (
    beam_prediction,
    completion_prediction,
    evaluate_cases,
)


def case(prefix="ca", truth="cat", query_id=1):
    return DevelopmentCase(query_id, 0, prefix, prefix, truth, truth)


def test_generic_evaluation_receives_only_prefix_and_preserves_case_fields():
    cases = [case(), case("dog", "dog food", 2)]
    received = []

    def predict(prefix):
        received.append(prefix)
        return {
            "suggestions": ["car", "cat"] if prefix == "ca" else [],
            "stop_reason": "eos" if prefix == "ca" else "no_candidates",
        }

    result = evaluate_cases(cases, predict)
    assert received == ["ca", "dog"]
    assert result["summary"] == {
        "request_count": 2,
        "success_at_10": 0.5,
        "mrr_at_10": 0.25,
    }
    assert result["prediction_summary"] == {"response_coverage": 0.5}
    assert all(
        result["results"][0][key] == value for key, value in asdict(cases[0]).items()
    )
    assert "timing_summary" not in result


def test_empty_generic_evaluation_keeps_none_metrics_and_no_calls():
    result = evaluate_cases([], lambda _: pytest.fail("Predictor should not be called"))
    assert result == {
        "results": [],
        "summary": {"request_count": 0, "success_at_10": None, "mrr_at_10": None},
        "prediction_summary": {"response_coverage": None},
    }


def test_empty_predictions_are_misses_without_timing_machinery():
    result = evaluate_cases(
        [case()],
        lambda _: {"suggestions": [], "stop_reason": "length_limit"},
    )
    assert result["summary"]["request_count"] == 1
    assert result["summary"]["success_at_10"] == 0
    assert result["prediction_summary"]["response_coverage"] == 0
    assert "generation_seconds" not in result["results"][0]
    assert "timing_summary" not in result


@pytest.mark.parametrize(
    "field",
    ["query_id", "normalized_final_search_term", "hit_at_10"],
)
def test_case_fields_and_scores_take_precedence_over_prediction_metadata(field):
    result = evaluate_cases(
        [case()],
        lambda _: {
            "suggestions": [],
            "stop_reason": "no_candidates",
            field: "not the real value",
        },
    )
    row = result["results"][0]
    assert row["query_id"] == 1
    assert row["normalized_final_search_term"] == "cat"
    assert row["hit_at_10"] is False


@pytest.mark.parametrize(
    "suggestions",
    [
        ["cat", "cat"],
        ["Cat"],
        [""],
        ["dog"],
        ["ca" + "x" * 65],
        [f"cat{i}" for i in range(11)],
    ],
)
def test_invalid_outputs_are_rejected_not_repaired(suggestions):
    with pytest.raises(ValueError):
        evaluate_cases(
            [case()], lambda _: {"suggestions": suggestions, "stop_reason": "eos"}
        )


def test_rejected_input_cannot_produce_suggestions():
    with pytest.raises(ValueError, match="Rejected"):
        evaluate_cases(
            [case("", "cat")], lambda _: {"suggestions": ["cat"], "stop_reason": "eos"}
        )


def test_predictor_failure_is_not_dropped_or_disguised_as_a_miss():
    def predict(_):
        raise RuntimeError("broken setup")

    with pytest.raises(RuntimeError, match="broken setup"):
        evaluate_cases([case()], predict)


def test_completion_adapter_keeps_legacy_row_shape_and_score():
    prediction = {"completion": "cat", "log_score": -2.0, "stop_reason": "eos"}
    result = evaluate_cases([case()], lambda _: completion_prediction(prediction))
    assert result["results"] == [
        {
            **asdict(case()),
            "suggestions": ["cat"],
            "log_score": -2.0,
            "stop_reason": "eos",
            "hit_at_10": True,
            "reciprocal_rank_at_10": 1.0,
        }
    ]
    assert (
        completion_prediction(
            {"completion": None, "log_score": None, "stop_reason": "length_limit"}
        )["suggestions"]
        == []
    )


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -float("inf")])
def test_completion_adapter_rejects_nonfinite_scores(score):
    with pytest.raises(ValueError, match="finite"):
        completion_prediction(
            {"completion": "cat", "log_score": score, "stop_reason": "eos"}
        )


@pytest.fixture
def pool():
    return {
        "candidates": [
            {"completion": f"cat{i:02d}", "log_score": -float(i + 1)} for i in range(12)
        ],
        "candidate_count": 12,
        "candidate_budget": 50,
        "beam_width": 20,
        "stop_reason": "score_bound",
    }


def test_beam_adapter_preserves_full_pool_without_fingerprint_or_mutation(pool):
    before = copy.deepcopy(pool)
    prediction = beam_prediction(pool)
    result = evaluate_cases([case(truth="cat11")], lambda _: prediction)
    row = result["results"][0]
    assert len(row["candidates"]) == 12
    assert len(row["suggestions"]) == 10
    assert row["hit_at_10"] is False  # Pool membership is not a displayed hit.
    assert "pool_sha256" not in row
    assert pool == before


def test_duplicate_displayed_suggestions_are_rejected_before_scoring(pool):
    pool["candidates"][1]["completion"] = pool["candidates"][0]["completion"]
    with pytest.raises(ValueError):
        evaluate_cases([case()], lambda _: beam_prediction(pool))


def test_evaluator_scores_displayed_suggestions_not_hidden_pool_entries(pool):
    pool["candidates"][-1]["completion"] = "dog"
    result = evaluate_cases([case(truth="dog")], lambda _: beam_prediction(pool))
    assert result["summary"]["success_at_10"] == 0


def test_free_and_grounded_share_the_same_scored_pool(pool):
    free = beam_prediction(pool)
    grounded = beam_prediction(pool, catalog={"cat02", "cat11"})
    assert grounded["suggestions"] == ["cat02", "cat11"]
    assert grounded["candidates"] == free["candidates"] == pool["candidates"]
    empty = beam_prediction(pool, catalog=set())
    assert empty["suggestions"] == []
    assert empty["stop_reason"] == "no_catalog_candidates"


def test_suggestions_are_checked_even_when_pool_metadata_is_present():
    prediction = {
        "suggestions": ["dog"],
        "candidates": [{"completion": "cat", "log_score": -1.0}],
        "stop_reason": "score_bound",
    }
    with pytest.raises(ValueError, match="prefix/length"):
        evaluate_cases([case()], lambda _: prediction)
