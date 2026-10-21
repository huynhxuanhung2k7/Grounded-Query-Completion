from dataclasses import asdict

import pytest

from autocomplete.data import DevelopmentCase
from autocomplete.evaluation import evaluate_popularity, score_case, summarize_scores


@pytest.mark.parametrize(
    ("suggestions", "truth", "expected"),
    [
        # First and second positions.
        (["cat food", "cat toy"], "cat food", (True, 1.0)),
        (["cat toy", "cat food"], "cat food", (True, 0.5)),
        # Rank ten counts; rank eleven does not.
        ([f"query {i}" for i in range(1, 12)], "query 10", (True, 0.1)),
        ([f"query {i}" for i in range(1, 12)], "query 11", (False, 0.0)),
        # Missing answer and empty output.
        (["cat toy", "cat bed"], "cat food", (False, 0.0)),
        ([], "cat food", (False, 0.0)),
    ],
)
def test_score_case(
    suggestions: list[str],
    truth: str,
    expected: tuple[bool, float],
) -> None:
    hit, reciprocal_rank = score_case(suggestions, truth)

    assert hit is expected[0]
    assert reciprocal_rank == pytest.approx(expected[1])


def test_summarize_scores_mixed_results() -> None:
    result = summarize_scores(
        [
            (True, 1.0),
            (True, 0.5),
            (False, 0.0),
        ]
    )

    assert result["request_count"] == 3
    assert result["success_at_10"] == pytest.approx(2 / 3)
    assert result["mrr_at_10"] == pytest.approx(0.5)


def test_summarize_scores_all_misses() -> None:
    assert summarize_scores([(False, 0.0), (False, 0.0)]) == {
        "request_count": 2,
        "success_at_10": 0.0,
        "mrr_at_10": 0.0,
    }


def test_summarize_scores_distinguishes_success_from_rank() -> None:
    result = summarize_scores([(True, 0.5), (True, 0.1)])

    assert result["request_count"] == 2
    assert result["success_at_10"] == 1.0
    assert result["mrr_at_10"] == pytest.approx(0.3)


def test_summarize_scores_single_first_rank_hit() -> None:
    assert summarize_scores([(True, 1.0)]) == {
        "request_count": 1,
        "success_at_10": 1.0,
        "mrr_at_10": 1.0,
    }


def test_summarize_scores_empty_input() -> None:
    assert summarize_scores([]) == {
        "request_count": 0,
        "success_at_10": None,
        "mrr_at_10": None,
    }


def test_evaluate_popularity_preserves_rows_and_summarizes() -> None:
    cases = [
        DevelopmentCase(
            query_id=202,
            prefix_index=1,
            raw_prefix="  CAT",
            normalized_prefix="cat",
            raw_final_search_term="Cat Food",
            normalized_final_search_term="cat food",
        ),
        DevelopmentCase(
            query_id=101,
            prefix_index=0,
            raw_prefix="phone",
            normalized_prefix="phone",
            raw_final_search_term="Phone Case",
            normalized_final_search_term="phone case",
        ),
    ]

    result = evaluate_popularity(
        catalog={"cat food", "cat toy"},
        counts={"cat food": 3, "cat toy": 5},
        cases=cases,
    )

    assert result["results"] == [
        {
            **asdict(cases[0]),
            "suggestions": ["cat toy", "cat food"],
            "hit_at_10": True,
            "reciprocal_rank_at_10": 0.5,
        },
        {
            **asdict(cases[1]),
            "suggestions": [],
            "hit_at_10": False,
            "reciprocal_rank_at_10": 0.0,
        },
    ]
    assert result["summary"] == {
        "request_count": 2,
        "success_at_10": 0.5,
        "mrr_at_10": 0.25,
    }


def test_evaluate_popularity_handles_empty_cases() -> None:
    assert evaluate_popularity(
        catalog={"cat food"},
        counts={"cat food": 3},
        cases=[],
    ) == {
        "results": [],
        "summary": {
            "request_count": 0,
            "success_at_10": None,
            "mrr_at_10": None,
        },
    }


def test_evaluate_popularity_limits_suggestions_to_ten() -> None:
    terms = [f"item {i:02d}" for i in range(11)]
    case = DevelopmentCase(
        query_id=1,
        prefix_index=0,
        raw_prefix="item",
        normalized_prefix="item",
        raw_final_search_term="item 10",
        normalized_final_search_term="item 10",
    )

    result = evaluate_popularity(
        catalog=set(terms),
        counts={term: 11 - i for i, term in enumerate(terms)},
        cases=[case],
    )

    [row] = result["results"]
    assert row["suggestions"] == terms[:10]
    assert row["hit_at_10"] is False
    assert row["reciprocal_rank_at_10"] == 0.0
    assert result["summary"] == {
        "request_count": 1,
        "success_at_10": 0.0,
        "mrr_at_10": 0.0,
    }
