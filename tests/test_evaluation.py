import pytest

from autocomplete.evaluation import score_case, summarize_scores


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
    result = summarize_scores([
        (True, 1.0),
        (True, 0.5),
        (False, 0.0),
    ])

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