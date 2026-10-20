import pytest

from autocomplete.evaluation import score_case


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