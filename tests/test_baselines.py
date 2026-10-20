import pytest

from autocomplete.baselines import suggest_popular


CATALOG = {"cat bed", "cat food", "cat toy", "dog food"}
COUNTS = {
    "cat bed": 2,
    "cat food": 5,
    "cat toy": 5,
    "dog food": 20,
}


def test_suggest_popular_ranks_by_count_with_lexical_ties() -> None:
    assert suggest_popular(
        catalog=CATALOG,
        counts=COUNTS,
        prefix="cat",
        k=3,
    ) == ["cat food", "cat toy", "cat bed"]


def test_suggest_popular_ranks_before_taking_top_k() -> None:
    # "cat bed" comes first alphabetically but has the lowest count.
    assert suggest_popular(
        catalog=CATALOG,
        counts=COUNTS,
        prefix="cat",
        k=1,
    ) == ["cat food"]


def test_suggest_popular_returns_available_matches_when_fewer_than_k() -> None:
    assert suggest_popular(
        catalog=CATALOG,
        counts=COUNTS,
        prefix="cat",
        k=10,
    ) == ["cat food", "cat toy", "cat bed"]


def test_suggest_popular_normalizes_prefix() -> None:
    assert suggest_popular(
        catalog=CATALOG,
        counts=COUNTS,
        prefix="  CAT   ",
        k=2,
    ) == ["cat food", "cat toy"]


def test_suggest_popular_returns_empty_when_no_matches() -> None:
    assert suggest_popular(
        catalog=CATALOG,
        counts=COUNTS,
        prefix="phone",
        k=10,
    ) == []


def test_suggest_popular_handles_empty_catalog() -> None:
    assert suggest_popular(
        catalog=set(),
        counts={},
        prefix="cat",
        k=10,
    ) == []


@pytest.mark.parametrize("prefix", ["", "   ", "\t\n"])
def test_suggest_popular_handles_empty_normalized_prefix(prefix: str) -> None:
    assert suggest_popular(
        catalog=CATALOG,
        counts=COUNTS,
        prefix=prefix,
        k=10,
    ) == []


@pytest.mark.parametrize("k", [True, False, 1.5, "10", None])
def test_suggest_popular_rejects_non_integer_k(k: object) -> None:
    with pytest.raises(TypeError, match="integer"):
        suggest_popular(
            catalog=CATALOG,
            counts=COUNTS,
            prefix="cat",
            k=k,
        )


@pytest.mark.parametrize("k", [0, -1])
def test_suggest_popular_rejects_non_positive_k(k: int) -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        suggest_popular(
            catalog=CATALOG,
            counts=COUNTS,
            prefix="cat",
            k=k,
        )


def test_suggest_popular_validates_k_even_with_empty_prefix() -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        suggest_popular(
            catalog=CATALOG,
            counts=COUNTS,
            prefix="",
            k=0,
        )