import pytest

from autocomplete.baseline.baselines import suggest_popular


def test_popularity_ranks_by_count_then_alphabetically():
    counts = {"cat toy": 1, "cat food": 2, "cat bed": 2, "dog food": 10}

    result = suggest_popular(set(counts), counts, "cat ", k=2)

    assert result == ["cat bed", "cat food"]


def test_empty_or_unmatched_prefix_returns_no_suggestions():
    assert suggest_popular({"cat"}, {"cat": 1}, "") == []
    assert suggest_popular({"cat"}, {"cat": 1}, "dog") == []


def test_complete_query_can_be_suggested_without_extension():
    assert suggest_popular({"cat"}, {"cat": 1}, "cat") == ["cat"]


def test_nonpositive_limit_is_rejected():
    with pytest.raises(ValueError):
        suggest_popular({"cat"}, {"cat": 1}, "ca", k=0)
