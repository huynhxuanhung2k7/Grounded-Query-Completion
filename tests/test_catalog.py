from autocomplete.baseline.catalog import (
    build_catalog,
    count_history_terms,
    find_prefix_matches,
)


def test_catalog_normalizes_and_deduplicates_terms():
    records = [
        {"final_search_term": " CAT Food "},
        {"final_search_term": "cat food"},
        {"final_search_term": "   "},
    ]

    assert build_catalog(records) == {"cat food"}


def test_counts_use_interactions_instead_of_supplied_popularity():
    records = [
        {"final_search_term": "cat", "popularity": 999},
        {"final_search_term": "CAT", "popularity": 999},
    ]

    assert count_history_terms(records) == {"cat": 2}


def test_prefix_matching_preserves_word_boundaries_and_alphabetical_order():
    catalog = {"cat toy", "caterpillar", "cat food"}

    assert find_prefix_matches(catalog, " CAT ") == ["cat food", "cat toy"]
    assert find_prefix_matches(catalog, "bird") == []


def test_empty_history_and_prefix_have_no_matches():
    assert build_catalog([]) == set()
    assert count_history_terms([]) == {}
    assert find_prefix_matches({"cat"}, "") == []
