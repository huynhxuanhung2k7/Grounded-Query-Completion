from autocomplete.baseline.catalog import (
    build_catalog,
    count_history_terms,
    find_prefix_matches,
)


def test_build_catalog_normalizes_and_deduplicates() -> None:
    records = [
        {"final_search_term": "  Wireless   Mouse "},
        {"final_search_term": "wireless mouse"},
        {"final_search_term": "Mouse Pad"},
    ]

    assert build_catalog(records) == {"wireless mouse", "mouse pad"}


def test_build_catalog_excludes_empty_normalized_terms() -> None:
    records = [
        {"final_search_term": ""},
        {"final_search_term": "   "},
        {"final_search_term": "\t\n"},
        {"final_search_term": "Mouse"},
    ]

    assert build_catalog(records) == {"mouse"}


def test_build_catalog_handles_empty_input() -> None:
    assert build_catalog([]) == set()


def test_build_catalog_accepts_single_pass_iterator() -> None:
    records = (
        {"final_search_term": term} for term in ["Keyboard", "  KEYBOARD  ", "Mouse"]
    )

    assert build_catalog(records) == {"keyboard", "mouse"}


def test_build_catalog_preserves_meaningful_punctuation() -> None:
    records = [
        {"final_search_term": " C++ "},
        {"final_search_term": "C#"},
    ]

    assert build_catalog(records) == {"c++", "c#"}


def test_count_history_terms_counts_repeated_interactions() -> None:
    records = [
        {"query_id": 1, "final_search_term": "mouse"},
        {"query_id": 2, "final_search_term": "mouse"},
        {"query_id": 3, "final_search_term": "keyboard"},
    ]

    assert count_history_terms(records) == {
        "mouse": 2,
        "keyboard": 1,
    }


def test_count_history_terms_normalizes_before_counting() -> None:
    records = [
        {"final_search_term": "  Wireless   Mouse "},
        {"final_search_term": "WIRELESS MOUSE"},
        {"final_search_term": "wireless mouse"},
    ]

    assert count_history_terms(records) == {"wireless mouse": 3}


def test_count_history_terms_excludes_empty_normalized_terms() -> None:
    records = [
        {"final_search_term": ""},
        {"final_search_term": "   "},
        {"final_search_term": "\t\n"},
        {"final_search_term": "Mouse"},
    ]

    assert count_history_terms(records) == {"mouse": 1}


def test_count_history_terms_handles_empty_input() -> None:
    assert count_history_terms([]) == {}


def test_count_history_terms_accepts_single_pass_iterator() -> None:
    records = ({"final_search_term": term} for term in ["Mouse", " mouse ", "Keyboard"])

    assert count_history_terms(records) == {
        "mouse": 2,
        "keyboard": 1,
    }


def test_count_history_terms_preserves_distinct_punctuation() -> None:
    records = [
        {"final_search_term": "C++"},
        {"final_search_term": " c++ "},
        {"final_search_term": "C#"},
    ]

    assert count_history_terms(records) == {
        "c++": 2,
        "c#": 1,
    }


def test_count_history_terms_ignores_supplied_popularity() -> None:
    records = [
        {"final_search_term": "mouse", "popularity": 1000},
        {"final_search_term": "mouse", "popularity": 1000},
        {"final_search_term": "keyboard", "popularity": 5000},
    ]

    assert count_history_terms(records) == {
        "mouse": 2,
        "keyboard": 1,
    }


def test_find_prefix_matches_returns_matching_full_terms() -> None:
    catalog = {"wireless mouse", "wireless keyboard", "mouse pad"}

    assert find_prefix_matches(catalog, "wireless") == [
        "wireless keyboard",
        "wireless mouse",
    ]


def test_find_prefix_matches_normalizes_input() -> None:
    catalog = {"wireless mouse", "wireless keyboard", "mouse pad"}

    assert find_prefix_matches(catalog, "  WIRELESS   ") == [
        "wireless keyboard",
        "wireless mouse",
    ]


def test_find_prefix_matches_preserves_trailing_boundary() -> None:
    catalog = {"cat", "cat food", "cat toy", "catalog"}

    assert find_prefix_matches(catalog, "cat") == [
        "cat",
        "cat food",
        "cat toy",
        "catalog",
    ]
    assert find_prefix_matches(catalog, "cat ") == [
        "cat food",
        "cat toy",
    ]


def test_find_prefix_matches_handles_empty_or_whitespace_prefix() -> None:
    catalog = {"mouse", "keyboard"}

    for prefix in ["", "   ", "\t\n"]:
        assert find_prefix_matches(catalog, prefix) == []


def test_find_prefix_matches_handles_no_matches() -> None:
    assert find_prefix_matches({"mouse", "keyboard"}, "phone") == []


def test_find_prefix_matches_handles_empty_catalog() -> None:
    assert find_prefix_matches(set(), "mouse") == []


def test_find_prefix_matches_preserves_punctuation() -> None:
    catalog = {"c++ book", "c# book", "coffee"}

    assert find_prefix_matches(catalog, "C++") == ["c++ book"]
