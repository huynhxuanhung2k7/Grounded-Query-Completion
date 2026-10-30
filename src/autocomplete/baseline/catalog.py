from collections.abc import Iterable

from autocomplete.data.normalization import normalize_prefix, normalize_term


def build_catalog(
    history_records: Iterable[dict[str, object]],
) -> set[str]:
    """Collect distinct historical queries, skipping empty terms.

    Args:
        history_records: Interactions already checked for time eligibility.
    Returns:
        A set of normalized terms.
    """
    catalog = set()

    for record in history_records:
        final_search_term = record["final_search_term"]
        term = normalize_term(final_search_term)

        if term == "":
            continue

        catalog.add(term)

    return catalog


def count_history_terms(
    history_records: Iterable[dict[str, object]],
) -> dict[str, int]:
    """Count interactions per query, ignoring supplied popularity metadata.

    Args:
        history_records: Eligible interactions with unique IDs.
    Returns:
        Normalized term to interaction count.
    """
    counts: dict[str, int] = {}

    for record in history_records:
        term = normalize_term(record["final_search_term"])

        if term:
            counts[term] = counts.get(term, 0) + 1

    return counts


def find_prefix_matches(
    catalog: set[str],
    prefix: str,
) -> list[str]:
    """Find matching terms in alphabetical order.

    Args:
        catalog: Normalized historical terms.
        prefix: Raw typed prefix.
    Returns:
        Matching full terms; empty input gives no matches.
    """
    normalized_prefix = normalize_prefix(prefix)

    if not normalized_prefix:
        return []

    return sorted(term for term in catalog if term.startswith(normalized_prefix))


if __name__ == "__main__":
    history = [
        {"final_search_term": term}
        for term in (" Cat food ", "cat food", "cat toy", "   ")
    ]
    catalog = build_catalog(history)
    print("Unique normalized terms:", sorted(catalog))
    print("Interaction counts:", count_history_terms(history))
    print("Matches for 'cat ':", find_prefix_matches(catalog, "cat "))
    print("Empty prefix:", find_prefix_matches(catalog, ""))
