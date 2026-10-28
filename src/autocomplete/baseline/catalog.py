from collections.abc import Iterable

from autocomplete.data.normalization import normalize_prefix, normalize_term


def build_catalog(
    history_records: Iterable[dict[str, object]],
) -> set[str]:
    """Build a catalog of distinct, nonempty normalized historical search terms.

    Args:
        history_records: Validated, chronologically eligible interaction records.
            Each record must contain a string-valued ``final_search_term``.
            Supports lists and single-pass iterators.

    Returns:
        A set of normalized final terms, excluding empty normalized strings.
        Empty input produces an empty set.

    Notes:
        The caller selects eligible history before calling this function.
        This function does not perform chronological partitioning or count
        repeated interactions.
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
    """Count historical interactions per nonempty normalized final search term.

    Args:
        history_records: Validated, chronologically eligible records with unique
            query IDs and string-valued ``final_search_term`` fields.
            Supports lists and single-pass iterators.

    Returns:
        A dictionary mapping each normalized term to its interaction count.
        Empty input or exclusively empty normalized terms produces an empty
        dictionary.

    Notes:
        The caller handles eligibility and duplicate query IDs. Different
        interactions with the same normalized term each contribute one count.
        Supplied popularity metadata is ignored.
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
    """Return catalog terms starting with the normalized prefix, sorted lexically.

    Args:
        catalog: A set of nonempty, normalized historical search terms.
        prefix: Raw typed text, normalized while preserving a trailing word boundary.

    Returns:
        Matching full terms in lexical order. Returns an empty list if the
        normalized prefix is empty or no terms match.

    Notes:
        Matching preserves the distinction between "cat" and "cat ".
        Results are not ranked by popularity or limited to a top-k subset.
    """
    normalized_prefix = normalize_prefix(prefix)

    if not normalized_prefix:
        return []

    return sorted(term for term in catalog if term.startswith(normalized_prefix))
