from autocomplete.baseline.catalog import find_prefix_matches
from autocomplete.contract import is_compatible_completion, prepare_prefix


def suggest_popular(
    catalog: set[str],
    counts: dict[str, int],
    prefix: str,
    k: int = 10,
) -> list[str]:
    """Rank prefix matches by count, breaking ties alphabetically.

    Args:
        catalog: Normalized historical terms.
        counts: Interaction count per term.
        prefix: Raw typed prefix.
        k: Maximum number of suggestions.
    Returns:
        Up to k compatible full queries.
    """
    if type(k) is not int:
        raise TypeError("k must be an integer")

    if k <= 0:
        raise ValueError("k must be greater than zero")

    normalized, reason = prepare_prefix(prefix)

    if reason is not None:
        return []

    eligible = [
        term
        for term in find_prefix_matches(catalog, prefix)
        if is_compatible_completion(normalized, term)
    ]

    ranked = sorted(eligible, key=lambda term: (-counts[term], term))

    return ranked[:k]


if __name__ == "__main__":
    counts = {"cat food": 2, "cat bed": 2, "cat toy": 1, "dog food": 10}
    catalog = set(counts)
    print("Historical counts:", counts)
    print("Top two for 'cat ':", suggest_popular(catalog, counts, "cat ", k=2))
    print("No matching term:", suggest_popular(catalog, counts, "bird"))
