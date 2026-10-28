from autocomplete.baseline.catalog import find_prefix_matches
from autocomplete.contract import is_compatible_completion, prepare_prefix


def suggest_popular(
    catalog: set[str],
    counts: dict[str, int],
    prefix: str,
    k: int = 10,
) -> list[str]:
    """Return up to k prefix-compatible catalog terms ranked by historical count.

    Args:
        catalog: Nonempty, normalized historical search terms.
        counts: Historical interaction counts for every term in the catalog.
        prefix: Raw typed prefix, normalized by the prefix lookup function.
        k: Maximum number of suggestions; must be a positive integer, excluding bool.

    Returns:
        Matching full terms ranked by descending count, with lexical tie-breaking.
        Returns fewer than k terms when fewer matches exist, or an empty list for
        an empty normalized prefix or no matches.

    Raises:
        TypeError: If k is not an integer or is a boolean.
        ValueError: If k is zero or negative.
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
