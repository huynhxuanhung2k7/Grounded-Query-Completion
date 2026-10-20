from autocomplete.catalog import (
    find_prefix_matches
)


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
    
    prefix_matches = sorted(find_prefix_matches(catalog, prefix), key=lambda term: (-counts[term], term))
    
    return prefix_matches[:k]

if __name__ == "__main__":
    CATALOG = {"cat bed", "cat food", "cat toy", "dog food"}
    COUNTS = {
        "cat bed": 2,
        "cat food": 5,
        "cat toy": 5,
        "dog food": 20,
    }
    print(suggest_popular(
        catalog=CATALOG,
        counts=COUNTS,
        prefix="dog",
        k=2,
    ))