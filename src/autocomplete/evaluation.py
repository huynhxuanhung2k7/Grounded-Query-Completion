def score_case(
    suggestions: list[str],
    normalized_truth: str
) ->tuple[bool, float]:
    """Score one request using Success@10 and Reciprocal Rank@10.

    Args:
        suggestions: Ranked, normalized full-query suggestions, best first.
        normalized_truth: The normalized final search recorded for the request.

    Returns:
        A pair containing whether the truth appears in the first ten suggestions
        and its reciprocal one-based rank. A miss or empty suggestion list returns
        (False, 0.0).

    Notes:
        Suggestions beyond position ten do not contribute to either score.
    """
    top_ten = suggestions[:10]
    hit = normalized_truth in top_ten
    
    reciprocal_rank = (
        1.0 / (suggestions.index(normalized_truth) + 1)
        if hit 
        else 0
    ) 
    
    return hit, reciprocal_rank