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

def summarize_scores(
    scores: list[tuple[bool, float]],
) -> dict[str, int | float | None]:
    """Summarize hit and reciprocal-rank scores across requests.

    Args:
        scores: One (hit, reciprocal_rank) tuple per request.

    Returns:
        A dictionary with request_count, success_at_10 and mrr_at_10.
        Misses contribute zero to both metrics and remain in the denominator.
        With no requests, both metrics are None.
    """

    request_count = len(scores)
    hit_count = sum(hit for hit, _ in scores)
    reciprocal_rank_sum = sum(rr for _, rr in scores)

    success_at_10 = hit_count / request_count if request_count else None
    mrr_at_10 = reciprocal_rank_sum / request_count if request_count else None

    return {
        "request_count": request_count,
        "success_at_10": success_at_10,
        "mrr_at_10": mrr_at_10,
    }
