"""Score normalized autocomplete predictions and retain auditable case results."""

from dataclasses import asdict

from autocomplete.baselines import suggest_popular
from autocomplete.data import DevelopmentCase
from autocomplete.contract import prepare_prefix


def score_case(suggestions: list[str], normalized_truth: str) -> tuple[bool, float]:
    """Score one request using Hit@10 and reciprocal rank at ten.

    Args:
        suggestions: Ranked, normalized full-query suggestions, best first.
        normalized_truth: The normalized final search recorded for the request.

    Returns:
        A pair containing whether the truth appears in the first ten suggestions
        and its reciprocal one-based rank. A miss or empty suggestion list returns
        (False, 0.0).

    Notes:
        Suggestions beyond position ten do not contribute to either score.
        The caller supplies unique, normalized predictions in ranking order.
        Average hit flags across requests for Success@10, and average reciprocal
        ranks for MRR@10. This function does not normalize or reorder inputs.
    """
    top_ten = suggestions[:10]
    hit = normalized_truth in top_ten

    reciprocal_rank = 1.0 / (suggestions.index(normalized_truth) + 1) if hit else 0

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


def evaluate_popularity(
    catalog: set[str],
    counts: dict[str, int],
    cases: list[DevelopmentCase],
) -> dict[str, object]:
    """Evaluate historical-popularity suggestions on fixed development cases.

    Args:
        catalog: Nonempty, normalized historical search terms.
        counts: Historical interaction counts for every catalog term.
        cases: Development cases containing normalized prefixes and recorded truths.

    Returns:
        A dictionary with:
            results: One row per case, preserving input order and all six case
                fields, plus suggestions, hit_at_10 and reciprocal_rank_at_10.
            summary: Request count, Success@10 and MRR@10 from summarize_scores.

        Every case is evaluated with k=10, including empty predictions and misses.
        Empty input produces no result rows and None for both summary metrics.

    Notes:
        Recorded truths are used only for scoring, never for generating suggestions.
    """
    result = []
    scores = []
    for case in cases:
        normalized_prefix, stop_reason = prepare_prefix(case.normalized_prefix)
        
        if stop_reason is not None:
            suggestions = []
        else:
            suggestions = suggest_popular(
                catalog, counts, normalized_prefix, k=10
            )
            if not suggestions:
                stop_reason = "no_candidates"
                

        hit, rr = score_case(suggestions, case.normalized_final_search_term)
    
        result.append(
            {
                **asdict(case),
                "suggestions": suggestions,
                "hit_at_10": hit,
                "reciprocal_rank_at_10": rr,
                "stop_reason": stop_reason
            }
        )
        scores.append((hit, rr))

    return {"results": result, "summary": summarize_scores(scores)}


if __name__ == "__main__":
    import json

    catalog = {"cat food", "cat toy"}
    counts = {"cat food": 2, "cat toy": 5}

    cases = [
        DevelopmentCase(
            query_id=1,
            prefix_index=0,
            raw_prefix="cat",
            normalized_prefix="cat",
            raw_final_search_term="cat food",
            normalized_final_search_term="cat food",
        ),
        DevelopmentCase(
            query_id=2,
            prefix_index=0,
            raw_prefix="phone",
            normalized_prefix="phone",
            raw_final_search_term="phone case",
            normalized_final_search_term="phone case",
        ),
    ]

    evaluation = evaluate_popularity(catalog, counts, cases)
    print(json.dumps(evaluation, indent=2))
