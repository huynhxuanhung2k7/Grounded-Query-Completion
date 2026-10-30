"""Predict, compare with the recorded answer, and average the scores."""

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict

from autocomplete.baseline.baselines import suggest_popular
from autocomplete.contract import is_compatible_completion, prepare_prefix
from autocomplete.data import DevelopmentCase
from autocomplete.data.normalization import normalize_term
from autocomplete.decoding import exact_filter


def score_case(suggestions: list[str], normalized_truth: str) -> tuple[bool, float]:
    """Score an exact answer match within the first ten suggestions.

    Args:
        suggestions: Ranked normalized queries.
        normalized_truth: Recorded normalized answer.
    Returns:
        Hit flag and reciprocal rank; a miss gives zero.
    """
    top_ten = suggestions[:10]

    if normalized_truth not in top_ten:
        return False, 0

    rank = top_ten.index(normalized_truth) + 1
    reciprocal_rank = 1.0 / rank

    return True, reciprocal_rank


def summarize_scores(
    scores: list[tuple[bool, float]],
) -> dict[str, int | float | None]:
    """Average request scores, including misses.

    Args:
        scores: Hit and reciprocal-rank pairs.
    Returns:
        Request count, Success@10 and MRR@10; empty rates are None.
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


def _check_suggestions(suggestions: object) -> None:
    if not isinstance(suggestions, list) or any(
        not isinstance(term, str) for term in suggestions
    ):
        raise TypeError("Predictions must contain a list of suggestion strings")

    if len(suggestions) > 10:
        raise ValueError("At most 10 displayed suggestions are allowed")

    if len(suggestions) != len(set(suggestions)):
        raise ValueError("Suggestions must be unique")

    if any(not term or normalize_term(term) != term for term in suggestions):
        raise ValueError("Suggestions must be nonempty normalized full terms")


def completion_prediction(completion: Mapping[str, object]) -> dict[str, object]:
    """Adapt one completion to the shared evaluation format.

    Args:
        completion: Decoder result.
    Returns:
        Suggestion list, score and stop reason.
    """
    term = completion["completion"]
    score = completion["log_score"]

    if score is not None and not math.isfinite(score):
        raise ValueError("Completion score must be finite")

    return {
        "suggestions": [] if term is None else [term],
        "log_score": score,
        "stop_reason": completion["stop_reason"],
    }


def beam_prediction(
    pool: Mapping[str, object], *, k: int = 10, catalog: set[str] | None = None
) -> dict[str, object]:
    """Take top candidates after optional catalog filtering.

    Args:
        pool: Ranked beam result.
        k: Maximum displayed suggestions, up to ten.
        catalog: Optional historical membership filter.
    Returns:
        Suggestions, original pool and stop reason.
    """
    if type(k) is not int or not 1 <= k <= 10:
        raise ValueError("Displayed k must be an integer between 1 and 10")

    candidates = pool["candidates"]
    filtered_candidates = (
        exact_filter(candidates, catalog) if catalog is not None else candidates
    )
    terms = [candidate["completion"] for candidate in filtered_candidates]

    reason = pool["stop_reason"]

    if catalog is not None and candidates and not filtered_candidates:
        reason = "no_catalog_candidates"

    return {
        "suggestions": terms[:k],
        "candidates": candidates,
        "stop_reason": reason,
    }


def prediction_summary(results: list[dict]) -> dict[str, float | None]:
    """Count how often a request receives at least one suggestion.

    Args:
        results: Per-request prediction records.
    Returns:
        Response coverage; None when there are no requests.
    """
    count = len(results)
    answered = sum(bool(row["suggestions"]) for row in results)
    coverage = answered / count if count else None

    return {"response_coverage": coverage}


def evaluate_cases(
    cases: Sequence[DevelopmentCase],
    predict: Callable[[str], Mapping[str, object]],
    *,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, object]:
    """Predict from prefixes, then compare with the recorded answers.

    Args:
        cases: Evaluation requests.
        predict: Function receiving only the raw prefix.
        progress: Optional completed/total callback.
    Returns:
        Per-case results, ranking metrics and response coverage.
    """
    results = []
    scores = []

    for done, case in enumerate(cases, start=1):
        prediction = dict(predict(case.raw_prefix))

        suggestions = prediction["suggestions"]
        _check_suggestions(suggestions)

        prefix, rejection = prepare_prefix(case.raw_prefix)

        if rejection is not None and suggestions:
            raise ValueError("Rejected prefixes must not produce suggestions")

        if any(not is_compatible_completion(prefix, term) for term in suggestions):
            raise ValueError("Prediction violates the shared prefix/length contract")

        hit, rr = score_case(suggestions, case.normalized_final_search_term)

        results.append(
            {
                **prediction,
                **asdict(case),
                "hit_at_10": hit,
                "reciprocal_rank_at_10": rr,
            }
        )
        scores.append((hit, rr))

        if progress is not None:
            progress(done, len(cases))

    return {
        "results": results,
        "summary": summarize_scores(scores),
        "prediction_summary": prediction_summary(results),
    }


def evaluate_popularity(
    catalog: set[str],
    counts: dict[str, int],
    cases: list[DevelopmentCase],
) -> dict[str, object]:
    """Evaluate historical popularity through the shared scoring loop.

    Args:
        catalog: Historical queries.
        counts: Interaction counts.
        cases: Evaluation requests.
    Returns:
        Per-case results and ranking metrics.
    """

    def predict(raw_prefix: str) -> dict[str, object]:
        normalized_prefix, stop_reason = prepare_prefix(raw_prefix)

        if stop_reason is not None:
            suggestions = []
        else:
            suggestions = suggest_popular(catalog, counts, normalized_prefix, k=10)

            if not suggestions:
                stop_reason = "no_candidates"

        return {"suggestions": suggestions, "stop_reason": stop_reason}

    evaluation = evaluate_cases(cases, predict)

    return {key: evaluation[key] for key in ("results", "summary")}


if __name__ == "__main__":
    suggestions = ["cat food", "cat toy"]
    print("Ranked suggestions:", suggestions)

    hit = score_case(suggestions, "cat toy")
    miss = score_case(suggestions, "cat bed")
    print("Answer 'cat toy' (hit, reciprocal rank):", hit)
    print("Answer 'cat bed' (hit, reciprocal rank):", miss)

    print("Summary:", summarize_scores([hit, miss]))
    print("No requests:", summarize_scores([]))
