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
    """Return (hit, 1/rank) for a top-ten match, otherwise (False, 0).

    Suggestions and truth must already use the same normalization.
    """
    top_ten = suggestions[:10]
    hit = normalized_truth in top_ten
    reciprocal_rank = 1.0 / (suggestions.index(normalized_truth) + 1) if hit else 0

    return hit, reciprocal_rank


def summarize_scores(
    scores: list[tuple[bool, float]],
) -> dict[str, int | float | None]:
    """Average all request scores, including misses. Empty input gives None."""

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
    """Adapt either model's single-completion decoder to the shared evaluator."""
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
    """Show the first k candidates, optionally keeping only catalog members."""
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
    """Coverage is the fraction of requests receiving at least one suggestion."""
    count = len(results)
    return {
        "response_coverage": sum(bool(row["suggestions"]) for row in results) / count
        if count
        else None,
    }


def evaluate_cases(
    cases: Sequence[DevelopmentCase],
    predict: Callable[[str], Mapping[str, object]],
    *,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, object]:
    """Predict from each prefix, score it, and keep misses in the averages."""
    results = []
    scores = []
    for done, case in enumerate(cases, start=1):
        # The predictor sees only the input
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
    """Run popularity through the same scoring loop, preserving its public format."""

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
    # Preserve the existing public return shape and saved popularity artifacts.
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
