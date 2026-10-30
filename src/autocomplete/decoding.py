import math
from collections.abc import Callable
from itertools import islice
from typing import TypedDict

import torch

from autocomplete.baseline.catalog import find_prefix_matches
from autocomplete.contract import (
    MAX_ADDED_LENGTH,
    is_compatible_completion,
    prepare_prefix,
)
from autocomplete.data.normalization import normalize_term
from autocomplete.model.model import NeuralCharModel
from autocomplete.model.vocabulary import EOS, PAD_ID, UNK_ID, CharVocabulary

NextLogProbs = Callable[[str], dict[str, float]]


class CompletionResult(TypedDict):
    completion: str | None
    log_score: float | None
    stop_reason: str


class NeuralPredictor:
    def __init__(
        self,
        model: NeuralCharModel,
        vocabulary: CharVocabulary,
    ) -> None:
        self.model = model
        self.vocabulary = vocabulary

    def next_log_probs(self, text: str) -> dict[str, float]:
        """Score the next character while excluding PAD and UNK outputs.

        Args:
            text: Prefix plus any generated characters.
        Returns:
            Allowed token to log probability.
        """
        row = self.next_log_probs_batch([text])[0].tolist()

        return {
            token: row[token_id]
            for token_id, token in enumerate(self.vocabulary.id_to_token)
            if token_id not in (PAD_ID, UNK_ID)
        }

    def next_log_probs_batch(self, texts: list[str]) -> torch.Tensor:
        """Pad or truncate contexts and score them together.

        Args:
            texts: Prefixes plus any generated characters.
        Returns:
            Log probabilities shaped batch by vocabulary size.
        """
        parameter = next(self.model.parameters())
        device = parameter.device

        if not texts:
            return torch.empty(
                (0, len(self.vocabulary)), device=device, dtype=parameter.dtype
            )

        rows = []

        for text in texts:
            ids = self.vocabulary.encode(text)[-self.model.context_length :]
            padding = [PAD_ID] * (self.model.context_length - len(ids))
            rows.append(padding + ids)

        batch = torch.tensor(rows, dtype=torch.long, device=device)

        self.model.eval()

        with torch.no_grad():
            return torch.log_softmax(self.model(batch), dim=-1)

    def score_catalog_terms(
        self,
        prefix: str,
        catalog: set[str],
        batch_size: int = 256,
    ) -> list[dict]:
        """Rank catalog matches by the suffix and EOS log probabilities.

        Args:
            prefix: Typed prefix.
            catalog: Normalized historical queries.
            batch_size: Maximum character windows scored together.
        Returns:
            Full queries ordered by score, then alphabetically.
        """
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")

        prefix, reason = prepare_prefix(prefix)

        if reason:
            return []

        token_ids = self.vocabulary.token_to_id
        terms = [
            term
            for term in find_prefix_matches(catalog, prefix)
            if is_compatible_completion(prefix, term)
            and all(char in token_ids for char in term[len(prefix) :])
        ]
        totals = [0.0] * len(terms)

        def windows():

            for owner, term in enumerate(terms):
                for position in range(len(prefix), len(term) + 1):
                    token = term[position] if position < len(term) else EOS
                    yield term[:position], token_ids[token], owner

        pending = windows()

        while True:
            batch = list(islice(pending, batch_size))

            if not batch:
                break

            log_probs = self.next_log_probs_batch([text for text, _, _ in batch])
            allowed = log_probs[:, 2:]

            if torch.isnan(allowed).any().item() or (allowed > 0).any().item():
                raise ValueError(
                    "Completion log probabilities must be nonpositive and not NaN"
                )

            targets = torch.tensor(
                [target for _, target, _ in batch],
                device=log_probs.device,
            )
            scores = log_probs.gather(1, targets[:, None]).squeeze(1).tolist()

            for (_, _, owner), score in zip(batch, scores, strict=True):
                totals[owner] += score

        ranked = [
            {"completion": term, "log_score": total}
            for term, total in zip(terms, totals, strict=True)
            if math.isfinite(total)
        ]

        return sorted(ranked, key=lambda row: (-row["log_score"], row["completion"]))


def greedy_complete(
    raw_prefix: str,
    next_log_probs: NextLogProbs,
) -> CompletionResult:
    """Extend a prefix using the best next token until EOS or the limit.

    Args:
        raw_prefix: Typed input.
        next_log_probs: Function scoring the next token from current text.
    Returns:
        Completion, suffix/EOS log score and stop reason.
    """
    prefix, stop_reason = prepare_prefix(raw_prefix)

    if stop_reason is not None:
        return {
            "completion": None,
            "log_score": None,
            "stop_reason": stop_reason,
        }

    text = prefix
    log_score = 0.0

    for added_length in range(MAX_ADDED_LENGTH + 1):
        scores = next_log_probs(text)

        if any(math.isnan(score) or score > 0 for score in scores.values()):
            raise ValueError(
                "Decoder log probabilities must be nonpositive and not NaN"
            )

        token = min(scores, key=lambda candidate: (-scores[candidate], candidate))

        if token == EOS:
            completion = normalize_term(text)

            if not is_compatible_completion(prefix, completion):
                return {
                    "completion": None,
                    "log_score": None,
                    "stop_reason": "no_candidates",
                }

            return {
                "completion": completion,
                "log_score": log_score + scores[EOS],
                "stop_reason": "eos",
            }

        if added_length == MAX_ADDED_LENGTH:
            return {
                "completion": None,
                "log_score": None,
                "stop_reason": "length_limit",
            }

        text += token
        log_score += scores[token]

    raise RuntimeError("Unreachable decoder state")


def beam_candidates(
    raw_prefix: str,
    next_log_probs: NextLogProbs,
    beam_width: int = 20,
    candidate_budget: int = 50,
) -> dict[str, object]:
    """Keep several partial paths and rank completed queries.

    Args:
        raw_prefix: Typed input.
        next_log_probs: Function scoring the next token.
        beam_width: Maximum live paths.
        candidate_budget: Maximum completed queries.
    Returns:
        Ordered candidate pool and search settings.
    """
    for name, value in (
        ("beam_width", beam_width),
        ("candidate_budget", candidate_budget),
    ):
        if type(value) is not int:
            raise TypeError(f"{name} must be an integer, excluding bool")

        if value <= 0:
            raise ValueError(f"{name} must be positive")

    prefix, reason = prepare_prefix(raw_prefix)

    if reason is not None:
        return {
            "candidates": [],
            "candidate_count": 0,
            "beam_width": beam_width,
            "candidate_budget": candidate_budget,
            "stop_reason": reason,
        }

    live_paths: list[tuple[str, float]] = [(prefix, 0.0)]
    finished: dict[str, float] = {}
    stop_reason = "exhausted"

    for added_length in range(MAX_ADDED_LENGTH + 1):
        next_live: list[tuple[str, float]] = []
        reached_length_limit = False

        for text, path_score in live_paths:
            scores = next_log_probs(text)

            for token, token_score in scores.items():
                if math.isnan(token_score) or token_score > 0:
                    raise ValueError(
                        "Decoder log probabilities must be nonpositive and not NaN"
                    )

                if token_score == -math.inf:
                    continue

                score = path_score + token_score

                if token == EOS:
                    completion = normalize_term(text)

                    if is_compatible_completion(prefix, completion):
                        previous = finished.get(completion, -math.inf)

                        if score > previous:
                            finished[completion] = score

                elif added_length < MAX_ADDED_LENGTH:
                    next_live.append((text + token, score))

                else:
                    reached_length_limit = True

        ranked_finished = sorted(
            finished.items(),
            key=lambda item: (-item[1], item[0]),
        )[:candidate_budget]

        finished = dict(ranked_finished)

        live_paths = sorted(
            next_live,
            key=lambda item: (-item[1], item[0]),
        )[:beam_width]

        if not live_paths:
            stop_reason = "length_limit" if reached_length_limit else "exhausted"
            break

        if len(ranked_finished) == candidate_budget:
            best_live_score = live_paths[0][1]
            worst_finished_score = ranked_finished[-1][1]

            if best_live_score < worst_finished_score:
                stop_reason = "score_bound"
                break

    candidates = [
        {"completion": text, "log_score": score} for text, score in finished.items()
    ]

    return {
        "candidates": candidates,
        "candidate_count": len(candidates),
        "beam_width": beam_width,
        "candidate_budget": candidate_budget,
        "stop_reason": stop_reason,
    }


def exact_filter(
    candidates: list[dict],
    catalog: set[str],
) -> list[dict]:
    """Keep catalog members without changing candidate order.

    Args:
        candidates: Ranked generated queries.
        catalog: Allowed historical queries.
    Returns:
        Candidates present in the catalog.
    """
    results = []

    for candidate in candidates:
        if candidate["completion"] in catalog:
            results.append(candidate)

    return results


if __name__ == "__main__":
    probabilities = {
        "ca": {"t": 0.6, "r": 0.4},
        "cat": {EOS: 0.5, "s": 0.5},
        "car": {EOS: 1.0},
        "cats": {EOS: 1.0},
    }

    def toy_next_log_probs(text):
        return {token: math.log(p) for token, p in probabilities[text].items()}

    print("Toy next-token probabilities:", probabilities)

    print("Greedy:", greedy_complete("ca", toy_next_log_probs))
    pool = beam_candidates("ca", toy_next_log_probs, beam_width=2, candidate_budget=3)
    print(
        "Beam (full-path probabilities):",
        [
            (row["completion"], round(math.exp(row["log_score"]), 3))
            for row in pool["candidates"]
        ],
    )

    filtered = exact_filter(pool["candidates"], {"cat", "cats"})
    print("Catalog {'cat', 'cats'} keeps:", [row["completion"] for row in filtered])
    print("Empty prefix:", greedy_complete("", toy_next_log_probs)["stop_reason"])
