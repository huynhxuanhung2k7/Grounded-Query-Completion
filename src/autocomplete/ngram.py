import math
from collections import Counter
from collections.abc import Iterable

from autocomplete.contract import (
    MAX_ADDED_LENGTH,
    is_compatible_completion,
    prepare_prefix,
)
from autocomplete.normalization import normalize_term

EOS = "<EOS>"


class NgramModel:
    def __init__(self, order: int = 4, alpha: float = 0.1):
        if isinstance(order, bool) or not isinstance(order, int) or order < 1:
            raise ValueError("order must be a positive integer")

        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be finite and positive")

        self.order = order
        self.alpha = alpha
        self.counts: dict[str, Counter[str]] = {}
        self.alphabet: set[str] = {EOS}

    def fit(self, normalized_terms: Iterable[str]) -> None:
        """Learn counts from normalized terms, preserving repeated terms."""
        self.counts = {}
        self.alphabet = {EOS}
        term_count = 0

        for term in normalized_terms:
            if not term:
                continue

            term_count += 1
            self.alphabet.update(term)

            for position in range(len(term) + 1):
                target = term[position] if position < len(term) else EOS
                max_length = min(self.order - 1, position)

                # Include the empty context and every available suffix.
                for length in range(max_length + 1):
                    context = term[position - length : position]

                    if context not in self.counts:
                        self.counts[context] = Counter()

                    self.counts[context][target] += 1

        if term_count == 0:
            raise ValueError("Training requires at least one nonempty term")

    def next_log_probs(self, context: str) -> dict[str, float]:
        """Return smoothed log probabilities using the longest seen suffix."""
        if not self.counts:
            raise ValueError("Call fit() before requesting predictions")

        selected_counts = self.counts[""]

        max_length = min(self.order - 1, len(context))

        for length in range(max_length, 0, -1):
            candidate = context[-length:]

            if candidate in self.counts:
                selected_counts = self.counts[candidate]
                break

        total = sum(selected_counts.values())
        vocabulary_size = len(self.alphabet)
        denominator = total + self.alpha * vocabulary_size

        return {
            token: math.log((selected_counts[token] + self.alpha) / denominator)
            for token in sorted(self.alphabet)
        }

    def greedy_complete(self, raw_prefix: str) -> dict[str, object]:
        """Extend a prefix until EOS, or return an explicit empty outcome."""
        if not self.counts:
            raise ValueError("Call fit() before generating completions")

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
            log_probs = self.next_log_probs(text)

            token = min(
                log_probs, key=lambda candidate: (-log_probs[candidate], candidate)
            )

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
                    "log_score": log_score + log_probs[EOS],
                    "stop_reason": "eos",
                }

            if added_length == MAX_ADDED_LENGTH:
                return {
                    "completion": None,
                    "log_score": None,
                    "stop_reason": "length_limit",
                }

            text += token
            log_score += log_probs[token]

        raise RuntimeError("Unreachable decoder state")


if __name__ == "__main__":
    training_terms = (
        ["cat food"] * 6
        + ["cat treats"] * 3
        + ["cat litter"] * 2
        + ["dog food"] * 5
        + ["dog treats"] * 2
        + ["wireless headphones"] * 5
        + ["wireless mouse"] * 3
        + ["wireless keyboard"] * 2
        + ["usb cable"] * 4
        + ["usb charger"] * 2
    )

    model = NgramModel()
    model.fit(training_terms)

    contexts = [
        "cat ",  # Competing continuations: food, treats, litter
        "dog ",  # Competing continuations: food, treats
        "wireless ",  # Shared beginning, different products
        "usb c",  # Cable versus charger
        "xxcat ",  # Same final context as "cat "
        "qqq",  # Unseen suffixes: fall back to unigram
        "cat food",  # EOS should be likely
    ]

    from pprint import pprint

    pprint(dict(model.counts))
    print(model.greedy_complete("usb"))
