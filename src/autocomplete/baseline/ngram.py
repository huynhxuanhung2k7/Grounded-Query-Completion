import math
from collections import Counter
from collections.abc import Iterable

from autocomplete.decoding import greedy_complete

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
        """using the shared decoding algorithm."""
        if not self.counts:
            raise ValueError("Call fit() before generating completions")
        return greedy_complete(raw_prefix, self.next_log_probs)


if __name__ == "__main__":
    model = NgramModel(order=3, alpha=0.1)
    model.fit(["cat", "cat", "car"])
    print("Terms: ['cat', 'cat', 'car']; at most two context characters")
    print("Next-character counts after 'ca':", dict(model.counts["ca"]))
    for context in ("ca", "za"):
        probabilities = {
            token: round(math.exp(score), 3)
            for token, score in model.next_log_probs(context).items()
        }
        print(f"P(next | {context!r}):", probabilities)
    print("End counts after 'at':", dict(model.counts["at"]))
    print("Greedy completion:", model.greedy_complete("ca"))
