from collections.abc import Iterable
from collections import Counter

EOS = "<EOS>"

class NgramModel:
    def __init__(self, order: int = 4, alpha: float = 0.1):
        self.order = order
        self.alpha = alpha
        self.counts : dict[str, Counter[str]] = {}
        self.alphabet : set[str] = {EOS}

    def fit(self, normalized_terms: Iterable[str]) -> None:
        ...

    def next_log_probs(self, context: str) -> dict[str, float]:
        ...

    def greedy_complete(self, raw_prefix: str) -> dict[str, object]:
        ...