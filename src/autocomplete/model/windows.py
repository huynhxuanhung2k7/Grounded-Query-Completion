from bisect import bisect_right
from collections.abc import Iterable

import torch
from torch import Tensor
from torch.utils.data import Dataset

from autocomplete.model.vocabulary import (
    EOS_ID,
    PAD_ID,
    UNK_ID,
    CharVocabulary,
)


class NextCharacterDataset(Dataset):
    def __init__(
        self,
        normalized_terms: Iterable[str],
        vocabulary: CharVocabulary,
        context_length: int = 32,
        allow_unknown: bool = False,
    ):
        """Store encoded terms and cumulative example counts.

        Args:
            normalized_terms: Whole terms from one data partition.
            vocabulary: Training character IDs.
            context_length: Number of earlier characters per example.
            allow_unknown: Allow validation text outside the vocabulary.
        """
        if context_length < 1:
            raise ValueError("context_length must be positive")

        self.context_length = context_length
        self.encoded_terms: list[list[int]] = []
        self.ends: list[int] = []

        total = 0

        for term in normalized_terms:
            encoded = vocabulary.encode(term)

            if not encoded:
                continue

            if UNK_ID in encoded and not allow_unknown:
                raise ValueError("Training characters must exist in the vocabulary")

            self.encoded_terms.append(encoded)

            total += len(encoded) + 1
            self.ends.append(total)

        if not self.encoded_terms:
            raise ValueError("No nonempty training terms")

    def __len__(self) -> int:
        return self.ends[-1]

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        """Build one padded context and its next-character or EOS target.

        Args:
            index: Global example position across stored terms.
        Returns:
            Context tensor and scalar target tensor.
        """
        if not 0 <= index < len(self):
            raise IndexError("Window index outside dataset")

        term_index = bisect_right(self.ends, index)
        encoded_term = self.encoded_terms[term_index]

        start = self.ends[term_index - 1] if term_index else 0
        local_target_position = index - start

        context_start = max(0, local_target_position - self.context_length)
        context = encoded_term[context_start:local_target_position]

        padding = [PAD_ID] * (self.context_length - len(context))
        context = padding + context

        target = (
            encoded_term[local_target_position]
            if local_target_position < len(encoded_term)
            else EOS_ID
        )

        return (
            torch.tensor(context, dtype=torch.long),
            torch.tensor(target, dtype=torch.long),
        )


if __name__ == "__main__":
    vocabulary = CharVocabulary(["cat"])
    dataset = NextCharacterDataset(["cat"], vocabulary, context_length=2)
    print("Stored terms:", dataset.encoded_terms)
    print("Number of examples:", len(dataset))

    for context, target in dataset:
        context_tokens = [
            vocabulary.id_to_token[token_id] for token_id in context.tolist()
        ]
        target_token = vocabulary.id_to_token[target.item()]
        print(context_tokens, "->", target_token)
