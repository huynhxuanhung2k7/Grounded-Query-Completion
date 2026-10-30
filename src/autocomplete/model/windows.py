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
                raise ValueError(
                    "Training characters must exist in the vocabulary"
                )

            self.encoded_terms.append(encoded)

            # One target per character, plus EOS.
            total += len(encoded) + 1
            self.ends.append(total)

        if not self.encoded_terms:
            raise ValueError("No nonempty training terms")

    def __len__(self) -> int:
        return self.ends[-1]

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        if not 0 <= index < len(self):
            raise IndexError("Window index outside dataset")

        # Locate the interaction containing this example.
        term_index = bisect_right(self.ends, index)
        encoded_chosen_word = self.encoded_terms[term_index]

        start = self.ends[term_index - 1] if term_index else 0
        local_target_position = index - start

        # Only characters BEFORE the target enter the context.
        context = encoded_chosen_word[
            max(0, local_target_position - self.context_length):local_target_position
        ]

        padding = [PAD_ID] * (self.context_length - len(context))
        context = padding + context

        target = (
            encoded_chosen_word[local_target_position]
            if local_target_position < len(encoded_chosen_word)
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
    print("Number of examples:", len(dataset), "(three characters plus EOS)")
    for context, target in dataset:
        context_tokens = [
            vocabulary.id_to_token[token_id]
            for token_id in context.tolist()
        ]
        target_token = vocabulary.id_to_token[target.item()]
        print(context_tokens, "->", target_token)
