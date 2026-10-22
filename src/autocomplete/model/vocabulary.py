"""Deterministic character IDs built only from supplied training terms.

Normalize text before passing it here. Encoding does not add padding or EOS;
those belong to context/target window construction.
"""

from collections.abc import Iterable

PAD = "<PAD>"
UNK = "<UNK>"
EOS = "<EOS>"

PAD_ID = 0
UNK_ID = 1
EOS_ID = 2


class CharVocabulary:
    def __init__(self, normalized_terms: Iterable[str]):
        characters: set[str] = set()
        for term in normalized_terms:
            if not isinstance(term, str):
                raise TypeError("Training terms must be strings")
            characters.update(term)

        if not characters:
            raise ValueError("Vocabulary requires at least one nonempty term")

        self.id_to_token = [PAD, UNK, EOS, *sorted(characters)]
        self.token_to_id = {
            token: token_id for token_id, token in enumerate(self.id_to_token)
        }

    def __len__(self) -> int:
        """Return the number of characters plus the three special tokens."""
        return len(self.id_to_token)

    def encode(self, text: str) -> list[int]:
        """Encode characters without changing text; unseen characters become UNK."""
        if not isinstance(text, str):
            raise TypeError("Text must be a string")
        return [self.token_to_id.get(character, UNK_ID) for character in text]

    def decode(self, ids: Iterable[int]) -> str:
        """Decode ordinary character IDs, rejecting special or invalid IDs.

        UNK cannot recover the original character. Callers must handle PAD/UNK/EOS
        explicitly rather than silently turning them into query text.
        """
        characters = []
        for token_id in ids:
            if isinstance(token_id, bool) or not isinstance(token_id, int):
                raise TypeError("Token IDs must be integers")
            if not 0 <= token_id < len(self):
                raise ValueError(f"Token ID outside vocabulary: {token_id}")
            if token_id in (PAD_ID, UNK_ID, EOS_ID):
                raise ValueError("Special tokens cannot be decoded as characters")
            characters.append(self.id_to_token[token_id])
        return "".join(characters)


def main() -> None:
    """Demonstrate stable IDs, a round trip and unknown-character handling."""
    training_terms = ["cat", "car", "cat"]
    vocab = CharVocabulary(training_terms)

    print(f"Training terms: {training_terms}")
    print(f"Vocabulary ({len(vocab)} tokens): {vocab.token_to_id}")

    known_text = "cat"
    known_ids = vocab.encode(known_text)
    print(f"Known text: {known_text!r} -> {known_ids} -> {vocab.decode(known_ids)!r}")

    unknown_text = "cap"
    unknown_ids = vocab.encode(unknown_text)
    print(f"Unknown text: {unknown_text!r} -> {unknown_ids} (p becomes UNK)")
    try:
        vocab.decode(unknown_ids)
    except ValueError as error:
        print(f"Expected decode rejection: {error}")

    print(f"Special IDs: PAD={PAD_ID}, UNK={UNK_ID}, EOS={EOS_ID}")
    print("Encoding adds neither padding nor EOS; windows will handle those.")


if __name__ == "__main__":
    main()
