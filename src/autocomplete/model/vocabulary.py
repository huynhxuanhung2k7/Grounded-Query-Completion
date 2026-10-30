"""Map characters to IDs and restore saved vocabularies."""

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

    @classmethod
    def from_tokens(cls, tokens: list[str]) -> "CharVocabulary":
        """Restore the saved token order so IDs still match model weights.

        Args:
            tokens: Saved special tokens followed by characters.
        Returns:
            Restored vocabulary.
        """
        if not isinstance(tokens, list):
            raise TypeError("Saved tokens must be a list")

        if any(not isinstance(token, str) for token in tokens):
            raise TypeError("Every token must be a string")

        if tokens[:3] != [PAD, UNK, EOS]:
            raise ValueError("Expected PAD, UNK, EOS at IDs 0, 1, 2")

        if len(set(tokens)) != len(tokens):
            raise ValueError("Saved tokens must be unique")

        characters = tokens[3:]

        if not characters or any(len(token) != 1 for token in characters):
            raise ValueError("Expected nonempty single-character vocabulary")

        vocabulary = cls.__new__(cls)
        vocabulary.id_to_token = list(tokens)
        vocabulary.token_to_id = {
            token: token_id for token_id, token in enumerate(tokens)
        }

        return vocabulary

    def __len__(self) -> int:
        """Return the number of characters plus the three special tokens."""
        return len(self.id_to_token)

    def encode(self, text: str) -> list[int]:
        """Convert characters to IDs; unseen characters become UNK.

        Args:
            text: Text without added padding or EOS.
        Returns:
            One ID per character.
        """
        if not isinstance(text, str):
            raise TypeError("Text must be a string")

        return [self.token_to_id.get(character, UNK_ID) for character in text]

    def decode(self, ids: Iterable[int]) -> str:
        """Convert ordinary character IDs back to text.

        Args:
            ids: Character IDs, excluding PAD, UNK and EOS.
        Returns:
            Decoded text.
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


if __name__ == "__main__":
    vocabulary = CharVocabulary(["cat"])
    print("Token IDs:", vocabulary.token_to_id)
    encoded = vocabulary.encode("cat")
    print("Encoded:", encoded)
    print("Decoded:", vocabulary.decode(encoded))
    print("Unknown character 'z':", vocabulary.encode("z"))
