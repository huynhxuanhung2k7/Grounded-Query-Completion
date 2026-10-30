import unicodedata

NORMALIZATION_VERSION = "nfkc-casefold-whitespace-v1"


def _normalized_nfkc(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError(f"text must be a str, got {type(text).__name__}")

    return unicodedata.normalize("NFKC", text).casefold()


def normalize_term(text: str) -> str:
    """Normalize case, Unicode and spacing in a completed query.

    Args:
        text: Raw completed query.
    Returns:
        Normalized text without outer spaces.
    """
    normalized = _normalized_nfkc(text)

    return " ".join(normalized.split())


def normalize_prefix(text: str) -> str:
    """Normalize typed text while preserving a trailing word boundary.

    Args:
        text: Raw typed prefix.
    Returns:
        Normalized prefix, possibly ending with one space.
    """
    normalized = _normalized_nfkc(text)

    had_trailing_whitespace = bool(normalized) and normalized[-1].isspace()

    collapsed = " ".join(normalized.split())

    trailing_space = " " if collapsed and had_trailing_whitespace else ""

    return collapsed + trailing_space


if __name__ == "__main__":
    raw = "  ＣＡＴ   Food \t"
    print("Raw input:", repr(raw))
    print("Completed term:", repr(normalize_term(raw)))
    print("Typed prefix:", repr(normalize_prefix(raw)))

    print("Whitespace-only prefix:", repr(normalize_prefix("   ")))
