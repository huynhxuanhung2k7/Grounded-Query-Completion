import unicodedata

NORMALIZATION_VERSION = "nfkc-casefold-whitespace-v1"


def _normalized_nfkc(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError(f"text must be a str, got {type(text).__name__}")

    return unicodedata.normalize("NFKC", text).casefold()


def normalize_term(text: str) -> str:
    normalized = _normalized_nfkc(text)
    return " ".join(normalized.split())


def normalize_prefix(text: str) -> str:
    normalized = _normalized_nfkc(text)

    had_trailing_whitespace = bool(normalized) and normalized[-1].isspace()

    collapsed = " ".join(normalized.split())

    return collapsed + (" " if collapsed and had_trailing_whitespace else "") #check collapsed here to prevent returning " ", but ""


if __name__ == "__main__":
    raw = "  ＣＡＴ   Food \t"
    print("Raw input:", repr(raw))
    print("Completed term:", repr(normalize_term(raw)))
    print("Typed prefix:", repr(normalize_prefix(raw)))

    print("Whitespace-only prefix:", repr(normalize_prefix("   ")))
