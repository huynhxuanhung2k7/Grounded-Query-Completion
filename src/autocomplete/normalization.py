import unicodedata

NORMALIZATION_VERSION = "nfkc-casefold-whitespace-v1"

def normalized_nfkc(text: str) -> str:
    if not isinstance(text, str):
            raise TypeError(
                f"text must be a str, got {type(text).__name__}"
            )   

    return unicodedata.normalize("NFKC", text).casefold()

def normalize_term(text: str) -> str:       
    normalized = normalized_nfkc(text)
    return " ".join(normalized.split())

def normalize_prefix(text: str) -> str:
    normalized = normalized_nfkc(text)
    
    had_trailing_whitespace = (
        bool(normalized) and normalized[-1].isspace()
    )

    collapsed = " ".join(normalized.split())
    if had_trailing_whitespace:
        normalized = normalized + " "
    
    return collapsed + (" " if had_trailing_whitespace and collapsed else "")