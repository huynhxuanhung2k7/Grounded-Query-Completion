MAX_INPUT_LENGTH = 256
MAX_ADDED_LENGTH = 64

from autocomplete.normalization import normalize_prefix

def prepare_prefix(raw_prefix: str) -> tuple[str, str | None]:
    """Normalize input and return an optional rejection reason."""
    normalized = normalize_prefix(raw_prefix)

    if not normalized:
        return normalized, "empty_prefix"

    if len(normalized) > MAX_INPUT_LENGTH:
        return normalized, "input_too_long"

    return normalized, None


def is_compatible_completion(
    normalized_prefix: str,
    normalized_term: str,
) -> bool:
    """Check a normalized completion against an accepted normalized prefix."""
    if not normalized_prefix:
        return False
    
    if not normalized_term.startswith(normalized_prefix):
        return False
    
    added_length = len(normalized_term) - len(normalized_prefix)
    return added_length <= MAX_ADDED_LENGTH