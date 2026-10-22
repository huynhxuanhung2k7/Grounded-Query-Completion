import pytest

from autocomplete.contract import (
    is_compatible_completion,
    prepare_prefix,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (" CAT ", ("cat ", None)),
        ("", ("", "empty_prefix")),
        (" \t\n", ("", "empty_prefix")),
        ("a" * 256, ("a" * 256, None)),
        ("a" * 257, ("a" * 257, "input_too_long")),
        # Raw length exceeds the limit, but normalized length does not.
        (" " * 300 + "cat", ("cat", None)),
        # Case folding expands ß into ss: 129 characters become 258.
        ("ß" * 129, ("ss" * 129, "input_too_long")),
    ],
)
def test_prepare_prefix(
    raw: str,
    expected: tuple[str, str | None],
) -> None:
    assert prepare_prefix(raw) == expected


@pytest.mark.parametrize(
    ("prefix", "term", "expected"),
    [
        ("cat", "cat food", True),
        ("cat", "cat", True),
        ("cat", "catalogue", True),
        ("cat ", "cat food", True),
        ("cat ", "cat", False),
        ("cat", "dog food", False),
        ("", "cat", False),
        ("", "", False),
        ("cat", "cat" + "a" * 64, True),
        ("cat", "cat" + "a" * 65, False),
    ],
)
def test_is_compatible_completion(
    prefix: str,
    term: str,
    expected: bool,
) -> None:
    assert is_compatible_completion(prefix, term) is expected