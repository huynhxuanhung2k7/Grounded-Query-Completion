import pytest

from autocomplete.normalization import (
    _normalized_nfkc,
    normalize_prefix,
    normalize_term,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("ＣＡＴ", "cat"),
        ("Straße", "strasse"),
        ("CAT\u00a0FOOD", "cat food"),
        ("  CAT  ", "  cat  "),
    ],
)
def test_normalized_nfkc_cases(raw: str, expected: str) -> None:
    assert _normalized_nfkc(raw) == expected


@pytest.mark.parametrize("value", [123, None, ["cat"]])
def test_normalized_nfkc_rejects_non_strings(value: object) -> None:
    with pytest.raises(TypeError):
        _normalized_nfkc(value)


@pytest.mark.parametrize(
    "raw",
    [
        "ＣＡＴ",
        "Straße",
        "  CAT  ",
    ],
)
def test_normalized_nfkc_is_idempotent(raw: str) -> None:
    once = _normalized_nfkc(raw)
    assert _normalized_nfkc(once) == once


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  CAT   Food  ", "cat food"),
        ("C++ / C# 50% 10.5-inch", "c++ / c# 50% 10.5-inch"),
        ("ＣＡＴ", "cat"),
        ("Straße", "strasse"),
        ("\tWireless\n Headphones\t", "wireless headphones"),
        ("   ", ""),
        ("", ""),
    ],
)
def test_normalize_term_cases(raw: str, expected: str) -> None:
    assert normalize_term(raw) == expected


@pytest.mark.parametrize("value", [123, None, ["cat"]])
def test_normalize_term_rejects_non_strings(value: object) -> None:
    with pytest.raises(TypeError):
        normalize_term(value)


@pytest.mark.parametrize(
    "raw",
    [
        "  CAT   Food  ",
        "ＣＡＴ",
        "Straße",
        "   ",
    ],
)
def test_normalize_term_is_idempotent(raw: str) -> None:
    once = normalize_term(raw)
    assert normalize_term(once) == once


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  Cat   Food", "cat food"),
        ("  Cat   Food  ", "cat food "),
        ("cat", "cat"),
        ("cat ", "cat "),
        ("\tCat\t", "cat "),
        ("   ", ""),
    ],
)
def test_normalize_prefix_cases(raw: str, expected: str) -> None:
    assert normalize_prefix(raw) == expected


@pytest.mark.parametrize("value", [123, None, ["cat"]])
def test_normalize_prefix_rejects_non_strings(value: object) -> None:
    with pytest.raises(TypeError):
        normalize_prefix(value)


@pytest.mark.parametrize(
    "raw",
    [
        "  Cat   Food",
        "  Cat   Food  ",
        "cat ",
        "   ",
    ],
)
def test_normalize_prefix_is_idempotent(raw: str) -> None:
    once = normalize_prefix(raw)
    assert normalize_prefix(once) == once
