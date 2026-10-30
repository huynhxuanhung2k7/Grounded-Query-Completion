import pytest

from autocomplete.data.normalization import normalize_prefix, normalize_term


def test_term_normalizes_unicode_case_and_spaces():
    assert normalize_term("  ＣＡＴ   Food\t") == "cat food"
    assert normalize_term("Straße") == "strasse"


def test_prefix_preserves_a_trailing_word_boundary():
    assert normalize_prefix("  CAT   Food\t") == "cat food "
    assert normalize_prefix("CAT") == "cat"


def test_empty_text_stays_empty():
    assert normalize_term("") == ""
    assert normalize_prefix(" \t\n") == ""


def test_normalization_is_repeatable_and_preserves_punctuation():
    text = normalize_term("  C++ / C#  ")

    assert text == "c++ / c#"
    assert normalize_term(text) == text


def test_non_string_input_is_rejected():
    with pytest.raises(TypeError):
        normalize_term(None)
