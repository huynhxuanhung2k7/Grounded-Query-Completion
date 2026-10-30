from autocomplete.contract import (
    MAX_ADDED_LENGTH,
    MAX_INPUT_LENGTH,
    is_compatible_completion,
    prepare_prefix,
)


def test_prepare_prefix_normalizes_before_checking_length():
    assert prepare_prefix(" " * 300 + "CAT ") == ("cat ", None)


def test_empty_prefix_is_rejected():
    assert prepare_prefix(" \t") == ("", "empty_prefix")


def test_input_length_boundary():
    assert prepare_prefix("a" * MAX_INPUT_LENGTH)[1] is None
    assert prepare_prefix("a" * (MAX_INPUT_LENGTH + 1))[1] == "input_too_long"


def test_completions_preserve_the_prefix_and_extension_limit():
    assert is_compatible_completion("cat", "cat")
    assert is_compatible_completion("cat ", "cat food")
    assert not is_compatible_completion("cat ", "caterpillar")
    assert not is_compatible_completion("", "cat")

    assert is_compatible_completion("cat", "cat" + "a" * MAX_ADDED_LENGTH)
    assert not is_compatible_completion("cat", "cat" + "a" * (MAX_ADDED_LENGTH + 1))
