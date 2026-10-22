import subprocess
import sys

import pytest

from autocomplete.model.vocabulary import (
    EOS,
    EOS_ID,
    PAD,
    PAD_ID,
    UNK,
    UNK_ID,
    CharVocabulary,
    main,
)


def test_special_ids_and_sorted_character_ids() -> None:
    vocab = CharVocabulary(["cat", "car"])
    assert vocab.id_to_token == [PAD, UNK, EOS, "a", "c", "r", "t"]
    assert vocab.token_to_id == {
        PAD: PAD_ID,
        UNK: UNK_ID,
        EOS: EOS_ID,
        "a": 3,
        "c": 4,
        "r": 5,
        "t": 6,
    }
    assert (PAD_ID, UNK_ID, EOS_ID) == (0, 1, 2)
    assert len(vocab) == 7


def test_order_and_repeated_terms_do_not_change_ids() -> None:
    first = CharVocabulary(["cat", "car"])
    second = CharVocabulary(iter(["car", "cat", "cat", ""]))
    assert first.id_to_token == second.id_to_token
    assert first.token_to_id == second.token_to_id


def test_encode_decode_known_text_and_empty_text() -> None:
    vocab = CharVocabulary(["cat", "car"])
    assert vocab.encode("cat") == [4, 3, 6]
    assert vocab.decode(iter([4, 3, 6])) == "cat"
    assert vocab.encode("") == []
    assert vocab.decode([]) == ""


def test_unknown_input_does_not_expand_vocabulary() -> None:
    vocab = CharVocabulary(["cat", "car"])
    before = vocab.token_to_id.copy()
    assert vocab.encode("cap") == [4, 3, UNK_ID]
    assert vocab.token_to_id == before
    assert "p" not in vocab.id_to_token


def test_encoding_preserves_unicode_case_and_trailing_space() -> None:
    text = "Café 猫 "
    vocab = CharVocabulary([text])
    assert vocab.decode(vocab.encode(text)) == text
    assert vocab.encode("c") == [UNK_ID]  # No implicit case-folding.


def test_encode_never_automatically_inserts_padding_or_eos() -> None:
    vocab = CharVocabulary(["cat"])
    ids = vocab.encode("cat")
    assert len(ids) == 3
    assert PAD_ID not in ids
    assert EOS_ID not in ids


@pytest.mark.parametrize("terms", [[], [""], ["", ""]])
def test_empty_training_vocabulary_is_rejected(terms: list[str]) -> None:
    with pytest.raises(ValueError, match="nonempty term"):
        CharVocabulary(terms)


@pytest.mark.parametrize("value", [None, 3, ["cat"]])
def test_non_string_training_term_is_rejected(value: object) -> None:
    with pytest.raises(TypeError, match="strings"):
        CharVocabulary([value])


@pytest.mark.parametrize("value", [None, 3, ["c"]])
def test_non_string_encoding_input_is_rejected(value: object) -> None:
    with pytest.raises(TypeError, match="string"):
        CharVocabulary(["cat"]).encode(value)


@pytest.mark.parametrize("token_id", [PAD_ID, UNK_ID, EOS_ID])
def test_decode_rejects_special_tokens(token_id: int) -> None:
    with pytest.raises(ValueError, match="Special tokens"):
        CharVocabulary(["cat"]).decode([token_id])


@pytest.mark.parametrize("token_id", [-1, 6, 999])
def test_decode_rejects_out_of_range_ids(token_id: int) -> None:
    with pytest.raises(ValueError, match="outside vocabulary"):
        CharVocabulary(["cat"]).decode([token_id])


@pytest.mark.parametrize("token_id", [True, False, 3.0, "3"])
def test_decode_rejects_non_integer_ids(token_id: object) -> None:
    with pytest.raises(TypeError, match="integers"):
        CharVocabulary(["cat"]).decode([token_id])


@pytest.mark.parametrize("text", ["a", "cat food", "c++ / c#", "猫🐈", "<EOS>"])
def test_round_trip_for_character_edge_cases(text: str) -> None:
    vocab = CharVocabulary([text])
    encoded = vocab.encode(text)
    assert len(encoded) == len(text)
    assert all(token_id > EOS_ID for token_id in encoded)
    assert vocab.decode(encoded) == text


def test_all_unknown_characters_map_individually_to_unk() -> None:
    vocab = CharVocabulary(["cat"])
    assert vocab.encode("xyz") == [UNK_ID, UNK_ID, UNK_ID]
    assert len(vocab) == 6


def test_encode_returns_independent_lists() -> None:
    vocab = CharVocabulary(["cat"])
    original = vocab.encode("cat")
    modified = vocab.encode("cat")
    modified.clear()
    assert vocab.encode("cat") == original


def test_demo_explains_round_trip_and_unknown_character(
    capsys: pytest.CaptureFixture[str],
) -> None:
    main()
    output = capsys.readouterr().out
    assert "Known text: 'cat' -> [4, 3, 6] -> 'cat'" in output
    assert "Unknown text: 'cap' -> [4, 3, 1] (p becomes UNK)" in output
    assert "Expected decode rejection:" in output
    assert "Special IDs: PAD=0, UNK=1, EOS=2" in output
    assert "Encoding adds neither padding nor EOS" in output


def test_demo_module_entrypoint() -> None:
    completed = subprocess.run(
        [sys.executable, "-B", "-m", "autocomplete.model.vocabulary"],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    assert "Vocabulary (7 tokens):" in completed.stdout
    assert "Expected decode rejection:" in completed.stdout
    assert completed.stderr == ""
