import pytest

from autocomplete.model.vocabulary import EOS, PAD, UNK, UNK_ID, CharVocabulary


def test_character_ids_are_sorted_and_round_trip():
    vocabulary = CharVocabulary(["cat", "car"])

    assert vocabulary.id_to_token == [PAD, UNK, EOS, "a", "c", "r", "t"]
    assert vocabulary.encode("cat") == [4, 3, 6]
    assert vocabulary.decode([4, 3, 6]) == "cat"


def test_unknown_characters_do_not_expand_the_vocabulary():
    vocabulary = CharVocabulary(["cat"])
    before = vocabulary.id_to_token.copy()

    assert vocabulary.encode("z") == [UNK_ID]
    assert vocabulary.id_to_token == before
    assert vocabulary.encode("") == []


def test_restoration_preserves_saved_character_order():
    vocabulary = CharVocabulary.from_tokens([PAD, UNK, EOS, "t", "a", "c"])

    assert vocabulary.encode("cat") == [5, 4, 3]


def test_special_tokens_cannot_be_decoded_as_query_text():
    vocabulary = CharVocabulary(["cat"])

    with pytest.raises(ValueError, match="Special tokens"):
        vocabulary.decode([0])


def test_empty_training_vocabulary_is_rejected():
    with pytest.raises(ValueError, match="nonempty"):
        CharVocabulary([""])
