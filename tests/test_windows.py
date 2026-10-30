import pytest
import torch

from autocomplete.model.vocabulary import EOS_ID, PAD_ID, UNK_ID, CharVocabulary
from autocomplete.model.windows import NextCharacterDataset


def test_windows_have_only_earlier_characters_and_an_eos_target():
    vocabulary = CharVocabulary(["cat"])
    dataset = NextCharacterDataset(["cat"], vocabulary, context_length=3)
    c, a, t = vocabulary.encode("cat")

    examples = [(context.tolist(), target.item()) for context, target in dataset]

    assert examples == [
        ([PAD_ID, PAD_ID, PAD_ID], c),
        ([PAD_ID, PAD_ID, c], a),
        ([PAD_ID, c, a], t),
        ([c, a, t], EOS_ID),
    ]
    assert dataset[0][0].dtype == torch.long


def test_long_contexts_keep_only_the_latest_characters():
    vocabulary = CharVocabulary(["cat"])
    dataset = NextCharacterDataset(["cat"], vocabulary, context_length=2)

    context, target = dataset[3]

    assert context.tolist() == vocabulary.encode("at")
    assert target.item() == EOS_ID


def test_indexing_moves_to_the_next_term_without_crossing_contexts():
    vocabulary = CharVocabulary(["cat", "car"])
    dataset = NextCharacterDataset(["cat", "car"], vocabulary, context_length=2)

    context, target = dataset[4]

    assert len(dataset) == 8
    assert context.tolist() == [PAD_ID, PAD_ID]
    assert target.item() == vocabulary.encode("c")[0]

    with pytest.raises(IndexError):
        dataset[len(dataset)]


def test_unknown_characters_are_allowed_only_when_requested():
    vocabulary = CharVocabulary(["cat"])

    with pytest.raises(ValueError, match="vocabulary"):
        NextCharacterDataset(["cap"], vocabulary)

    validation = NextCharacterDataset(["cap"], vocabulary, allow_unknown=True)

    assert validation[2][1].item() == UNK_ID


def test_empty_terms_or_nonpositive_context_are_rejected():
    vocabulary = CharVocabulary(["cat"])

    with pytest.raises(ValueError, match="nonempty"):
        NextCharacterDataset([""], vocabulary)

    with pytest.raises(ValueError, match="positive"):
        NextCharacterDataset(["cat"], vocabulary, context_length=0)
