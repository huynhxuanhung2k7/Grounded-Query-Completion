import torch
from torch.utils.data import DataLoader

from autocomplete.model.model import NeuralCharModel
from autocomplete.model.validation import evaluate
from autocomplete.model.vocabulary import CharVocabulary
from autocomplete.model.windows import NextCharacterDataset

vocabulary = CharVocabulary(["cat"])
model = NeuralCharModel(len(vocabulary))

validation_data = NextCharacterDataset(
    ["cap"],
    vocabulary,
    allow_unknown=True,
)
validation_loader = DataLoader(
    validation_data,
    batch_size=2,
    shuffle=False,
)

before = {
    name: parameter.detach().clone() for name, parameter in model.named_parameters()
}

report = evaluate(model, validation_loader)
print(report)

assert report["total_targets"] == 4
assert report["scored_targets"] == 3
assert report["unknown_targets"] == 1
assert report["unknown_target_fraction"] == 0.25

for name, parameter in model.named_parameters():
    assert torch.equal(parameter.detach(), before[name])
