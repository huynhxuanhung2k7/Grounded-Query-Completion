import torch
from torch import nn
from torch.utils.data import DataLoader

from autocomplete.model.model import NeuralCharModel
from autocomplete.model.vocabulary import CharVocabulary
from autocomplete.model.windows import NextCharacterDataset

def main() -> None:
    