"""Shared file names and the PyTorch model."""
from pathlib import Path

import torch
import torch.nn as nn

BASE = Path(__file__).resolve().parent
CORPUS_FILE = BASE / "corpus.txt"
MODEL_FILE = BASE / "model.pt"
HISTORY_FILE = BASE / "history.json"
IMAGES_DIR = BASE / "images"
SAMPLES_FILE = BASE / "samples.md"


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CharLSTM(nn.Module):
    """Reads one character at a time and predicts the next one.

    characters -> embedding -> LSTM layers (they carry a memory) -> one score per possible next character
    """

    def __init__(self, vocab_size, embed_size=64, hidden_size=256, num_layers=2, dropout=0.3):
        super().__init__()
        self.config = {"vocab_size": vocab_size, "embed_size": embed_size,
                       "hidden_size": hidden_size, "num_layers": num_layers, "dropout": dropout}
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, state=None):
        # x: (batch, time) of character ids -> logits: (batch, time, vocab)
        h, state = self.lstm(self.embed(x), state)
        return self.out(self.drop(h)), state


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
