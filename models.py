
import torch
import torch.nn as nn


class BandNameGenerator(nn.Module):
    def __init__(self, alphabet_size, hidden_size=256):
        super(BandNameGenerator, self).__init__()
        self.gru = nn.GRU(alphabet_size, hidden_size)
        self.linear = nn.Linear(hidden_size, alphabet_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, h=None):
        h = torch.zeros(1, 1, self.gru.hidden_size) if h is None else h
        x, h = self.gru(x, h)
        x = self.linear(x)
        y = self.softmax(x)
        return y, h
