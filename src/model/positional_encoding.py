import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super().__init__()

        self.pe = torch.zeros(max_len, embed_dim)    # (S, D)

        pos = torch.arange(0, max_len).unsqueeze(1)                         # (S, 1)
        div = 1 / (10000 ** (torch.arange(0, embed_dim, 2) / embed_dim))    # (D/2,)

        self.pe[:, 0::2] = torch.sin(pos * div)  # (S, D/2)
        self.pe[:, 1::2] = torch.cos(pos * div)  # (S, D/2)

        self.pe.unsqueeze(0)

        self.register_buffer("pe", self.pe)

    def forward(self, x):
        # x: (B, S, D)
        S = x.shape[1]
        return x + self.pe[:, :S, :]
