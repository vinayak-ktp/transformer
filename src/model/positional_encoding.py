import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, embed_dim)    # (S, D)

        pos = torch.arange(0, max_len).unsqueeze(1)  # (S, 1)

        # slow (???)
        # div = 1 / (10000 ** (torch.arange(0, embed_dim, 2) / embed_dim))    # (D/2)

        # fast, apparently
        div = torch.exp(
            (torch.arange(0, embed_dim, 2) / embed_dim) * -math.log(10000.0)
        )   # (D/2,)

        pe[:, 0::2] = torch.sin(pos * div)  # (S, D/2)
        pe[:, 1::2] = torch.cos(pos * div)  # (S, D/2)

        pe = pe.unsqueeze(0)    # (1, S, D)

        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (B, S, D)
        S = x.shape[1]
        return x + self.pe[:, :S, :]
