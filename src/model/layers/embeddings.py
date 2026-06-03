import math

import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        # self.weight = nn.Parameter(torch.randn(vocab_size, embed_dim))

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.weight = self.embedding.weight

    def forward(self, x):
        # x: (B, S)
        # out = self.weight[x]
        out = self.embedding(x)
        return out * math.sqrt(self.embed_dim)
