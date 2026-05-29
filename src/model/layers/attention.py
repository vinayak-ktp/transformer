import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V, mask=None):
        # Q, K, V: (B, heads, seq_len, d_k)
        # mask: (B, 1, seq_len, seq_len)

        d_k = Q.shape[-1]
        scores = torch.matmul(Q, K.transpose(-2, -1))   # (B, H, S, S)
        scores = scores / math.sqrt(d_k)                # (B, H, S, S)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = F.softmax(scores, dim=-1)   # (B, H, S, S)
        out = torch.matmul(attention_weights, V)        # (B, H, S, D)

        return out, attention_weights
