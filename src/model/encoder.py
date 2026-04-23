import torch
from torch.functional import norm
import torch.nn as nn

from src.model.feed_forward import PositionwiseFeedForward
from src.model.layer_norm import LayerNorm
from src.model.multi_head_attention import MultiheadAttention


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, dropout=0.1):
        super().__init__()

        self.self_attn = MultiheadAttention(embed_dim, num_heads)
        self.layer_norm1 = LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)

        self.ffnet = PositionwiseFeedForward(embed_dim, hidden_dim)
        self.layer_norm2 = LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out, _ = self.self_attn(x, x, x, mask)
        x = self.layer_norm1(x + self.dropout1(attn_out))

        ffnet_out = self.ffnet(x)
        x = self.layer_norm2(x + self.dropout2(ffnet_out))
