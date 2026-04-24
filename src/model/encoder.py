import torch
import torch.nn as nn

from src.model.embeddings import TokenEmbedding
from src.model.feed_forward import PositionwiseFeedForward
from src.model.layer_norm import LayerNorm
from src.model.multihead_attention import MultiheadAttention
from src.model.positional_encoding import PositionalEncoding


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, dropout=0.1):
        super().__init__()

        # Self-Attention
        self.self_attn = MultiheadAttention(embed_dim, num_heads)
        self.layer_norm1 = LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)

        # FeedForward Network
        self.ffnet = PositionwiseFeedForward(embed_dim, hidden_dim)
        self.layer_norm2 = LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, mask=None):
        attn_out, _ = self.self_attn(src, src, src, mask)
        src = self.layer_norm1(src + self.dropout1(attn_out))

        ffnet_out = self.ffnet(src)
        src = self.layer_norm2(src + self.dropout2(ffnet_out))


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        num_layers,
        vocab_size,
        embed_dim,
        num_heads,
        hidden_dim,
        dropout=0.1,
        max_len=5000
    ):
        super().__init__()

        self.token_embedding = TokenEmbedding(vocab_size, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim, max_len)

        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, hidden_dim, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, src, mask=None):
        # src: (B, S)
        src = self.token_embedding(src)         # (B, S, D)
        src = self.positional_encoding(src)     # (B, S, D)
        src = self.dropout(src)

        for layer in self.layers:
            src = layer(src, mask)

        return src    # (B, S, D)
