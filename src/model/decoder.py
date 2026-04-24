import torch
import torch.nn as nn

from src.model.embeddings import TokenEmbedding
from src.model.feed_forward import PositionwiseFeedForward
from src.model.layer_norm import LayerNorm
from src.model.multihead_attention import MultiheadAttention
from src.model.positional_encoding import PositionalEncoding


class TransformerDecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, dropout=0.1):
        super().__init__()

        # Masked-Attention
        self.masked_attn = MultiheadAttention(embed_dim, num_heads)
        self.layer_norm1 = LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)

        # Cross-Attention
        self.cross_attn = MultiheadAttention(embed_dim, num_heads)
        self.layer_norm2 = LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)

        # FeedForward Netword
        self.ffnet = PositionwiseFeedForward(embed_dim, hidden_dim, dropout)
        self.layer_norm3 = LayerNorm(embed_dim)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        tgt,        # target text
        memory,     # encoder output
        tgt_mask=None,
        memory_mask=None
    ):
        attn_out, _ = self.masked_attn(tgt, tgt, tgt, tgt_mask)
        tgt = self.layer_norm1(tgt + self.dropout1(attn_out))

        attn_out, _ = self.cross_attn(tgt, memory, memory, memory_mask)
        tgt = self.layer_norm2(tgt + self.dropout2(attn_out))

        ffnet_out = self.ffnet(tgt)
        tgt = self.layer_norm3(tgt + self.dropout3(ffnet_out))

        return tgt


class TransformerDecoder(nn.Module):
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
            TransformerDecoderLayer(embed_dim, num_heads, hidden_dim, dropout)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None
    ):
        tgt = self.token_embedding(tgt)
        tgt = self.positional_encoding(tgt)
        tgt = self.dropout(tgt)

        for layer in self.layers:
            tgt = layer(tgt, memory, tgt_mask, memory_mask)

        return tgt
