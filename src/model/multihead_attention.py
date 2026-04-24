import torch.nn as nn

from src.model.attention import ScaledDotProductAttention


class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.attention = ScaledDotProductAttention()

    def forward(self, query, key, value, mask=None):
        B, q_len, D = query.shape
        _, k_len, _ = key.shape
        _, v_len, _ = value.shape

        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)

        Q = Q.view(B, self.num_heads, q_len, self.head_dim)
        K = K.view(B, self.num_heads, k_len, self.head_dim)
        V = V.view(B, self.num_heads, v_len, self.head_dim)

        out, attention_weights = self.attention(Q, K, V, mask)

        out = out.view(B, q_len, D)
        out = self.out_proj(out)

        return out, attention_weights
