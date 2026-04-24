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
        B, S, D = query.shape

        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)

        Q = Q.view(B, self.num_heads, S, self.head_dim)
        K = K.view(B, self.num_heads, S, self.head_dim)
        V = V.view(B, self.num_heads, S, self.head_dim)

        out, attention_weights = self.attention(Q, K, V, mask)

        out = out.view(B, S, D)
        out = self.out_proj(out)

        return out, attention_weights
