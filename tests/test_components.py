import torch

from src.model.layers.attention import ScaledDotProductAttention
from src.model.layers.embeddings import TokenEmbedding
from src.model.layers.feed_forward import PositionwiseFeedForward
from src.model.layers.layer_norm import LayerNorm
from src.model.layers.multihead_attention import MultiheadAttention
from src.model.layers.positional_encoding import PositionalEncoding


def test_embedding():
    x = torch.randint(0, 1000, (2, 10))

    embedding = TokenEmbedding(
        vocab_size=1000,
        embed_dim=128
    )

    out = embedding(x)

    assert out.shape == (2, 10, 128)


def test_positional_encoding():
    x = torch.randn(2, 10, 128)

    pe = PositionalEncoding(128)

    out = pe(x)

    assert out.shape == x.shape


def test_layer_norm():
    x = torch.randn(2, 10, 128)

    norm = LayerNorm(128)

    out = norm(x)

    assert out.shape == x.shape


def test_feed_forward():
    x = torch.randn(2, 10, 128)

    ffn = PositionwiseFeedForward(
        embed_dim=128,
        hidden_dim=512
    )

    out = ffn(x)

    assert out.shape == x.shape


def test_scaled_dot_product_attention():
    B, H, S, D = 2, 4, 10, 64

    Q = torch.randn(B, H, S, D)
    K = torch.randn(B, H, S, D)
    V = torch.randn(B, H, S, D)

    attention = ScaledDotProductAttention()

    out, weights = attention(Q, K, V)

    assert out.shape == (B, H, S, D)
    assert weights.shape == (B, H, S, S)


def test_multihead_attention():
    x = torch.randn(2, 10, 512)

    mha = MultiheadAttention(
        embed_dim=512,
        num_heads=8
    )

    out, weights = mha(x, x, x)

    assert out.shape == (2, 10, 512)
    assert weights.shape == (2, 8, 10, 10)
