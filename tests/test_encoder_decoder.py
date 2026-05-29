import torch

from src.model.decoder import TransformerDecoder, TransformerDecoderLayer
from src.model.encoder import TransformerEncoder, TransformerEncoderLayer


def test_encoder_layer():
    x = torch.randn(2, 10, 512)

    layer = TransformerEncoderLayer(
        embed_dim=512,
        num_heads=8,
        hidden_dim=2048
    )

    out = layer(x)

    assert out.shape == x.shape


def test_encoder():
    src = torch.randint(0, 1000, (2, 10))

    encoder = TransformerEncoder(
        num_layers=2,
        vocab_size=1000,
        embed_dim=128,
        num_heads=4,
        hidden_dim=512
    )

    out = encoder(src)

    assert out.shape == (2, 10, 128)


def test_decoder_layer():
    tgt = torch.randn(2, 8, 128)
    memory = torch.randn(2, 10, 128)

    layer = TransformerDecoderLayer(
        embed_dim=128,
        num_heads=4,
        hidden_dim=512
    )

    out = layer(tgt, memory)

    assert out.shape == tgt.shape


def test_decoder():
    tgt = torch.randint(0, 1000, (2, 8))
    memory = torch.randn(2, 10, 128)

    decoder = TransformerDecoder(
        num_layers=2,
        vocab_size=1000,
        embed_dim=128,
        num_heads=4,
        hidden_dim=512
    )

    out = decoder(tgt, memory)

    assert out.shape == (2, 8, 128)
