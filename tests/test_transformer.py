import torch

from src.model.transformer import Transformer


def test_transformer():
    src = torch.randint(0, 1000, (2, 10))
    tgt = torch.randint(0, 1000, (2, 8))

    model = Transformer(
        src_vocab_size=1000,
        tgt_vocab_size=1200,
        num_layers=2,
        embed_dim=128,
        num_heads=4,
        hidden_dim=512
    )

    logits = model(src, tgt)

    assert logits.shape == (2, 8, 1200)
