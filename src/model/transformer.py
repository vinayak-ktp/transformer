import torch
import torch.nn as nn

from src.model.decoder import TransformerDecoder
from src.model.encoder import TransformerEncoder


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        num_layers,
        embed_dim,
        num_heads,
        hidden_dim,
        dropout=0.1
    ):
        super().__init__()

        self.encoder = TransformerEncoder(
            num_layers=num_layers,
            vocab_size=src_vocab_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            dropout=dropout
        )

        self.decoder = TransformerDecoder(
            num_layers=num_layers,
            vocab_size=tgt_vocab_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            dropout=dropout
        )

        self.fc = nn.Linear(embed_dim, tgt_vocab_size)
        # weight tying
        self.fc.weight = self.decoder.token_embedding.weight

    def forward(
        self,
        src,
        tgt,
        src_mask=None,
        tgt_mask=None,
        memory_mask=None
    ):
        memory = self.encoder(src, src_mask)
        out = self.decoder(tgt, memory, tgt_mask, memory_mask)
        logits = self.fc(out)

        return logits
