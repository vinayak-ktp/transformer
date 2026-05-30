import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataloader import collate_fn
from src.data.dataset import TranslationDataset
from src.data.vocab import Vocabulary
from src.model.transformer import Transformer
from src.training.checkpoint import save_checkpoint
from src.training.scheduler import TransformerLRScheduler
from src.training.trainer import train_one_epoch
from src.utils.seed import set_seed


def main():

    set_seed(42)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    src_sentences = [
        "i love ai",
        "how are you",
        "good morning",
        "thank you",
        "hello",
    ]

    tgt_sentences = [
        "j aime ai",
        "comment allez vous",
        "bonjour",
        "merci",
        "salut",
    ]

    src_vocab = Vocabulary(min_freq=1)
    tgt_vocab = Vocabulary(min_freq=1)

    src_vocab.build_vocab(
        [s.split() for s in src_sentences]
    )

    tgt_vocab.build_vocab(
        [s.split() for s in tgt_sentences]
    )

    dataset = TranslationDataset(
        src_sentences,
        tgt_sentences,
        src_vocab,
        tgt_vocab,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=collate_fn,
    )

    embed_dim = 128

    model = Transformer(
        src_vocab_size=len(src_vocab.token_to_idx),
        tgt_vocab_size=len(tgt_vocab.token_to_idx),
        num_layers=2,
        embed_dim=embed_dim,
        num_heads=4,
        hidden_dim=512,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.0,
    )
    scheduler = TransformerLRScheduler(
        optimizer,
        embed_dim=embed_dim,
        warmup_steps=4000,
    )

    criterion = nn.CrossEntropyLoss(
        ignore_index=0,
        label_smoothing=0.1
    )

    epochs = 100

    for epoch in range(epochs):

        loss = train_one_epoch(
            model,
            dataloader,
            optimizer,
            criterion,
            device,
            scheduler=scheduler,
        )

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Loss: {loss:.4f}"
        )

        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                model,
                optimizer,
                epoch,
                loss,
                f"checkpoints/epoch_{epoch+1}.pt",
                scheduler=scheduler,
            )


if __name__ == "__main__":
    main()
