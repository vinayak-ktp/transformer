import random
from functools import partial
from pathlib import Path

import sacrebleu
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from scripts.download_data import download_data
from src.data.dataloader import collate_fn
from src.data.dataset import TranslationDataset
from src.data.tokenizer import SentencePieceTokenizer, train_sentencepiece_tokenizer
from src.inference.greedy_decode import greedy_decode
from src.model.masks import create_padding_mask, create_tgt_mask
from src.model.transformer import Transformer
from src.training.checkpoint import load_checkpoint, save_checkpoint
from src.training.scheduler import TransformerLRScheduler
from src.training.trainer import train_one_epoch
from src.utils.config import load_config
from src.utils.seed import set_seed


def load_pairs(path, max_pairs):
    src_sentences, tgt_sentences = [], []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_pairs:
                break
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            src_sentences.append(parts[0].strip())
            tgt_sentences.append(parts[1].strip())
    return src_sentences, tgt_sentences


def filter_by_length(src_list, tgt_list, tokenizer, max_src, max_tgt):
    filtered_src, filtered_tgt = [], []
    for s, t in zip(src_list, tgt_list):
        # tgt in the dataset gets SOS + tokens + EOS, so reserve 2 slots
        if len(tokenizer.encode(s)) <= max_src and len(tokenizer.encode(t)) <= max_tgt - 2:
            filtered_src.append(s)
            filtered_tgt.append(t)
    return filtered_src, filtered_tgt


def deduplicate_pairs(src_list, tgt_list, seed=42):
    """Keep one randomly-chosen French translation per English sentence.

    ManyThings maps the same EN sentence to many FR variants (formal/informal,
    gendered, regional). Without deduplication, 57% of training pairs are
    contradictory (same src, different tgt), which floors the loss at the
    entropy of the target distribution and prevents convergence.
    """
    from collections import defaultdict
    import random
    rng = random.Random(seed)
    groups = defaultdict(list)
    for s, t in zip(src_list, tgt_list):
        groups[s].append(t)
    deduped_src, deduped_tgt = [], []
    for s, targets in groups.items():
        deduped_src.append(s)
        deduped_tgt.append(rng.choice(targets))
    return deduped_src, deduped_tgt


def train_val_split(src, tgt, val_ratio=0.1, seed=42):
    pairs = list(zip(src, tgt))
    rng = random.Random(seed)
    rng.shuffle(pairs)
    split = int(len(pairs) * (1 - val_ratio))
    train_pairs = pairs[:split]
    val_pairs = pairs[split:]
    train_src, train_tgt = zip(*train_pairs)
    val_src, val_tgt = zip(*val_pairs)
    return list(train_src), list(train_tgt), list(val_src), list(val_tgt)


def compute_val_loss(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for src, tgt in dataloader:
            src = src.to(device)
            tgt = tgt.to(device)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            src_mask = create_padding_mask(src)
            tgt_mask = create_tgt_mask(tgt_input)
            logits = model(src, tgt_input, src_mask=src_mask, tgt_mask=tgt_mask, memory_mask=src_mask)
            logits = logits.reshape(-1, logits.shape[-1])
            tgt_output = tgt_output.reshape(-1)
            loss = criterion(logits, tgt_output)
            total_loss += loss.item()
    return total_loss / len(dataloader)


def compute_bleu(model, val_src, val_tgt, tokenizer, device, max_len, num_samples, seed):
    model.eval()

    indices = list(range(len(val_src)))
    random.seed(seed)
    random.shuffle(indices)
    indices = indices[:num_samples]

    hypotheses = []
    references = []

    sos_idx = tokenizer.sos_id
    eos_idx = tokenizer.eos_id

    with torch.no_grad():
        for idx in indices:
            src_ids = tokenizer.encode(val_src[idx])
            src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)

            output_ids = greedy_decode(
                model=model,
                src=src_tensor,
                sos_idx=sos_idx,
                eos_idx=eos_idx,
                max_len=max_len,
                device=device,
            )

            hyp_ids = output_ids.squeeze().tolist()
            hyp_text = tokenizer.decode(hyp_ids)

            hypotheses.append(hyp_text)
            references.append(val_tgt[idx])

    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    return bleu.score


def main():
    data_cfg = load_config("configs/data_config.yaml")
    model_cfg = load_config("configs/model_config.yaml")
    train_cfg = load_config("configs/train_config.yaml")

    seed = train_cfg["seed"]
    set_seed(seed)

    device = train_cfg["device"] if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Download data if needed
    data_path = data_cfg["data_path"]
    download_data(data_path)

    # Load raw pairs
    max_pairs = data_cfg["max_pairs"]
    print(f"\nLoading up to {max_pairs:,} pairs from {data_path} …")
    src_raw, tgt_raw = load_pairs(data_path, max_pairs)
    print(f"  Loaded {len(src_raw):,} pairs")

    # Deduplicate: keep one FR translation per EN sentence
    src_raw, tgt_raw = deduplicate_pairs(src_raw, tgt_raw, seed=seed)
    print(f"  After deduplication: {len(src_raw):,} unique pairs")

    # Train shared BPE tokenizer
    tokenizer_prefix = data_cfg["tokenizer_prefix"]
    vocab_size = data_cfg["tokenizer_vocab_size"]
    tok_model = Path(tokenizer_prefix).with_suffix(".model")

    if tok_model.exists():
        print(f"\nFound existing tokenizer at {tok_model}, loading …")
        tokenizer = SentencePieceTokenizer(tok_model)
    else:
        print(f"\nTraining shared BPE tokenizer (vocab_size={vocab_size}) …")
        all_sentences = src_raw + tgt_raw
        tokenizer = train_sentencepiece_tokenizer(
            sentences=all_sentences,
            model_prefix=tokenizer_prefix,
            vocab_size=vocab_size,
            model_type=data_cfg["tokenizer_model_type"],
        )
    print(f"  Tokenizer vocab size: {len(tokenizer):,}")

    # Filter by length
    max_src_len = data_cfg["max_src_len"]
    max_tgt_len = data_cfg["max_tgt_len"]
    print(f"\nFiltering pairs to max_src={max_src_len}, max_tgt={max_tgt_len} …")
    src_filt, tgt_filt = filter_by_length(src_raw, tgt_raw, tokenizer, max_src_len, max_tgt_len)
    print(f"  Retained {len(src_filt):,} pairs after filtering")

    # Train / val split
    train_src, train_tgt, val_src, val_tgt = train_val_split(
        src_filt, tgt_filt, val_ratio=data_cfg["val_ratio"], seed=seed
    )
    print(f"  Train: {len(train_src):,}  |  Val: {len(val_src):,}")

    # Datasets & dataloaders
    train_dataset = TranslationDataset(train_src, train_tgt, tokenizer, tokenizer)
    val_dataset = TranslationDataset(val_src, val_tgt, tokenizer, tokenizer)

    _collate = partial(collate_fn, pad_idx=tokenizer.pad_id)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        collate_fn=_collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        collate_fn=_collate,
    )

    # Model
    embed_dim = model_cfg["embed_dim"]
    vocab_size = len(tokenizer)

    model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        num_layers=model_cfg["num_layers"],
        embed_dim=embed_dim,
        num_heads=model_cfg["num_heads"],
        hidden_dim=model_cfg["hidden_dim"],
        dropout=model_cfg["dropout"],
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {num_params:,}")

    # Optimizer, scheduler, criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = TransformerLRScheduler(
        optimizer,
        embed_dim=embed_dim,
        warmup_steps=train_cfg["warmup_steps"],
        factor=train_cfg["scheduler_factor"],
    )
    criterion = nn.CrossEntropyLoss(
        ignore_index=tokenizer.pad_id,
        label_smoothing=train_cfg["label_smoothing"],
    )

    # Training loop
    epochs = train_cfg["epochs"]
    save_every = train_cfg["save_every"]
    bleu_eval_samples = data_cfg["bleu_eval_samples"]
    max_decode_len = max_tgt_len + 5

    best_bleu = -1.0
    best_epoch = 0

    print(f"\n{'─'*70}")
    print(f"{'Epoch':>6}  {'Train Loss':>10}  {'Val Loss':>9}  {'BLEU':>7}")
    print(f"{'─'*70}")

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            scheduler=scheduler, grad_clip=train_cfg.get("gradient_clip", 1.0),
        )

        val_loss = compute_val_loss(model, val_loader, criterion, device)

        bleu = compute_bleu(
            model, val_src, val_tgt, tokenizer, device,
            max_len=max_decode_len,
            num_samples=bleu_eval_samples,
            seed=seed,
        )

        is_best = bleu > best_bleu
        if is_best:
            best_bleu = bleu
            best_epoch = epoch
            save_checkpoint(
                model, optimizer, epoch, train_loss,
                path=str(checkpoint_dir / "best.pt"),
                scheduler=scheduler,
            )

        if epoch % save_every == 0:
            save_checkpoint(
                model, optimizer, epoch, train_loss,
                path=str(checkpoint_dir / f"epoch_{epoch:02d}.pt"),
                scheduler=scheduler,
            )

        marker = " ←" if is_best else ""
        print(f"{epoch:>6}  {train_loss:>10.4f}  {val_loss:>9.4f}  {bleu:>7.2f}{marker}")

    print(f"{'─'*70}")
    print(f"\nTraining complete. Best BLEU: {best_bleu:.2f} at epoch {best_epoch}.")
    print(f"Best checkpoint saved to {checkpoint_dir / 'best.pt'}")

    # Final qualitative examples
    print("\n── Sample translations (from best checkpoint) ──")
    load_checkpoint(
        model, optimizer,
        path=str(checkpoint_dir / "best.pt"),
        device=device,
        scheduler=scheduler,
    )
    model.eval()

    samples = [
        "Go.",
        "Hello!",
        "Thank you.",
        "I love you.",
        "How are you?",
        "I don't know.",
        "She is very kind.",
        "We need to talk.",
    ]

    sos_idx = tokenizer.sos_id
    eos_idx = tokenizer.eos_id

    with torch.no_grad():
        for sent in samples:
            src_ids = tokenizer.encode(sent)
            src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)
            output_ids = greedy_decode(
                model=model,
                src=src_tensor,
                sos_idx=sos_idx,
                eos_idx=eos_idx,
                max_len=max_decode_len,
                device=device,
            )
            hyp = tokenizer.decode(output_ids.squeeze().tolist())
            print(f"  EN: {sent!r:30s}  →  FR: {hyp!r}")


if __name__ == "__main__":
    main()
