import argparse
from pathlib import Path

import torch

from src.data.tokenizer import SentencePieceTokenizer
from src.inference.beam_search import beam_search_decode
from src.inference.greedy_decode import greedy_decode
from src.model.transformer import Transformer
from src.training.checkpoint import load_checkpoint
from src.utils.config import load_config


def build_model(model_cfg, vocab_size, device):
    model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        num_layers=model_cfg["num_layers"],
        embed_dim=model_cfg["embed_dim"],
        num_heads=model_cfg["num_heads"],
        hidden_dim=model_cfg["hidden_dim"],
        dropout=model_cfg["dropout"],
    ).to(device)
    return model


def translate(model, sentence, tokenizer, device, mode, beam_width, max_len):
    src_ids = tokenizer.encode(sentence)
    src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)

    if mode == "beam":
        output_ids = beam_search_decode(
            model=model,
            src=src_tensor,
            sos_idx=tokenizer.sos_id,
            eos_idx=tokenizer.eos_id,
            max_len=max_len,
            beam_width=beam_width,
            device=device,
        )
    else:
        output_ids = greedy_decode(
            model=model,
            src=src_tensor,
            sos_idx=tokenizer.sos_id,
            eos_idx=tokenizer.eos_id,
            max_len=max_len,
            device=device,
        )

    return tokenizer.decode(output_ids.squeeze().tolist())


def main():
    parser = argparse.ArgumentParser(description="Translate English to Hinglish.")
    parser.add_argument("sentence", nargs="?", help="Sentence to translate. Omit for interactive mode.")
    parser.add_argument("--checkpoint", default="best", help="Checkpoint name under checkpoints/ (e.g. best, epoch_20).")
    parser.add_argument("--mode", choices=["greedy", "beam"], default="greedy", help="Decoding strategy.")
    parser.add_argument("--beam-width", type=int, default=4, help="Beam width (only used with --mode beam).")
    parser.add_argument("--max-len", type=int, default=50, help="Maximum output length.")
    args = parser.parse_args()

    checkpoint_path = Path("checkpoints") / args.checkpoint
    if not checkpoint_path.suffix:
        checkpoint_path = checkpoint_path.with_suffix(".pt")

    data_cfg = load_config("configs/data_config.yaml")
    model_cfg = load_config("configs/model_config.yaml")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok_path = Path(data_cfg["tokenizer_prefix"]).with_suffix(".model")
    if not tok_path.exists():
        raise FileNotFoundError(
            f"Tokenizer not found at {tok_path}. Run train.py first."
        )
    tokenizer = SentencePieceTokenizer(tok_path)

    model = build_model(model_cfg, len(tokenizer), device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0)
    load_checkpoint(model, optimizer, path=str(checkpoint_path), device=device)
    model.eval()

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Decoding: {args.mode}" + (f"  (beam_width={args.beam_width})" if args.mode == "beam" else ""))

    EXIT_CMDS = {"exit", "quit", "q"}

    if args.sentence:
        result = translate(model, args.sentence, tokenizer, device, args.mode, args.beam_width, args.max_len)
        print(f"\n  EN: {args.sentence}")
        print(f"  FR: {result}")
    else:
        print("\nInteractive mode — type a sentence and press Enter. Type 'exit' to quit.\n")
        while True:
            try:
                sentence = input("EN > ").strip()
            except KeyboardInterrupt:
                print("\nExiting.")
                break
            if not sentence:
                continue
            if sentence.lower() in EXIT_CMDS:
                print("Exiting.")
                break
            result = translate(model, sentence, tokenizer, device, args.mode, args.beam_width, args.max_len)
            print(f"FR > {result}\n\n")


if __name__ == "__main__":
    main()
