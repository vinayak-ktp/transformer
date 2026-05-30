import torch

from src.inference.greedy_decode import greedy_decode


def predict(
    model,
    src_tensor,
    sos_idx,
    eos_idx,
    tgt_tokenizer,
    device,
):

    ids = greedy_decode(
        model,
        src_tensor,
        sos_idx,
        eos_idx,
        max_len=20,
        device=device,
    )

    token_ids = ids.squeeze().tolist()
    if hasattr(tgt_tokenizer, "decode"):
        return tgt_tokenizer.decode(token_ids)

    tokens = []

    for idx in token_ids:

        token = tgt_tokenizer.idx_to_token[idx]

        if token in (
            "<SOS>",
            "<EOS>",
            "<PAD>",
        ):
            continue

        tokens.append(token)

    return " ".join(tokens)
