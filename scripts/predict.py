import torch

from src.inference.greedy_decode import greedy_decode


def predict(
    model,
    src_tensor,
    sos_idx,
    eos_idx,
    tgt_vocab,
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

    tokens = []

    for idx in ids.squeeze().tolist():

        token = tgt_vocab.idx_to_token[idx]

        if token in (
            "<SOS>",
            "<EOS>",
            "<PAD>",
        ):
            continue

        tokens.append(token)

    return " ".join(tokens)
