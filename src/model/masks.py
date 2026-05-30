import torch


def create_padding_mask(sequence, pad_idx=0):
    return (sequence != pad_idx).unsqueeze(1).unsqueeze(2)


def create_causal_mask(seq_len, device=None):
    mask = torch.tril(
        torch.ones(
            seq_len,
            seq_len,
            dtype=torch.bool,
            device=device
        )
    )

    return mask.unsqueeze(0).unsqueeze(0)


def create_tgt_mask(tgt, pad_idx=0):
    _, seq_len = tgt.shape

    padding_mask = create_padding_mask(
        tgt,
        pad_idx
    )

    causal_mask = create_causal_mask(
        seq_len,
        tgt.device
    )

    return padding_mask & causal_mask
