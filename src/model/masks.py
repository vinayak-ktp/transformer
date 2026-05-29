import torch


def generate_square_subsequent_mask(size):
    mask = torch.tril(torch.ones(size, size, dtype=torch.bool))
    return mask


def create_padding_mask(seq, pad_idx=0):
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)   # (B, 1, 1, S)


def create_tgt_mask(tgt, pad_idx=0):
    B, T = tgt.shape

    pad_mask = create_padding_mask(tgt, pad_idx)
    subsequent_mask = generate_square_subsequent_mask(T).to(tgt.device)

    return pad_mask & subsequent_mask
