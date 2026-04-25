import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(batch, pad_idx=0):
    src_batch = [item["src"] for item in batch]
    tgt_batch = [item["tgt"] for item in batch]

    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=pad_idx)
    tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=pad_idx)

    return src_batch, tgt_batch
