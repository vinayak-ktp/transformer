import torch

from src.inference.greedy_decode import greedy_decode


def evaluate_sample(
    model,
    src_tensor,
    sos_idx,
    eos_idx,
    device,
):
    output = greedy_decode(
        model=model,
        src=src_tensor,
        sos_idx=sos_idx,
        eos_idx=eos_idx,
        max_len=20,
        device=device,
    )

    return output
