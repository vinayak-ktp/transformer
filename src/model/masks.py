import torch


def generate_square_subsequent_mask(size):
    mask = torch.tril(torch.ones(size, size))
    return mask
