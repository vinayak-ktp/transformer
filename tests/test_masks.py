import torch

from src.model.masks import create_padding_mask, generate_square_subsequent_mask


def test_padding_mask():
    seq = torch.tensor([
        [1, 2, 3, 0, 0],
        [1, 2, 0, 0, 0]
    ])

    mask = create_padding_mask(seq)

    assert mask.shape == (2, 1, 1, 5)


def test_subsequent_mask():
    mask = generate_square_subsequent_mask(5)

    expected = torch.tensor([
        [1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [1, 1, 1, 0, 0],
        [1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1]
    ])

    assert torch.equal(mask, expected)
