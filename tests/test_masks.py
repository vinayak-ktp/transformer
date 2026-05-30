import torch

from src.model.masks import create_causal_mask, create_padding_mask, create_tgt_mask


def test_padding_mask():
    seq = torch.tensor([
        [1, 2, 3, 0, 0],
        [1, 2, 0, 0, 0]
    ])

    mask = create_padding_mask(seq)

    assert mask.shape == (2, 1, 1, 5)
    assert mask.dtype == torch.bool
    assert torch.equal(
        mask,
        torch.tensor([
            [[[True, True, True, False, False]]],
            [[[True, True, False, False, False]]]
        ])
    )


def test_causal_mask():
    mask = create_causal_mask(5)

    expected = torch.tensor([
        [
            [
                [True, False, False, False, False],
                [True, True, False, False, False],
                [True, True, True, False, False],
                [True, True, True, True, False],
                [True, True, True, True, True],
            ]
        ]
    ])

    assert mask.shape == (1, 1, 5, 5)
    assert mask.dtype == torch.bool
    assert torch.equal(mask, expected)


def test_target_mask_combines_padding_and_causal_masks():
    tgt = torch.tensor([
        [1, 4, 2, 0],
        [1, 5, 0, 0],
    ])

    mask = create_tgt_mask(tgt)

    expected = torch.tensor([
        [
            [
                [True, False, False, False],
                [True, True, False, False],
                [True, True, True, False],
                [True, True, True, False],
            ]
        ],
        [
            [
                [True, False, False, False],
                [True, True, False, False],
                [True, True, False, False],
                [True, True, False, False],
            ]
        ],
    ])

    assert mask.shape == (2, 1, 4, 4)
    assert mask.dtype == torch.bool
    assert torch.equal(mask, expected)
