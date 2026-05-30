import math

import torch

from src.training.scheduler import TransformerLRScheduler


def test_transformer_lr_scheduler_matches_paper_formula():
    optimizer = torch.optim.Adam(
        torch.nn.Linear(2, 2).parameters(),
        lr=0.0
    )
    scheduler = TransformerLRScheduler(
        optimizer,
        embed_dim=512,
        warmup_steps=4000
    )

    expected = (512 ** -0.5) * min(
        4000 ** -0.5,
        4000 * (4000 ** -1.5)
    )

    assert math.isclose(
        scheduler.get_lr(4000),
        expected,
        rel_tol=1e-12
    )


def test_transformer_lr_scheduler_warms_up_then_decays():
    optimizer = torch.optim.Adam(
        torch.nn.Linear(2, 2).parameters(),
        lr=0.0
    )
    scheduler = TransformerLRScheduler(
        optimizer,
        embed_dim=512,
        warmup_steps=4
    )

    rates = [
        scheduler.get_lr(step)
        for step in range(1, 9)
    ]

    assert rates[0] < rates[1] < rates[2] < rates[3]
    assert rates[4] < rates[3]
    assert rates[7] < rates[4]


def test_transformer_lr_scheduler_updates_optimizer_lr():
    optimizer = torch.optim.Adam(
        torch.nn.Linear(2, 2).parameters(),
        lr=0.0
    )
    scheduler = TransformerLRScheduler(
        optimizer,
        embed_dim=16,
        warmup_steps=2
    )

    lr = scheduler.step()

    assert scheduler.step_num == 1
    assert optimizer.param_groups[0]["lr"] == lr
    assert lr == scheduler.get_lr(1)


def test_transformer_lr_scheduler_restores_state():
    optimizer = torch.optim.Adam(
        torch.nn.Linear(2, 2).parameters(),
        lr=0.0
    )
    scheduler = TransformerLRScheduler(
        optimizer,
        embed_dim=16,
        warmup_steps=2
    )
    scheduler.step()
    scheduler.step()

    restored = TransformerLRScheduler(
        optimizer,
        embed_dim=32,
        warmup_steps=4
    )
    restored.load_state_dict(scheduler.state_dict())

    assert restored.embed_dim == 16
    assert restored.warmup_steps == 2
    assert restored.step_num == 2
    assert optimizer.param_groups[0]["lr"] == restored.get_lr()
