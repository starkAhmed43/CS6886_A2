"""Learning-rate scheduler helpers."""

import torch


def warmup_cosine(
    optimizer: torch.optim.Optimizer, warmup_epochs: int = 5, max_epochs: int = 200
) -> torch.optim.lr_scheduler.SequentialLR:
    """Build a linear-warmup followed by cosine-annealing LR schedule.

    Linearly ramps the LR from 1% to 100% over `warmup_epochs`, then anneals it to 0
    following a cosine curve over the remaining epochs. Steps once per epoch.

    :param optimizer: The optimizer whose LR is scheduled.
    :param warmup_epochs: Number of epochs to linearly warm up the LR.
    :param max_epochs: Total number of training epochs.
    :return: A `SequentialLR` combining the warmup and cosine-annealing schedules.
    """
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs - warmup_epochs
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
    )
