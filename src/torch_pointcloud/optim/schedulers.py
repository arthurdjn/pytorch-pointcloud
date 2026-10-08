r"""Learning-rate schedules with a linear warm-up, complementing `torch.optim.lr_scheduler`.

The schedules count in *steps*: call `scheduler.step()` once per epoch for an epoch-based schedule and once per
iteration for an iteration-based one; `total_steps` and `warmup_steps` are in the same unit.
"""

import math
from typing import List, Union

from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

__all__ = ["CosineWarmupLR", "PolyLR"]


class _WarmupSchedule(LRScheduler):
    r"""Linear warm-up followed by a decay towards `lr_min`.

    The first `warmup_steps` steps go linearly from `warmup_lr_init` to the base learning rate; the decay then
    runs over `total_steps` counted from step $0$ or, with `warmup_prefix`, over the
    `total_steps - warmup_steps` remaining ones counted from the end of the warm-up. Past the schedule the
    learning rate stays at `lr_min`.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        total_steps: int,
        *,
        warmup_steps: int = 0,
        warmup_lr_init: float = 0.0,
        lr_min: float = 0.0,
        warmup_prefix: bool = False,
        last_epoch: int = -1,
    ) -> None:
        if total_steps <= 0:
            raise ValueError(f"`total_steps` must be positive, got {total_steps}.")

        if not 0 <= warmup_steps < total_steps:
            raise ValueError(f"`warmup_steps` must be in [0, total_steps), got {warmup_steps}.")

        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.warmup_lr_init = warmup_lr_init
        self.lr_min = lr_min
        self.warmup_prefix = warmup_prefix
        super().__init__(optimizer, last_epoch)

    def _decay(self, progress: float) -> float:
        r"""Decay factor in $[0, 1]$ at `progress` $\in [0, 1)$ of the decay phase."""
        raise NotImplementedError

    def get_lr(self) -> List[Union[float, Tensor]]:
        t = self.last_epoch
        base_lrs = [float(base) for base in self.base_lrs]
        if t < self.warmup_steps:
            return [self.warmup_lr_init + t * (base - self.warmup_lr_init) / self.warmup_steps for base in base_lrs]

        if self.warmup_prefix:
            t, span = t - self.warmup_steps, self.total_steps - self.warmup_steps
        else:
            span = self.total_steps
        if t >= span:
            return [self.lr_min for _ in base_lrs]

        factor = self._decay(t / span)
        return [self.lr_min + (base - self.lr_min) * factor for base in base_lrs]


class CosineWarmupLR(_WarmupSchedule):
    r"""Cosine decay with linear warm-up (one cycle).

    $$
    \eta_t = \eta_\min + \tfrac{1}{2} (\eta_\text{base} - \eta_\min) \left(1 + \cos \frac{\pi\, t}{T}\right)
    $$

    after a linear warm-up from `warmup_lr_init` over `warmup_steps`. With `warmup_prefix=False` (the default),
    $t$ counts from step $0$ and $T$ is `total_steps`, so the warm-up takes its steps from the cosine; with
    `warmup_prefix=True`, $t$ and $T$ count from the end of the warm-up.

    Args:
        optimizer: Wrapped optimizer.
        total_steps: Length of the schedule in steps (epochs or iterations, whichever `step()` counts).
        warmup_steps: Length of the linear warm-up.
        warmup_lr_init: Learning rate at step $0$ of the warm-up.
        lr_min: Final (and post-schedule) learning rate.
        warmup_prefix: Count the cosine from the end of the warm-up instead of from step $0$.
        last_epoch: The index of the last step, for resuming.

    Example:
        ```python
        import torch
        opt = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=1.0)
        sched = CosineWarmupLR(opt, total_steps=10, warmup_steps=2, lr_min=0.0)
        [round(sched.get_last_lr()[0], 3) for _ in range(10) if not sched.step()]  # [0.5, 0.905, 0.794, 0.655, 0.5, 0.345, 0.206, 0.095, 0.024, 0.0]
        ```
    """

    def _decay(self, progress: float) -> float:
        return 0.5 * (1 + math.cos(math.pi * progress))


class PolyLR(_WarmupSchedule):
    r"""Polynomial decay with linear warm-up (one cycle).

    $$
    \eta_t = \eta_\min + (\eta_\text{base} - \eta_\min) \left(1 - \frac{t}{T}\right)^{p}
    $$

    after the warm-up; see [`CosineWarmupLR`][torch_pointcloud.optim.schedulers.CosineWarmupLR] for how $t$
    and $T$ are counted. The decay reaches `lr_min` at step `total_steps`; to keep a non-zero rate on the last of
    $n$ steps, pass `total_steps=n + 1`.

    Args:
        optimizer: Wrapped optimizer.
        total_steps: Length of the schedule in steps.
        power: Exponent $p$ of the decay.
        warmup_steps: Length of the linear warm-up.
        warmup_lr_init: Learning rate at step $0$ of the warm-up.
        lr_min: Final (and post-schedule) learning rate.
        warmup_prefix: Count the decay from the end of the warm-up instead of from step $0$.
        last_epoch: The index of the last step, for resuming.

    Example:
        ```python
        import torch
        opt = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=1.0)
        sched = PolyLR(opt, total_steps=4, power=1.0)
        [round(sched.get_last_lr()[0], 3) for _ in range(4) if not sched.step()]  # [0.75, 0.5, 0.25, 0.0]
        ```
    """

    def __init__(
        self,
        optimizer: Optimizer,
        total_steps: int,
        *,
        power: float = 0.9,
        warmup_steps: int = 0,
        warmup_lr_init: float = 0.0,
        lr_min: float = 0.0,
        warmup_prefix: bool = False,
        last_epoch: int = -1,
    ) -> None:
        self.power = power
        super().__init__(
            optimizer,
            total_steps,
            warmup_steps=warmup_steps,
            warmup_lr_init=warmup_lr_init,
            lr_min=lr_min,
            warmup_prefix=warmup_prefix,
            last_epoch=last_epoch,
        )

    def _decay(self, progress: float) -> float:
        return (1 - progress) ** self.power
