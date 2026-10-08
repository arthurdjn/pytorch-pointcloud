r"""Stepwise exponential decay of the BatchNorm momentum over the epochs."""

from torch import nn

__all__ = ["BNMomentumScheduler"]


def bn_momentum(
    epoch: int,
    *,
    init: float = 0.5,
    decay_rate: float = 0.5,
    decay_step: int = 20,
    clip: float = 0.001,
) -> float:
    r"""The decayed BatchNorm momentum of an epoch.

    $$
    m_\text{epoch} = \max\left(m_0 \cdot \gamma^{\lfloor \text{epoch} / s \rfloor},\; m_\text{clip}\right)
    $$

    Args:
        epoch: Zero-based epoch.
        init: Initial momentum $m_0$.
        decay_rate: Per-step multiplicative decay $\gamma$.
        decay_step: Epochs between decay steps $s$.
        clip: Lower bound $m_\text{clip}$.

    Returns:
        The momentum to set on the `BatchNorm` layers for that epoch.

    Example:
        ```pycon
        >>> [bn_momentum(e) for e in (0, 19, 20, 40, 200)]
        [0.5, 0.5, 0.25, 0.125, 0.001]

        ```
    """
    return max(init * decay_rate ** (epoch // decay_step), clip)


def set_bn_momentum(model: nn.Module, momentum: float) -> None:
    r"""Set the momentum of every `BatchNorm1d` / `2d` / `3d` layer of `model`."""
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.momentum = momentum


class BNMomentumScheduler:
    r"""Apply [`bn_momentum`][torch_pointcloud.optim.bn_momentum.bn_momentum] to a model at the start of each epoch.

    Args:
        model: Model whose `BatchNorm` layers are scheduled.
        init: Initial momentum $m_0$.
        decay_rate: Per-step multiplicative decay $\gamma$.
        decay_step: Epochs between decay steps $s$.
        clip: Lower bound $m_\text{clip}$.

    Example:
        ```python
        import torch.nn as nn
        model = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4))
        scheduler = BNMomentumScheduler(model)
        scheduler.step(epoch=20)  # 0.25
        model[1].momentum  # 0.25
        ```
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        init: float = 0.5,
        decay_rate: float = 0.5,
        decay_step: int = 20,
        clip: float = 0.001,
    ) -> None:
        self.model = model
        self.init = init
        self.decay_rate = decay_rate
        self.decay_step = decay_step
        self.clip = clip

    def step(self, epoch: int) -> float:
        r"""Set the momentum of `epoch` on the model's `BatchNorm` layers and return it."""
        momentum = bn_momentum(
            epoch, init=self.init, decay_rate=self.decay_rate, decay_step=self.decay_step, clip=self.clip
        )
        set_bn_momentum(self.model, momentum)
        return momentum
