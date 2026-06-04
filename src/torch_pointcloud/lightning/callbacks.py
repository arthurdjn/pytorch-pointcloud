import lightning.pytorch as L
from torch import nn


class BNMomentumScheduler(L.Callback):
    r"""Exponentially decay BatchNorm momentum over training epochs.

    Reference implementation: :github:
    [facebookresearch/votenet](https://github.com/facebookresearch/votenet) (`train.py`).

    At the start of each training epoch, every `nn.BatchNorm*` module in `pl_module.model` has its
    momentum set to

    $$
    \max\left(m_0 \cdot \gamma^{\lfloor \text{epoch} / s \rfloor},\; m_\text{clip}\right)
    $$

    with $m_0$ the initial momentum, $\gamma$ the decay rate, $s$ the decay step (epochs) and
    $m_\text{clip}$ the floor.

    Args:
        bn_momentum_init: Initial BatchNorm momentum $m_0$.
        bn_decay_rate: Per-step multiplicative decay $\gamma$.
        bn_decay_step: Number of epochs between decay steps $s$.
        bn_momentum_clip: Lower bound on the momentum $m_\text{clip}$.
    """

    def __init__(
        self,
        bn_momentum_init: float = 0.5,
        bn_decay_rate: float = 0.5,
        bn_decay_step: int = 20,
        bn_momentum_clip: float = 0.001,
    ) -> None:
        super().__init__()
        self.bn_momentum_init = bn_momentum_init
        self.bn_decay_rate = bn_decay_rate
        self.bn_decay_step = bn_decay_step
        self.bn_momentum_clip = bn_momentum_clip

    def on_train_epoch_start(self, trainer: "L.Trainer", pl_module: "L.LightningModule") -> None:
        momentum = max(
            self.bn_momentum_init * self.bn_decay_rate ** (trainer.current_epoch // self.bn_decay_step),
            self.bn_momentum_clip,
        )
        model = pl_module.model
        assert isinstance(model, nn.Module)
        for module in model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                module.momentum = momentum
