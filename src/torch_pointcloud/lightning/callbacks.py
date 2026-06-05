from typing import Any, Callable, Dict, Optional, Sequence, Union

import lightning.pytorch as L
from torch import nn
from torchmetrics import Metric

from torch_pointcloud.models import ClassificationModel, DetectionModel, SegmentationModel


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


class MetricCallback(L.Callback):
    r"""Accumulate and log a torchmetrics metric over a validation/test epoch.

    Model- and task-agnostic: the LightningModule's `validation_step` / `test_step` returns a
    `{preds_key: ..., target_key: ...}` dict (the repo's `Lit*` modules do), and this callback updates a
    torchmetrics `Metric` with it each batch, logging the epoch value as `{stage}/{name}`. List one per
    metric in `configs/callbacks/*.yaml` to plug accuracy, mIoU, mAP, ... onto any model.

    `metric` is either a ready `Metric` or a factory (a Hydra `_partial_`) completed at fit time by calling
    it as `metric(num_classes=..., ignore_index=...)` with the model's `num_classes` and the module's
    `ignore_index`, so the class count is never duplicated in config.

    A metric whose `compute` returns a dict (e.g. `MeanAveragePrecision3D` returning `mAP@0.25` / `mAP@0.5`)
    is logged one entry per key as `{stage}/{key}`; a scalar metric is logged as `{stage}/{name}`.

    Args:
        metric: A torchmetrics `Metric`, or a callable `metric(num_classes=..., ignore_index=...) -> Metric`.
        name: Metric name; logged as `{stage}/{name}` (ignored for dict-valued metrics, whose keys name themselves).
        stages: Stages to score; each listed stage's `*_step` must return `preds_key` / `target_key`.
        preds_key: Key in the step output holding predictions (logits, probabilities, labels, or detections).
        target_key: Key in the step output holding the targets.
        prog_bar: Whether to show the metric on the progress bar.
    """

    def __init__(
        self,
        metric: Union[Metric, Callable[..., Metric]],
        name: str,
        *,
        stages: Sequence[str] = ("val", "test"),
        preds_key: str = "preds",
        target_key: str = "target",
        prog_bar: bool = True,
    ) -> None:
        super().__init__()
        self.name = name
        self.stages = tuple(stages)
        self.preds_key = preds_key
        self.target_key = target_key
        self.prog_bar = prog_bar
        self.metric: Optional[Metric] = metric if isinstance(metric, Metric) else None
        self._factory: Optional[Callable[..., Metric]] = None if isinstance(metric, Metric) else metric

    def setup(self, trainer: "L.Trainer", pl_module: "L.LightningModule", stage: str) -> None:
        if self.metric is not None or self._factory is None:
            return
        model = pl_module.model
        assert isinstance(model, (ClassificationModel, SegmentationModel, DetectionModel))
        self.metric = self._factory(
            num_classes=model.num_classes, ignore_index=getattr(pl_module, "ignore_index", None)
        )

    def _reset(self, pl_module: "L.LightningModule") -> None:
        assert self.metric is not None
        self.metric = self.metric.to(pl_module.device)
        self.metric.reset()

    def _update(self, outputs: Any) -> None:
        assert self.metric is not None
        if isinstance(outputs, dict):
            self.metric.update(outputs[self.preds_key], outputs[self.target_key])

    def _compute(self, pl_module: "L.LightningModule", stage: str) -> None:
        assert self.metric is not None
        value = self.metric.compute()
        if isinstance(value, dict):
            for i, (key, sub) in enumerate(value.items()):
                pl_module.log(f"{stage}/{key}", sub, prog_bar=self.prog_bar and i == 0)
        else:
            pl_module.log(f"{stage}/{self.name}", value, prog_bar=self.prog_bar)

    def on_validation_epoch_start(self, trainer: "L.Trainer", pl_module: "L.LightningModule") -> None:
        if "val" in self.stages:
            self._reset(pl_module)

    def on_validation_batch_end(
        self,
        trainer: "L.Trainer",
        pl_module: "L.LightningModule",
        outputs: Any,
        batch: Dict[str, Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if "val" in self.stages:
            self._update(outputs)

    def on_validation_epoch_end(self, trainer: "L.Trainer", pl_module: "L.LightningModule") -> None:
        if "val" in self.stages:
            self._compute(pl_module, "val")

    def on_test_epoch_start(self, trainer: "L.Trainer", pl_module: "L.LightningModule") -> None:
        if "test" in self.stages:
            self._reset(pl_module)

    def on_test_batch_end(
        self,
        trainer: "L.Trainer",
        pl_module: "L.LightningModule",
        outputs: Any,
        batch: Dict[str, Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if "test" in self.stages:
            self._update(outputs)

    def on_test_epoch_end(self, trainer: "L.Trainer", pl_module: "L.LightningModule") -> None:
        if "test" in self.stages:
            self._compute(pl_module, "test")
