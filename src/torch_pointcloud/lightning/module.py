from typing import Any, Callable, Dict, Optional, Sequence

import lightning.pytorch as L
import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torchmetrics import Accuracy, JaccardIndex, Metric

from torch_pointcloud.models import ClassificationModel, SegmentationModel
from torch_pointcloud.utils.optim import generate_param_groups


def _resolve_input(batch: Dict[str, Any], key: str) -> Any:
    """Resolve a batch entry, supporting dotted attribute access (e.g. `octree.depth`)."""
    name, _, attr = key.partition(".")
    value = batch.get(name)
    return getattr(value, attr) if attr else value


class LiTModel(L.LightningModule):
    """Shared base for the task-specific Lightning wrappers."""

    def __init__(
        self,
        model: nn.Module,
        metric: Metric,
        *,
        optimizer: Callable[..., Optimizer],
        scheduler: Optional[Callable[..., Any]] = None,
        criterion: Optional[nn.Module] = None,
        metric_name: str = "metric",
        input_keys: Sequence[str] = ("x", "pos", "batch"),
        target_key: str = "segment",
        scheduler_interval: str = "epoch",
        param_groups: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.metric = metric
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._param_groups = param_groups
        self.save_hyperparameters("metric_name", "input_keys", "target_key", "scheduler_interval")

    def forward(self, batch: Dict[str, Any]) -> Tensor:
        return self.model(*(_resolve_input(batch, key) for key in self.hparams["input_keys"]))

    def step(self, batch: Dict[str, Any], stage: str) -> Tensor:
        logits = self.forward(batch)
        target = batch[self.hparams["target_key"]].long()
        loss = self.criterion(logits, target)
        batch_size = int(batch["batch"][-1]) + 1
        self.log(f"{stage}/loss", loss, prog_bar=True, batch_size=batch_size)
        if stage != "train":
            self.metric(logits, target)
            self.log(f"{stage}/{self.hparams['metric_name']}", self.metric, prog_bar=True, batch_size=batch_size)
        return loss

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Tensor:
        return self.step(batch, "train")

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        self.step(batch, "val")

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        self.step(batch, "test")

    def configure_optimizers(self) -> Any:
        params = generate_param_groups(self, **self._param_groups) if self._param_groups else self.parameters()
        optimizer = self._optimizer(params)
        if self._scheduler is None:
            return optimizer
        scheduler = {"scheduler": self._scheduler(optimizer), "interval": self.hparams["scheduler_interval"]}
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


class LitClassificationModel(LiTModel):
    """LightningModule wrapping a point cloud classification model.

    Args:
        model: A classification model, typically built via `create_model`.
        optimizer: A callable that takes parameters and returns an optimizer
            (a `_partial_` config target).
        scheduler: An optional callable that takes an optimizer and returns a
            learning-rate scheduler.
        criterion: The loss module; defaults to `CrossEntropyLoss`.
        input_keys: Batch-dict keys passed positionally to the model's forward.
            A dotted key (e.g. `octree.depth`) resolves an attribute.
        target_key: Batch-dict key for the per-cloud label.
        scheduler_interval: Whether the scheduler steps per `"epoch"` or `"step"`.
        param_groups: Optional dict of kwargs forwarded to
            `torch_pointcloud.utils.optim.generate_param_groups` (`layer_matches`,
            `match_types`, `lr_values`, `include_others`). Same shape as MONAI's
            `generate_param_groups`.
    """

    def __init__(
        self,
        model: ClassificationModel,
        *,
        optimizer: Callable[..., Optimizer],
        scheduler: Optional[Callable[..., Any]] = None,
        criterion: Optional[nn.Module] = None,
        input_keys: Sequence[str] = ("x", "pos", "batch"),
        target_key: str = "label",
        scheduler_interval: str = "epoch",
        param_groups: Optional[Dict[str, Any]] = None,
    ) -> None:
        metric = Accuracy(task="multiclass", num_classes=model.num_classes)
        super().__init__(
            model,
            metric,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            metric_name="accuracy",
            input_keys=input_keys,
            target_key=target_key,
            scheduler_interval=scheduler_interval,
            param_groups=param_groups,
        )


class LitSegmentationModel(LiTModel):
    """LightningModule wrapping a point cloud semantic segmentation model.

    Args:
        model: A segmentation model, typically built via `create_model`.
        optimizer: A callable that takes parameters and returns an optimizer
            (a `_partial_` config target).
        scheduler: An optional callable that takes an optimizer and returns a
            learning-rate scheduler.
        criterion: The loss module; defaults to `CrossEntropyLoss` with the
            given `ignore_index`.
        ignore_index: Label index excluded from the loss and the mIoU metric.
        input_keys: Batch-dict keys passed positionally to the model's forward.
            A dotted key (e.g. `octree.depth`) resolves an attribute.
        target_key: Batch-dict key for the per-point segmentation labels.
        scheduler_interval: Whether the scheduler steps per `"epoch"` or `"step"`.
        param_groups: Optional dict of kwargs forwarded to
            `torch_pointcloud.utils.optim.generate_param_groups` (`layer_matches`,
            `match_types`, `lr_values`, `include_others`). Same shape as MONAI's
            `generate_param_groups`.
        mix_prob: Probability of applying Mix3D (merging scene pairs) on each training batch.
    """

    def __init__(
        self,
        model: SegmentationModel,
        *,
        optimizer: Callable[..., Optimizer],
        scheduler: Optional[Callable[..., Any]] = None,
        criterion: Optional[nn.Module] = None,
        ignore_index: int = -1,
        input_keys: Sequence[str] = ("x", "pos", "batch"),
        target_key: str = "segment",
        scheduler_interval: str = "epoch",
        param_groups: Optional[Dict[str, Any]] = None,
        mix_prob: float = 0.0,
    ) -> None:
        metric = JaccardIndex(task="multiclass", num_classes=model.num_classes, ignore_index=ignore_index)
        if criterion is None:
            criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
        super().__init__(
            model,
            metric,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            metric_name="mIoU",
            input_keys=input_keys,
            target_key=target_key,
            scheduler_interval=scheduler_interval,
            param_groups=param_groups,
        )
        self.mix_prob = mix_prob

    def on_after_batch_transfer(self, batch: Dict[str, Any], dataloader_idx: int) -> Dict[str, Any]:
        """Mix3D: on training batches, merge adjacent scene pairs by halving the packed `batch` index."""
        if self.mix_prob > 0 and self.trainer.training and torch.rand(1).item() < self.mix_prob:
            batch["batch"] = batch["batch"] // 2
        return batch
