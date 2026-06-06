from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Sequence

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from torch_pointcloud.models import create_model
from torch_pointcloud.models._registry import Task
from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.imports import optional_import
from torch_pointcloud.utils.optim import generate_param_groups

if TYPE_CHECKING:
    from lightning.pytorch import LightningModule
else:
    LightningModule, _ = optional_import("lightning.pytorch", "LightningModule")


def _resolve_input(batch: Dict[str, Any], key: str) -> Any:
    """Resolve a batch entry, supporting dotted attribute access (e.g. `octree.depth`)."""
    name, _, attr = key.partition(".")
    value = batch.get(name)
    return getattr(value, attr) if attr else value


class LiTModel(LightningModule):
    """Shared base for the task-specific Lightning wrappers.

    The task-specific subclasses build their model through `create_model`; this base only holds the
    built model (and its registered evaluation transform) and implements the shared train/val/test
    loop. `input_keys`, `target_key` and `scheduler_interval` are read from `self.hparams`, which the
    subclass populates via `save_hyperparameters`.
    """

    def __init__(
        self,
        name: str,
        task: Task,
        *,
        optimizer: Callable[..., Optimizer],
        scheduler: Optional[Callable[..., Any]] = None,
        criterion: Optional[nn.Module] = None,
        input_keys: Sequence[str] = ("x", "pos", "batch"),
        target_key: str = "label",
        scheduler_interval: str = "epoch",
        param_groups: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        model, info = create_model(name, task=task, return_info=True, **kwargs)

        self.model = model
        self.transform = info["transforms"]
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._param_groups = param_groups

        self.save_hyperparameters(
            {
                "name": name,
                "input_keys": list(input_keys),
                "target_key": target_key,
                "scheduler_interval": scheduler_interval,
                **info["hparams"],
                **kwargs,
            }
        )

    def forward(self, batch: Dict[str, Any]) -> Tensor:
        inputs = (_resolve_input(batch, key) for key in self.hparams["input_keys"])
        return self.model(*inputs)

    def step(self, batch: Dict[str, Any], stage: str) -> Dict[str, Tensor]:
        logits = self.forward(batch)
        target = batch[self.hparams["target_key"]].long()

        loss = self.criterion(logits, target)
        batch_size = int(batch[DataKeys.BATCH][-1]) + 1
        self.log(f"{stage}/loss", loss, prog_bar=True, batch_size=batch_size)
        return {"preds": logits, "target": target, "loss": loss}

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Tensor:
        return self.step(batch, "train")["loss"]

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, Tensor]:
        outputs = self.step(batch, "val")
        return {"preds": outputs["preds"], "target": outputs["target"]}

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, Tensor]:
        outputs = self.step(batch, "test")
        return {"preds": outputs["preds"], "target": outputs["target"]}

    def configure_optimizers(self) -> Any:
        params = generate_param_groups(self, **self._param_groups) if self._param_groups else self.parameters()
        optimizer = self._optimizer(params)
        if self._scheduler is None:
            return optimizer

        scheduler = {"scheduler": self._scheduler(optimizer), "interval": self.hparams["scheduler_interval"]}
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


class LitClassificationModel(LiTModel):
    """LightningModule for a point cloud classification model, built from the registry.

    Args:
        name: Registered classification model name (e.g. `pointnet2-yanx27-ssg.modelnet40`); built via
            `create_model(name, task="classification")`.
        optimizer: A callable that takes parameters and returns an optimizer (a `_partial_` target).
        scheduler: An optional callable that takes an optimizer and returns a learning-rate scheduler.
        criterion: The loss module; defaults to `CrossEntropyLoss`.
        input_keys: Batch-dict keys passed positionally to the model's forward. A dotted key
            (e.g. `octree.depth`) resolves an attribute.
        target_key: Batch-dict key for the per-cloud label.
        scheduler_interval: Whether the scheduler steps per `"epoch"` or `"step"`.
        param_groups: Optional dict of kwargs forwarded to
            `torch_pointcloud.utils.optim.generate_param_groups`.
        **kwargs: Forwarded to `create_model` (e.g. `pretrained=True`, or registry-hparam overrides).
    """

    def __init__(self, name: str, **kwargs: Any) -> None:
        super().__init__(name, task="classification", **kwargs)


class LitSegmentationModel(LiTModel):
    """LightningModule for a point cloud semantic segmentation model, built from the registry.

    Args:
        name: Registered segmentation model name; built via `create_model(name, task="segmentation")`.
        mix_prob: Probability of applying Mix3D (merging scene pairs) on each training batch.
        **kwargs: Forwarded to `create_model` (e.g. `pretrained=True`, or registry-hparam overrides).
    """

    def __init__(self, name: str, mix_prob: float = 0.0, **kwargs: Any) -> None:
        super().__init__(name, task="segmentation", **kwargs)
        self.mix_prob = mix_prob

    def on_after_batch_transfer(self, batch: Dict[str, Any], dataloader_idx: int) -> Dict[str, Any]:
        """Mix3D: on training batches, merge adjacent scene pairs by halving the packed `batch` index."""
        if self.mix_prob > 0 and self.trainer.training and torch.rand(1).item() < self.mix_prob:
            batch["batch"] = batch["batch"] // 2

        return batch


class LitDetectionModel(LiTModel):
    r"""LightningModule for a VoteNet-style 3D object detection model, built from the registry.

    The model returns a dense per-proposal prediction dict and the multi-task `VoteNetLoss` consumes it
    together with the dense padded ground-truth in the batch. The validation step returns the forward
    output so `DetectionMeanAPCallback` can decode it and log mean average precision; attach that
    callback to report `val/mAP@{t}`.

    Args:
        name: Registered detection model name; built via `create_model(name, task="detection")` (must
            expose `num_classes` and a `mean_sizes` buffer).
        **kwargs: Forwarded to `create_model` (e.g. `pretrained=True`, or registry-hparam overrides).
    """

    def __init__(self, name: str, **kwargs: Any) -> None:
        super().__init__(name, task="detection", **kwargs)
