from typing import Any, Callable, Dict, Optional, Sequence

import lightning.pytorch as L
import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from torch_pointcloud.lightning.metrics import boxes_from_packed
from torch_pointcloud.models import ClassificationModel, DetectionModel, SegmentationModel
from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.optim import generate_param_groups
from torch_pointcloud.utils.types import Boxes3D


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
        *,
        optimizer: Callable[..., Optimizer],
        scheduler: Optional[Callable[..., Any]] = None,
        criterion: Optional[nn.Module] = None,
        input_keys: Sequence[str] = ("x", "pos", "batch"),
        target_key: str = "segment",
        scheduler_interval: str = "epoch",
        param_groups: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._param_groups = param_groups
        self.save_hyperparameters("input_keys", "target_key", "scheduler_interval")

    def forward(self, batch: Dict[str, Any]) -> Tensor:
        inputs = (_resolve_input(batch, key) for key in self.hparams["input_keys"])
        return self.model(*inputs)

    def step(self, batch: Dict[str, Any], stage: str) -> Dict[str, Tensor]:
        logits = self.forward(batch)
        target = batch[self.hparams["target_key"]].long()
        loss = self.criterion(logits, target)
        batch_size = int(batch["batch"][-1]) + 1
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
            `match_types`, `lr_values`, `include_others`).
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
        super().__init__(
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
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
            `match_types`, `lr_values`, `include_others`).
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
        if criterion is None:
            criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)

        super().__init__(
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            input_keys=input_keys,
            target_key=target_key,
            scheduler_interval=scheduler_interval,
            param_groups=param_groups,
        )
        self.ignore_index = ignore_index
        self.mix_prob = mix_prob

    def on_after_batch_transfer(self, batch: Dict[str, Any], dataloader_idx: int) -> Dict[str, Any]:
        """Mix3D: on training batches, merge adjacent scene pairs by halving the packed `batch` index."""
        if self.mix_prob > 0 and self.trainer.training and torch.rand(1).item() < self.mix_prob:
            batch["batch"] = batch["batch"] // 2
        return batch


class LitDetectionModel(L.LightningModule):
    r"""LightningModule wrapping a VoteNet-style 3D object detection model.

    The model returns a dense per-proposal prediction dict and the multi-task `VoteNetLoss`
    consumes it together with the dense padded ground-truth in the batch. The validation step returns
    the forward output so `DetectionMeanAPCallback` can decode it and log mean average precision; attach
    that callback to report `val/mAP@{t}`.

    Args:
        model: A detection model, typically built via `create_model` (must expose `num_classes`
            and a `mean_sizes` buffer).
        optimizer: A callable that takes parameters and returns an optimizer (a `_partial_` config target).
        criterion: A callable that takes `mean_sizes=...` and returns the loss module (a `_partial_`
            `VoteNetLoss` target). The model's `mean_sizes` is injected here so it is not duplicated in config.
        scheduler: An optional callable that takes an optimizer and returns a learning-rate scheduler.
        input_keys: Batch-dict keys passed positionally to the model's forward. A dotted key
            (e.g. `octree.depth`) resolves an attribute.
        scheduler_interval: Whether the scheduler steps per `"epoch"` or `"step"`.
        param_groups: Optional dict of kwargs forwarded to
            `torch_pointcloud.utils.optim.generate_param_groups` (`layer_matches`, `match_types`,
            `lr_values`, `include_others`).
        target_transform: Maps the batch's packed `(box, batch_box)` to a `Boxes3D` for the metric;
            defaults to `boxes_from_packed` (the $(K, 8)$ half-extent layout).
        decode_kwargs: Extra keyword arguments forwarded to `model.decode` (e.g. `score_threshold`, `nms_iou`).
    """

    def __init__(
        self,
        model: DetectionModel,
        *,
        optimizer: Callable[..., Optimizer],
        criterion: Callable[..., nn.Module],
        scheduler: Optional[Callable[..., Any]] = None,
        input_keys: Sequence[str] = ("x", "pos", "batch"),
        scheduler_interval: str = "epoch",
        param_groups: Optional[Dict[str, Any]] = None,
        target_transform: Optional[Callable[[Tensor, Tensor], Boxes3D]] = None,
        decode_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.model = model
        loss = criterion(mean_sizes=model.get_buffer("mean_sizes"))

        self.criterion = loss
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._param_groups = param_groups
        self.target_transform = target_transform if target_transform is not None else boxes_from_packed
        self.decode_kwargs = decode_kwargs or {}
        self.save_hyperparameters("input_keys", "scheduler_interval")

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Tensor]:
        inputs = (_resolve_input(batch, key) for key in self.hparams["input_keys"])
        return self.model(*inputs)

    def step(self, batch: Dict[str, Any], stage: str) -> Dict[str, Any]:
        output = self.forward(batch)
        loss = self.criterion(output, batch)

        batch_size = int(batch[DataKeys.BATCH][-1]) + 1
        if isinstance(loss, dict):
            for name, value in loss.items():
                self.log(f"{stage}/{name}", value, prog_bar=name in ("loss", "obj_acc"), batch_size=batch_size)
            return {"output": output, "loss": loss["loss"]}

        self.log(f"{stage}/loss", loss, prog_bar=True, batch_size=batch_size)
        return {"output": output, "loss": loss}

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Tensor:
        loss = self.step(batch, "train")["loss"]
        assert isinstance(loss, Tensor)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, Any]:
        return self._decode_step(batch, "val")

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, Any]:
        return self._decode_step(batch, "test")

    def _decode_step(self, batch: Dict[str, Any], stage: str) -> Dict[str, Any]:
        output = self.step(batch, stage)["output"]
        preds = self.model.decode(output, batch[DataKeys.POS], batch[DataKeys.BATCH], **self.decode_kwargs)
        target = self.target_transform(batch[DataKeys.BOX], batch[DataKeys.BATCH_BOX])
        return {"preds": preds, "target": target}

    def configure_optimizers(self) -> Any:
        params = generate_param_groups(self, **self._param_groups) if self._param_groups else self.parameters()
        optimizer = self._optimizer(params)
        if self._scheduler is None:
            return optimizer

        scheduler = {"scheduler": self._scheduler(optimizer), "interval": self.hparams["scheduler_interval"]}
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
