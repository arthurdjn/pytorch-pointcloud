from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Sequence

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from torch_pointcloud.models import create_model
from torch_pointcloud.models._registry import Task
from torch_pointcloud.utils.box3d import nms3d
from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.imports import optional_import
from torch_pointcloud.utils.misc import deep_getattr
from torch_pointcloud.utils.optim import generate_param_groups

if TYPE_CHECKING:
    from lightning.pytorch import LightningModule
else:
    LightningModule, _ = optional_import("lightning.pytorch", "LightningModule")


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
        inputs = (deep_getattr(batch, key) for key in self.hparams["input_keys"])
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
    r"""LightningModule for a 3D object detection model, built from the registry.

    Detection breaks the shared classification/segmentation loop in three places, so this subclass
    overrides them and reuses the base for everything else (model construction, `forward`, optimizer
    wiring, `training_step`):

    - the loss is a factory completed at build time with the model's head-geometry params
      (`num_heading_bin`, `num_size_cluster`, `num_classes`, `mean_sizes`) rather than duplicating them in config;
    - `step` feeds the whole forward output and the batch to the loss, which returns a dict of named
      components (each logged), and reports the total `loss`;
    - the eval steps run the model's raw `decode`, postprocess it (drop boxes below `score_threshold`, then
      per-class 3D `nms3d` at `nms_iou`), and pair the result with the ground-truth boxes for a
      `MetricCallback` (e.g. `MeanAveragePrecision3D`).

    A model is swappable as long as it returns a prediction dict from `forward` and a raw `Detection3D`
    from `decode(output)`, and is paired with a `criterion(output, batch)`.

    Args:
        name: Registered detection model name; built via `create_model(name, task="detection")`.
        criterion: A loss factory completed with the model's head-geometry params, i.e. called as
            `criterion(num_heading_bin=..., num_size_cluster=..., num_classes=..., mean_sizes=...)` (e.g.
            `VoteNetLoss`); its `forward(output, batch)` returns a dict whose `loss` entry is the total to optimize.
        score_threshold: Minimum score to keep a decoded box in the eval postprocess.
        nms_iou: IoU threshold of the per-class 3D NMS in the eval postprocess.
        **kwargs: Forwarded to `create_model` and the base (e.g. `optimizer`, `scheduler`, `input_keys`).
    """

    def __init__(
        self,
        name: str,
        *,
        criterion: Callable[..., nn.Module],
        score_threshold: float = 0.05,
        nms_iou: float = 0.25,
        **kwargs: Any,
    ) -> None:
        super().__init__(name, task="detection", **kwargs)
        # The base registered a placeholder loss; drop it so the factory result (completed with the model's
        # head-geometry params) can take its place, including a non-Module test double.
        del self.criterion
        self.criterion = criterion(
            num_heading_bin=self.model.num_heading_bin,
            num_size_cluster=self.model.num_size_cluster,
            num_classes=self.model.num_classes,
            mean_sizes=self.model.mean_sizes,
        )
        self.score_threshold = score_threshold
        self.nms_iou = nms_iou

    def step(self, batch: Dict[str, Any], stage: str) -> Dict[str, Any]:
        output = self.forward(batch)
        losses = self.criterion(output, batch)
        batch_size = int(batch[DataKeys.BATCH][-1]) + 1
        for key, value in losses.items():
            self.log(f"{stage}/{key}", value, prog_bar=True, batch_size=batch_size)
        return {"output": output, "loss": losses["loss"]}

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, Any]:
        return self._eval_step(batch, "val")

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, Any]:
        return self._eval_step(batch, "test")

    def _eval_step(self, batch: Dict[str, Any], stage: str) -> Dict[str, Any]:
        output = self.step(batch, stage)["output"]
        det = self.model.decode(output)
        keep = det["scores"] > self.score_threshold
        boxes, scores, labels, det_batch = (
            det["boxes"][keep],
            det["scores"][keep],
            det["labels"][keep],
            det["batch"][keep],
        )
        idx = nms3d(boxes, scores, self.nms_iou, labels=labels, batch=det_batch)
        preds = {"boxes": boxes[idx], "scores": scores[idx], "labels": labels[idx], "batch": det_batch[idx]}
        # GT boxes store half-extents; the metric wants full edge lengths (matches the benchmark examples).
        box = batch[DataKeys.BOX]
        gt_boxes = torch.cat([box[:, :3], 2 * box[:, 3:6], box[:, 6:7]], dim=1)
        target = {"boxes": gt_boxes, "labels": box[:, 7].long(), "batch": batch[DataKeys.BATCH_BOX]}
        return {"preds": preds, "target": target}
