from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Sequence

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from torch_pointcloud.inferers import Inferer
from torch_pointcloud.lightning.metrics import boxes_from_packed
from torch_pointcloud.models import create_model
from torch_pointcloud.models._registry import Task
from torch_pointcloud.utils.box3d import count_points_in_boxes, nms3d
from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.imports import optional_import
from torch_pointcloud.utils.misc import deep_getattr
from torch_pointcloud.utils.optim import generate_param_groups
from torch_pointcloud.utils.types import Boxes3D, Detection3D

if TYPE_CHECKING:
    from lightning.pytorch import LightningModule
else:
    LightningModule, _ = optional_import("lightning.pytorch", "LightningModule")


class LitModel(LightningModule):
    """Shared base for the task-specific Lightning wrappers.

    The task-specific subclasses build their model through `create_model`; this base only holds the
    built model (and its registered evaluation transform) and implements the shared train/val/test
    loop. `input_keys`, `target_key` and `scheduler_interval` are read from `self.hparams`, which the
    subclass populates via `save_hyperparameters`. Without an `optimizer` the module is evaluation-only
    (benchmark mode): `Trainer.test` works and `Trainer.fit` raises.
    """

    def __init__(
        self,
        name: str,
        task: Task,
        *,
        optimizer: Optional[Callable[..., Optimizer]] = None,
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
        self.transform = info["transform"]
        self.criterion: Optional[nn.Module] = criterion if criterion is not None else nn.CrossEntropyLoss()
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

        assert self.criterion is not None
        loss = self.criterion(logits, target)
        batch_size = int(batch[DataKeys.BATCH][-1]) + 1
        self.log(f"{stage}/loss", loss, prog_bar=True, batch_size=batch_size)
        return {"preds": logits, "target": target, "loss": loss}

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Tensor:
        return self.step(batch, "train")["loss"]

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, Any]:
        return self._eval_step(batch, "val")

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, Any]:
        return self._eval_step(batch, "test")

    def _eval_step(self, batch: Dict[str, Any], stage: str) -> Dict[str, Any]:
        outputs = self.step(batch, stage)
        return {"preds": outputs["preds"], "target": outputs["target"]}

    def configure_optimizers(self) -> Any:
        if self._optimizer is None:
            raise RuntimeError(
                "No `optimizer` was provided, so this module is evaluation-only (benchmark mode); "
                "pass `optimizer=` to train."
            )
        params = generate_param_groups(self, **self._param_groups) if self._param_groups else self.parameters()
        optimizer = self._optimizer(params)
        if self._scheduler is None:
            return optimizer

        scheduler = {"scheduler": self._scheduler(optimizer), "interval": self.hparams["scheduler_interval"]}
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


class LitClassificationModel(LitModel):
    """LightningModule for a point cloud classification model, built from the registry.

    Args:
        name: Registered classification model name (e.g. `pointnet2-ssg.modelnet40.xu-yan`); built via
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


class LitSegmentationModel(LitModel):
    """LightningModule for a point cloud semantic segmentation model, built from the registry.

    Args:
        name: Registered segmentation model name; built via `create_model(name, task="segmentation")`.
        mix_prob: Probability of applying Mix3D (merging scene pairs) on each training batch.
        inferer: Optional test-time inference strategy (e.g. `SlidingWindowInferer`, `TTAInferer`) run in
            place of the plain forward on test batches; training and validation are unaffected. The inferer
            may return probabilities instead of logits (torchmetrics handles both), so no `test/loss` is
            logged on this path.
        inverse_key: Optional batch-dict key holding the voxel-to-raw inverse map. When set, eval
            predictions are broadcast to raw resolution (`preds[batch[inverse_key]]`) and scored against
            `origin_target_key`, the benchmark protocol of grid-sampled models; the loss stays at voxel
            resolution against `target_key`. Multi-scene batches need the key in the loader's `cat_keys`
            so the per-scene maps can be offset into the packed layout.
        origin_target_key: Batch-dict key of the raw-resolution labels used with `inverse_key`.
        **kwargs: Forwarded to `create_model` (e.g. `pretrained=True`, or registry-hparam overrides).
    """

    def __init__(
        self,
        name: str,
        mix_prob: float = 0.0,
        inferer: Optional[Inferer] = None,
        inverse_key: Optional[str] = None,
        origin_target_key: str = "origin_segment",
        **kwargs: Any,
    ) -> None:
        super().__init__(name, task="segmentation", **kwargs)
        self.mix_prob = mix_prob
        self.inferer = inferer
        self.inverse_key = inverse_key
        self.origin_target_key = origin_target_key

    def on_after_batch_transfer(self, batch: Dict[str, Any], dataloader_idx: int) -> Dict[str, Any]:
        """Mix3D: on training batches, merge adjacent scene pairs by halving the packed `batch` index."""
        if self.mix_prob > 0 and self.trainer.training and torch.rand(1).item() < self.mix_prob:
            batch["batch"] = batch["batch"] // 2

        return batch

    def _eval_step(self, batch: Dict[str, Any], stage: str) -> Dict[str, Any]:
        if self.inferer is not None and stage == "test":
            preds = self.inferer(batch, predictor=self.forward)
            target = batch[self.hparams["target_key"]].long()
        else:
            outputs = self.step(batch, stage)
            preds, target = outputs["preds"], outputs["target"]
        point_batch = batch[DataKeys.BATCH]
        if self.inverse_key is not None:
            inverse = self._batched_inverse(batch)
            preds = preds[inverse]
            target = batch[self.origin_target_key].long()
            point_batch = point_batch[inverse]
        return {"preds": preds, "target": target, "batch": point_batch}

    def _batched_inverse(self, batch: Dict[str, Any]) -> Tensor:
        """Per-scene inverse maps index into their own scene's rows; offset them by the rows of the scenes
        collated before (`collate` concatenates the inverse key as-is, and its `cat_keys` scene index says
        which scene each raw point belongs to)."""
        assert self.inverse_key is not None
        inverse = batch[self.inverse_key]
        scene = batch.get(f"batch_{self.inverse_key}")
        if scene is None:
            if int(batch[DataKeys.BATCH][-1]) > 0:
                raise RuntimeError(
                    f"`inverse_key={self.inverse_key!r}` on a multi-scene batch needs the per-point scene "
                    f"index `batch_{self.inverse_key}`; add {self.inverse_key!r} to the datamodule's "
                    "`cat_keys` (or use an eval batch size of 1)."
                )
            return inverse
        counts = torch.bincount(batch[DataKeys.BATCH], minlength=int(scene.max()) + 1)
        offsets = torch.cumsum(counts, dim=0) - counts
        return inverse + offsets[scene]


class LitDetectionModel(LitModel):
    r"""LightningModule for a 3D object detection model, built from the registry.

    Detection breaks the shared classification/segmentation loop in three places, so this subclass
    overrides them and reuses the base for everything else (model construction, `forward`, optimizer
    wiring, `training_step`):

    - the loss is a factory completed at build time with the model's head-geometry params
      (`num_heading_bin`, `num_size_cluster`, `num_classes`, `mean_sizes`) rather than duplicating them in
      config; without a `criterion` the module is evaluation-only (no loss is logged, training raises);
    - `step` feeds the whole forward output and the batch to the loss, which returns a dict of named
      components (each logged), and reports the total `loss`;
    - the eval steps run the model's raw `decode`, postprocess it (optional `min_points` filter, drop boxes
      below `score_threshold`, per-class 3D `nms3d` at `nms_iou`, and the indoor per-class expansion when
      `decode` emits `class_probs`), and pair the result with the ground-truth boxes for a
      `MetricCallback` (e.g. `MeanAveragePrecision3D`, `AveragePrecision3D`).

    A model is swappable as long as it returns a prediction dict from `forward` and a raw `Detection3D`
    from `decode(output)`, and (for training) is paired with a `criterion(output, batch)`.

    Args:
        name: Registered detection model name; built via `create_model(name, task="detection")`.
        criterion: An optional loss factory completed with the model's head-geometry params, i.e. called as
            `criterion(num_heading_bin=..., num_size_cluster=..., num_classes=..., mean_sizes=...)` (e.g.
            `VoteNetLoss`); its `forward(output, batch)` returns a dict whose `loss` entry is the total to
            optimize. Leave `None` to benchmark a detector whose training loss is not ported.
        score_threshold: Minimum score to keep a decoded box in the eval postprocess.
        nms_iou: IoU threshold of the per-class 3D NMS in the eval postprocess.
        min_points: Optional minimum number of points inside a decoded box for it to be kept (the
            VoteNet / 3DETR indoor protocol uses $5$).
        label_key: Optional batch-dict key of per-box ground-truth class labels. When set, ground-truth
            boxes are read as $(K, 7)$ full-extent rows paired with these labels (the KITTI / nuScenes
            convention); when `None`, they are $(K, 8)$ half-extent rows with the class in the last column
            (the SUN RGB-D / ScanNet convention).
        ignore_mask_key: Optional batch-dict key of a per-box ignore mask forwarded to the metric with the
            ground truth (used with `label_key` for KITTI-style ignore regions).
        **kwargs: Forwarded to `create_model` and the base (e.g. `optimizer`, `scheduler`, `input_keys`).
    """

    def __init__(
        self,
        name: str,
        *,
        criterion: Optional[Callable[..., nn.Module]] = None,
        score_threshold: float = 0.05,
        nms_iou: float = 0.25,
        min_points: Optional[int] = None,
        label_key: Optional[str] = None,
        ignore_mask_key: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name, task="detection", **kwargs)
        # The base registered a placeholder loss; drop it so the factory result (completed with the model's
        # head-geometry params) can take its place, including `None` or a non-Module test double.
        del self.criterion
        if criterion is None:
            self.criterion = None
        else:
            self.criterion = criterion(
                num_heading_bin=self.model.num_heading_bin,
                num_size_cluster=self.model.num_size_cluster,
                num_classes=self.model.num_classes,
                mean_sizes=self.model.mean_sizes,
            )
        self.score_threshold = score_threshold
        self.nms_iou = nms_iou
        self.min_points = min_points
        self.label_key = label_key
        self.ignore_mask_key = ignore_mask_key

    def step(self, batch: Dict[str, Any], stage: str) -> Dict[str, Any]:
        output = self.forward(batch)
        if self.criterion is None:
            if stage == "train":
                raise RuntimeError(
                    "No `criterion` was provided, so this module is evaluation-only (benchmark mode); "
                    "pass `criterion=` to train."
                )
            return {"output": output}
        losses = self.criterion(output, batch)
        batch_size = int(batch[DataKeys.BATCH][-1]) + 1
        for key, value in losses.items():
            self.log(f"{stage}/{key}", value, prog_bar=True, batch_size=batch_size)
        return {"output": output, "loss": losses["loss"]}

    def _eval_step(self, batch: Dict[str, Any], stage: str) -> Dict[str, Any]:
        output = self.step(batch, stage)["output"]
        det = self.model.decode(output)
        keep = det["scores"] > self.score_threshold
        if self.min_points is not None:
            counts = count_points_in_boxes(
                batch[DataKeys.POS], det["boxes"], pos_batch=batch[DataKeys.BATCH], box_batch=det["batch"]
            )
            keep &= counts >= self.min_points
        boxes, scores, labels, det_batch = (
            det["boxes"][keep],
            det["scores"][keep],
            det["labels"][keep],
            det["batch"][keep],
        )
        idx = nms3d(boxes, scores, self.nms_iou, labels=labels, batch=det_batch)
        preds: Detection3D = {
            "boxes": boxes[idx],
            "scores": scores[idx],
            "labels": labels[idx],
            "batch": det_batch[idx],
        }
        if "class_probs" in det:
            # Indoor AP convention: score every surviving box against each class by its class probability.
            probs = det["class_probs"][keep][idx]
            num_classes = probs.size(-1)
            preds = {
                "boxes": boxes[idx].repeat_interleave(num_classes, dim=0),
                "scores": (probs * scores[idx, None]).reshape(-1),
                "labels": torch.arange(num_classes, device=probs.device).repeat(idx.numel()),
                "batch": det_batch[idx].repeat_interleave(num_classes),
            }
        box = batch[DataKeys.BOX]
        if self.label_key is not None:
            target: Boxes3D = {"boxes": box, "labels": batch[self.label_key].long(), "batch": batch[DataKeys.BATCH_BOX]}
            if self.ignore_mask_key is not None:
                target["ignore_mask"] = batch[self.ignore_mask_key]
        else:
            target = boxes_from_packed(box, batch[DataKeys.BATCH_BOX])
        return {"preds": preds, "target": target}
