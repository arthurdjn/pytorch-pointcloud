"""Lightning modules for classification, segmentation, and detection training."""

from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Sequence, Union

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from torch_pointcloud.inferers import Inferer, SimpleInferer
from torch_pointcloud.models import create_model
from torch_pointcloud.models._registry import Task
from torch_pointcloud.utils.box3d import count_points_in_boxes, nms3d, projected_ignore_mask
from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.imports import _LIGHTNING_GITHUB_URL, optional_import
from torch_pointcloud.utils.misc import deep_getattr
from torch_pointcloud.utils.optim import generate_param_groups
from torch_pointcloud.utils.types import Boxes3D

if TYPE_CHECKING:
    from lightning.pytorch import LightningModule
else:
    LightningModule, _ = optional_import("lightning.pytorch", "LightningModule", url=_LIGHTNING_GITHUB_URL)

_MISSING = object()


class LitModel(LightningModule):
    """Shared base for the task-specific Lightning wrappers.

    The task-specific subclasses build their model through `create_model`; this base only holds the
    built model (and its registered evaluation transform) and implements the shared train/val/test
    loop. `input_keys`, `target_key` and `scheduler_interval` are read from `self.hparams`, which the
    subclass populates via `save_hyperparameters`. Without an `optimizer` the module is evaluation-only
    (benchmark mode): `Trainer.test` works and `Trainer.fit` raises.

    Args:
        name: Registered model name; built via `create_model(name, task=...)`.
        task: Which task head to build (`"classification"`, `"segmentation"`, `"detection"`); set by the subclass.
        optimizer: A callable that takes parameters and returns an optimizer (a `_partial_` target).
        scheduler: An optional callable that takes an optimizer and returns a learning-rate scheduler.
        criterion: The loss module; defaults to `CrossEntropyLoss`.
        inferer: Test-time inference strategy (e.g. `TTAInferer`, `SlidingWindowInferer`) run in place of the
            plain forward on test batches; defaults to `SimpleInferer` (one forward on the whole batch), so
            every test prediction goes through an inferer. Training and validation are unaffected. The
            inferer may return probabilities instead of logits (torchmetrics handles both), so no `test/loss`
            is logged.
        input_keys: Batch-dict keys passed positionally to the model's forward. A dotted key
            (e.g. `octree.depth`) resolves an attribute. A key missing from the batch raises, except
            `x`: a batch without point features resolves it to `None` (models accept `x=None`).
        target_key: Batch-dict key for the per-cloud label.
        metric_input_keys: Batch-dict keys copied as-is into the `validation_step` / `test_step` output
            dict, alongside the predictions and targets, for metrics whose `update` consumes extra inputs
            (`MetricCallback` forwards the keys each metric declares). A listed key missing from the
            batch raises.
        scheduler_interval: Whether the scheduler steps per `"epoch"` or `"step"`.
        param_groups: Optional dict of kwargs forwarded to
            `torch_pointcloud.utils.optim.generate_param_groups`.
        **kwargs: Forwarded to `create_model` (e.g. `pretrained=True`, or registry-hparam overrides).
    """

    def __init__(
        self,
        name: str,
        task: Task,
        *,
        optimizer: Optional[Callable[..., Optimizer]] = None,
        scheduler: Optional[Callable[..., Any]] = None,
        criterion: Optional[nn.Module] = None,
        inferer: Optional[Inferer] = None,
        input_keys: Sequence[str] = ("x", "pos", "batch"),
        target_key: str = "label",
        metric_input_keys: Sequence[str] = (),
        scheduler_interval: str = "epoch",
        param_groups: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        model, info = create_model(name, task=task, return_info=True, **kwargs)

        self.model = model
        self.transform = info["transform"]
        self.criterion: Optional[nn.Module] = criterion if criterion is not None else nn.CrossEntropyLoss()
        # Not a serializable hyperparameter, so it stays out of `save_hyperparameters` (like `criterion`).
        self.inferer: Inferer = inferer if inferer is not None else SimpleInferer()
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._param_groups = param_groups

        self.save_hyperparameters(
            {
                "name": name,
                "input_keys": list(input_keys),
                "target_key": target_key,
                "metric_input_keys": list(metric_input_keys),
                "scheduler_interval": scheduler_interval,
                **info["hparams"],
                **kwargs,
            }
        )

    def forward(self, batch: Dict[str, Any]) -> Union[Tensor, Dict[str, Tensor]]:
        inputs = []
        for key in self.hparams["input_keys"]:
            value = deep_getattr(batch, key, default=_MISSING)
            if value is _MISSING:
                if key != "x":
                    raise KeyError(
                        f"Input key {key!r} not found in the batch (available keys: {sorted(batch)}); "
                        "check the module's `input_keys`."
                    )
                value = None
            inputs.append(value)
        return self.model(*inputs)

    def predict(self, batch: Dict[str, Any]) -> Tensor:
        """Forward the batch and return the logits tensor (the predictor handed to the inferer)."""
        logits = self.forward(batch)
        assert isinstance(logits, Tensor)
        return logits

    def step(self, batch: Dict[str, Any], stage: str) -> Dict[str, Tensor]:
        """Predict the batch, log the criterion loss, and return the logits, targets and loss."""
        logits = self.predict(batch)
        target = batch[self.hparams["target_key"]].long()

        assert self.criterion is not None
        loss = self.criterion(logits, target)
        batch_size = int(batch[DataKeys.BATCH][-1]) + 1
        self.log(f"{stage}/loss", loss, prog_bar=True, batch_size=batch_size, sync_dist=stage != "train")
        return {"preds": logits, "target": target, "loss": loss}

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Tensor:
        """Run the shared `step` and return the loss to optimize."""
        return self.step(batch, "train")["loss"]

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, Any]:
        """Return the evaluation predictions and targets, plus the batch's `metric_input_keys`."""
        return self._attach_metric_inputs(self._eval_step(batch, "val"), batch)

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, Any]:
        """Return the inferer's predictions and targets, plus the batch's `metric_input_keys`."""
        return self._attach_metric_inputs(self._eval_step(batch, "test"), batch)

    def _eval_step(self, batch: Dict[str, Any], stage: str) -> Dict[str, Any]:
        if stage == "test":
            preds = self.inferer(batch, predictor=self.predict)
            return {"preds": preds, "target": batch[self.hparams["target_key"]].long()}
        outputs = self.step(batch, stage)
        return {"preds": outputs["preds"], "target": outputs["target"]}

    def _attach_metric_inputs(self, outputs: Dict[str, Any], batch: Dict[str, Any]) -> Dict[str, Any]:
        for key in self.hparams["metric_input_keys"]:
            if key not in batch:
                raise KeyError(
                    f"Metric input key {key!r} not found in the batch (available keys: {sorted(batch)}); "
                    "check the module's `metric_input_keys`."
                )
            outputs[key] = batch[key]
        return outputs

    def configure_optimizers(self) -> Any:
        """Build the configured optimizer (over `param_groups` when given) and its optional scheduler."""
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
        **kwargs: Forwarded to the base `LitModel` (e.g. `optimizer`, `scheduler`, `criterion`) and
            `create_model` (e.g. `pretrained=True`, or registry-hparam overrides).
    """

    def __init__(self, name: str, **kwargs: Any) -> None:
        super().__init__(name, task="classification", **kwargs)


class LitSegmentationModel(LitModel):
    """LightningModule for a point cloud semantic segmentation model, built from the registry.

    Args:
        name: Registered segmentation model name; built via `create_model(name, task="segmentation")`.
        inverse_key: Optional batch-dict key holding the voxel-to-raw inverse map. When set, eval
            predictions are broadcast to raw resolution (`preds[batch[inverse_key]]`) and scored against
            `origin_target_key`, the benchmark protocol of grid-sampled models; the loss stays at voxel
            resolution against `target_key`. Multi-scene batches need the key in the loader's `cat_keys`
            so the per-scene maps can be offset into the packed layout. Leave unset when the inferer already
            predicts at raw resolution (e.g. `VoxelPartitionInferer`, `SlidingWindowInferer`).
        origin_target_key: Batch-dict key of the raw-resolution labels used with `inverse_key`.
        target_key: Batch-dict key of the per-point labels.
        **kwargs: Forwarded to the base `LitModel` (e.g. `inferer`, `optimizer`, `criterion`) and
            `create_model` (e.g. `pretrained=True`, or registry-hparam overrides).
    """

    def __init__(
        self,
        name: str,
        inverse_key: Optional[str] = None,
        origin_target_key: str = "origin_segment",
        target_key: str = "segment",
        **kwargs: Any,
    ) -> None:
        super().__init__(name, task="segmentation", target_key=target_key, **kwargs)
        self.save_hyperparameters({"inverse_key": inverse_key, "origin_target_key": origin_target_key})
        self.inverse_key = inverse_key
        self.origin_target_key = origin_target_key

    def forward(self, batch: Dict[str, Any]) -> Tensor:
        out = super().forward(batch)
        assert isinstance(out, Tensor)
        return out

    def _eval_step(self, batch: Dict[str, Any], stage: str) -> Dict[str, Any]:
        outputs = super()._eval_step(batch, stage)
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

    - the loss is either a ready-built `nn.Module` (the general case: an anchor / center / set-matching loss
      whose geometry params are set explicitly in config) or a factory completed at build time with the
      model's head-geometry params (the `VoteNetLoss` carve-out, which reads `num_heading_bin`,
      `num_size_cluster`, `num_classes`, `mean_sizes` off the model); without a `criterion` the module is
      evaluation-only (no loss is logged, training raises);
    - `step` (training only) feeds the whole forward output and the batch to the loss, which returns a dict
      of named components (each logged), and reports the total `loss`;
    - the eval steps are metric-driven, not loss-driven: they run the model's raw `decode`, postprocess it
      (optional `min_points` filter, drop boxes below `score_threshold`, per-class 3D `nms3d` at `nms_iou`
      on the rotated BEV IoU when `nms_rotated`, and the indoor per-class expansion when `decode` emits
      `class_probs`), and pair the result with the
      ground truth for a `MetricCallback` (e.g. `MeanAveragePrecision3D`, `AveragePrecision3D`); any other
      per-box `decode` entry (e.g. the nuScenes heads' `velocity`) is filtered alongside the boxes and kept
      in the predictions dict, and when the batch carries `DataKeys.CALIB` / `DataKeys.IMAGE_SHAPE` (stacked
      per-frame $(B, 3, 4)$ / $(B, 2)$) the sub-25 px `projected_ignore_mask` of the surviving boxes is
      attached as the predictions' `ignore_mask` (the KITTI min-height rule); the ground-truth boxes are
      `DataKeys.BOX` packed as $(K, 7)$ rows $[c_x, c_y, c_z, d_x, d_y, d_z, \theta]$ (full extents,
      counter-clockwise heading $\theta$), with per-box classes under `label_key`. No loss is computed at
      validation, so a two-stage detector whose inference forward differs from its training forward (its
      eval output carries decoded proposals, not loss targets) validates cleanly.

    A model is swappable as long as it returns a prediction dict from `forward` and a raw `Detection3D`
    from `decode(output)`, and (for training) is paired with a `criterion(output, batch)`.

    Args:
        name: Registered detection model name; built via `create_model(name, task="detection")`.
        criterion: The training loss, in one of two forms. A ready-built `nn.Module` is used as-is (the
            general case: instantiate it in config with its geometry params, e.g. `AnchorLoss`). A callable
            factory is completed with the model's head-geometry params, i.e. called as
            `criterion(num_heading_bin=..., num_size_cluster=..., num_classes=..., mean_sizes=...)` (the
            `VoteNetLoss` carve-out). Either way its `forward(output, batch)` returns a dict whose `loss`
            entry is the total to optimize. Leave `None` to benchmark a detector whose training loss is not
            ported.
        score_threshold: Minimum score to keep a decoded box in the eval postprocess.
        nms_iou: IoU threshold of the per-class 3D NMS in the eval postprocess.
        nms_rotated: Suppress on the exact rotated BEV IoU instead of the axis-aligned 3D IoU (see
            `nms3d`); the KITTI outdoor protocol.
        min_points: Optional minimum number of points inside a decoded box for it to be kept (the
            VoteNet / 3DETR indoor protocol uses $5$).
        label_key: Batch-dict key of the per-box ground-truth class labels (defaults to `DataKeys.LABEL`).
        ignore_mask_key: Optional batch-dict key of a per-box ignore mask forwarded to the metric with the
            ground truth (KITTI-style ignore regions).
        **kwargs: Forwarded to `create_model` and the base (e.g. `optimizer`, `scheduler`, `input_keys`). The
            base's `inferer` does not apply: detection eval is the decode / postprocess path above.
    """

    def __init__(
        self,
        name: str,
        *,
        criterion: Union[nn.Module, Callable[..., nn.Module], None] = None,
        score_threshold: float = 0.05,
        nms_iou: float = 0.25,
        nms_rotated: bool = False,
        min_points: Optional[int] = None,
        label_key: str = DataKeys.LABEL,
        ignore_mask_key: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name, task="detection", **kwargs)
        self.save_hyperparameters(
            {
                "score_threshold": score_threshold,
                "nms_iou": nms_iou,
                "nms_rotated": nms_rotated,
                "min_points": min_points,
                "label_key": label_key,
                "ignore_mask_key": ignore_mask_key,
            }
        )
        # The base registered a placeholder loss; drop it so the configured criterion can take its place,
        # including `None` or a non-Module test double.
        del self.criterion
        if criterion is None:
            self.criterion = None
        elif isinstance(criterion, nn.Module):
            self.criterion = criterion
        else:
            self.criterion = criterion(
                num_heading_bin=self.model.num_heading_bin,
                num_size_cluster=self.model.num_size_cluster,
                num_classes=self.model.num_classes,
                mean_sizes=self.model.mean_sizes,
            )
        self.score_threshold = score_threshold
        self.nms_iou = nms_iou
        self.nms_rotated = nms_rotated
        self.min_points = min_points
        self.label_key = label_key
        self.ignore_mask_key = ignore_mask_key

    def step(self, batch: Dict[str, Any], stage: str) -> Dict[str, Any]:
        """Forward the batch, log every named loss component, and return the output and the total loss."""
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
            self.log(f"{stage}/{key}", value, prog_bar=True, batch_size=batch_size, sync_dist=stage != "train")
        return {"output": output, "loss": losses["loss"]}

    def _eval_step(self, batch: Dict[str, Any], stage: str) -> Dict[str, Any]:
        output = self.forward(batch)
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
        idx = nms3d(boxes, scores, self.nms_iou, labels=labels, batch=det_batch, rotated=self.nms_rotated)
        extras: Dict[str, Tensor] = {}
        for key, value in det.items():
            if key in ("boxes", "scores", "labels", "batch", "class_probs"):
                continue
            assert isinstance(value, Tensor)
            extras[key] = value[keep][idx]
        preds: Dict[str, Tensor] = {
            "boxes": boxes[idx],
            "scores": scores[idx],
            "labels": labels[idx],
            "batch": det_batch[idx],
            **extras,
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
                **{key: value.repeat_interleave(num_classes, dim=0) for key, value in extras.items()},
            }
        if DataKeys.CALIB in batch and DataKeys.IMAGE_SHAPE in batch:
            preds["ignore_mask"] = projected_ignore_mask(
                preds["boxes"], batch[DataKeys.CALIB][preds["batch"]], batch[DataKeys.IMAGE_SHAPE][preds["batch"]]
            )
        target: Boxes3D = {
            "boxes": batch[DataKeys.BOX],
            "labels": batch[self.label_key].long(),
            "batch": batch[DataKeys.BATCH_BOX],
        }
        if self.ignore_mask_key is not None:
            target["ignore_mask"] = batch[self.ignore_mask_key]
        return {"preds": preds, "target": target}
