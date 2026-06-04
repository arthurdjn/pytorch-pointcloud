from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import lightning.pytorch as L
import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torchmetrics import Accuracy, JaccardIndex, Metric

from torch_pointcloud.losses import VoteNetLoss
from torch_pointcloud.models import ClassificationModel, DetectionModel, SegmentationModel
from torch_pointcloud.utils.detection import APCalculator, DatasetConfig, parse_groundtruths, parse_predictions
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


class LitDetectionModel(L.LightningModule):
    r"""LightningModule wrapping a VoteNet-style 3D object detection model.

    The model returns a dense per-proposal prediction dict and the multi-task `VoteNetLoss`
    consumes it together with the dense padded ground-truth in the batch. Validation decodes
    boxes, runs 3D NMS and accumulates mean average precision with an `APCalculator` per IoU
    threshold over the whole epoch.

    Args:
        model: A detection model, typically built via `create_model` (must expose `num_classes`
            and a `mean_size_arr` buffer).
        optimizer: A callable that takes parameters and returns an optimizer (a `_partial_` config target).
        criterion: A callable that takes `mean_size_arr=...` and returns the loss module (a `_partial_`
            `VoteNetLoss` target). The model's `mean_size_arr` is injected here so it is not duplicated in config.
        scheduler: An optional callable that takes an optimizer and returns a learning-rate scheduler.
        input_keys: Batch-dict keys passed positionally to the model's forward. A dotted key
            (e.g. `octree.depth`) resolves an attribute.
        ap_iou_thresholds: IoU thresholds at which mean average precision is reported as `val/mAP@{t}`.
        scheduler_interval: Whether the scheduler steps per `"epoch"` or `"step"`.
        param_groups: Optional dict of kwargs forwarded to
            `torch_pointcloud.utils.optim.generate_param_groups` (`layer_matches`, `match_types`,
            `lr_values`, `include_others`). Same shape as MONAI's `generate_param_groups`.
    """

    def __init__(
        self,
        model: DetectionModel,
        *,
        optimizer: Callable[..., Optimizer],
        criterion: Callable[..., nn.Module],
        scheduler: Optional[Callable[..., Any]] = None,
        input_keys: Sequence[str] = ("x", "pos", "batch"),
        ap_iou_thresholds: Sequence[float] = (0.25, 0.5),
        scheduler_interval: str = "epoch",
        param_groups: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.model = model
        loss = criterion(mean_size_arr=model.get_buffer("mean_size_arr"))
        if not isinstance(loss, VoteNetLoss):
            raise TypeError(f"`criterion` must build a `VoteNetLoss`, got {type(loss).__name__}.")
        self.criterion = loss
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._param_groups = param_groups
        self._dataset_config = DatasetConfig(
            num_class=loss.num_class,
            num_heading_bin=loss.num_heading_bin,
            num_size_cluster=loss.num_size_cluster,
            mean_size_arr=loss.mean_size_arr.detach().cpu().numpy(),
            oriented=loss.num_heading_bin > 1,
        )
        self._ap_calculators: Dict[float, APCalculator] = {}
        self.save_hyperparameters("input_keys", "ap_iou_thresholds", "scheduler_interval")

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Tensor]:
        return self.model(*(_resolve_input(batch, key) for key in self.hparams["input_keys"]))

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Tensor:
        output = self.forward(batch)
        loss_dict = self.criterion(self._densify_seeds(output, batch), batch)
        self._log_losses(loss_dict, "train", batch)
        return loss_dict["loss"]

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        output = self.forward(batch)
        loss_dict = self.criterion(self._densify_seeds(output, batch), batch)
        self._log_losses(loss_dict, "val", batch)
        self._accumulate_ap(output, batch)

    @staticmethod
    def _densify_seeds(output: Dict[str, Tensor], batch: Dict[str, Any]) -> Dict[str, Tensor]:
        r"""Reshape the model's packed seed / vote tensors to the dense $(B, S, \cdot)$ the loss expects.

        Proposal tensors are already dense $(B, K, \cdot)$. The seeds are a fixed count per scene, so the
        packed `seed_pos` / `vote_pos` / `seed_inds` (and their batch vectors) reshape by the batch size.
        The model emits `seed_inds` as global indices into the packed points; the loss gathers per-scene
        `vote_label` $(B, N, \cdot)$, so they are localised to $[0, N)$ by subtracting each scene's offset.
        """
        batch_idx: Tensor = batch["batch"]
        batch_size = int(batch_idx[-1]) + 1
        num_points = batch_idx.shape[0] // batch_size
        dense = dict(output)
        for key in ("seed_pos", "vote_pos", "seed_inds", "seed_batch", "vote_batch"):
            tensor = output[key]
            dense[key] = tensor.reshape(batch_size, tensor.shape[0] // batch_size, *tensor.shape[1:])
        dense["seed_inds"] = dense["seed_inds"] - dense["seed_batch"] * num_points
        return dense

    def _log_losses(self, loss_dict: Dict[str, Tensor], stage: str, batch: Dict[str, Any]) -> None:
        batch_size = int(batch["batch"][-1]) + 1
        for name, value in loss_dict.items():
            self.log(f"{stage}/{name}", value, prog_bar=name in ("loss", "obj_acc"), batch_size=batch_size)

    def on_validation_epoch_start(self) -> None:
        self._ap_calculators = {float(t): APCalculator(float(t)) for t in self.hparams["ap_iou_thresholds"]}

    def on_validation_epoch_end(self) -> None:
        for threshold, calculator in self._ap_calculators.items():
            mean_ap, _ = calculator.compute()
            self.log(f"val/mAP@{threshold}", mean_ap, prog_bar=True)

    def _accumulate_ap(self, output: Dict[str, Tensor], batch: Dict[str, Any]) -> None:
        point_clouds = self._dense_point_clouds(batch)
        batch_pred = parse_predictions(output, point_clouds, self._dataset_config)
        batch_gt = self._dense_groundtruths(batch)
        for calculator in self._ap_calculators.values():
            calculator.step(batch_pred, batch_gt)

    def _dense_point_clouds(self, batch: Dict[str, Any]) -> Tensor:
        r"""Reshape the packed batch to dense $(B, N, 3 + C)$ (xyz first, then features)."""
        pos: Tensor = batch["pos"]
        batch_idx: Tensor = batch["batch"]
        bsize = int(batch_idx[-1]) + 1
        num_points = pos.shape[0] // bsize
        x = batch.get("x")
        point_clouds = pos if x is None else torch.cat([pos, x], dim=-1)
        return point_clouds.reshape(bsize, num_points, point_clouds.shape[-1])

    def _dense_groundtruths(self, batch: Dict[str, Any]) -> List[List[Tuple[int, np.ndarray]]]:
        center_label = batch["center_label"].detach().cpu().numpy()
        size_class_label = batch["size_class_label"].detach().cpu().numpy()
        size_residual_label = batch["size_residual_label"].detach().cpu().numpy()
        heading_class_label = batch["heading_class_label"].detach().cpu().numpy()
        heading_residual_label = batch["heading_residual_label"].detach().cpu().numpy()
        sem_cls_label = batch["sem_cls_label"].detach().cpu().numpy()
        box_label_mask = batch["box_label_mask"].detach().cpu().numpy()
        return [
            parse_groundtruths(
                center_label[i],
                size_class_label[i],
                size_residual_label[i],
                heading_class_label[i],
                heading_residual_label[i],
                sem_cls_label[i],
                box_label_mask[i],
                self._dataset_config,
            )
            for i in range(center_label.shape[0])
        ]

    def configure_optimizers(self) -> Any:
        params = generate_param_groups(self, **self._param_groups) if self._param_groups else self.parameters()
        optimizer = self._optimizer(params)
        if self._scheduler is None:
            return optimizer
        scheduler = {"scheduler": self._scheduler(optimizer), "interval": self.hparams["scheduler_interval"]}
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
