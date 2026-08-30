"""Abstract base classes for classification, segmentation, and detection models."""

from abc import ABCMeta, abstractmethod
from typing import Any

import torch.nn as nn


class BaseModel(nn.Module, metaclass=ABCMeta):
    r"""Base class for task-agnostic point cloud models, registered under the `"base"` task.

    Models whose output is neither logits nor detections inherit from this class: self-supervised
    pretraining models (masked autoencoders, generative pretraining) and generative models, whose
    `forward` returns pretext outputs such as `(pred, target)` group coordinates for a
    reconstruction objective. The class only stores `in_channels`; each subclass defines its own
    `forward` signature.

    Models with a task head inherit from one of the task ABCs instead: `ClassificationModel`,
    `SegmentationModel` or `DetectionModel`. The classification and segmentation ABCs require the
    split into `forward_features` (encoder), `forward_decoder` (segmentation only, optional) and
    `forward_head` (logits), plus `configure_head` / `reset_classifier` for building the head and
    the read-only `num_features` property for the width of the features entering it. Detection models
    split into `forward_features` (backbone), `forward_head` (raw predictions) and `decode` (boxes), and
    expose `num_features` as the backbone output width.

    Args:
        in_channels: Number of input feature channels.
    """

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels


class ClassificationModel(nn.Module, metaclass=ABCMeta):
    r"""Base class for point cloud classification models.

    Subclasses implement `forward_features` (encode the packed cloud), `forward_head` (pool and map to
    logits), `configure_head` and `reset_classifier`, and compose the first two in `forward`. Building a
    model with `num_classes=0` makes `configure_head` return `nn.Identity`, so `forward` returns the pooled
    features $(B, \text{num\_features})$; `forward_head(..., pre_logits=True)` returns those same features
    for any `num_classes`.

    Args:
        in_channels: Number of input feature channels.
        num_classes: Number of output classes; `0` drops the head.
    """

    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

    @property
    @abstractmethod
    def num_features(self) -> int:
        r"""Width $C$ of the pooled features entering the head.

        The last dimension of `forward(...)` when `num_classes=0` and of `forward_head(..., pre_logits=True)`.
        Read-only: it is derived from the instantiated submodules.
        """

    @abstractmethod
    def configure_head(self) -> nn.Module:
        r"""Build and return the classification head for the current `num_classes` (`nn.Identity` when 0).

        Called from both `__init__` and `reset_classifier` so the two always build the same module. The head
        is returned in the module's current training mode, so rebuilding it on an `eval()` model does not
        put BatchNorm and dropout back into train mode.
        """

    @abstractmethod
    def reset_classifier(self, num_classes: int) -> None:
        r"""Replace the classification head for `num_classes` outputs (`0` drops it).

        Concrete models may accept extra keyword options (e.g. `global_pool`, `head_channels`, `dropout`).
        """

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        r"""Run the model on a packed point cloud.

        Canonical signature: `forward(x, pos, batch)` with features `x` $(N, C)$ (or `None`), positions
        `pos` $(N, 3)$ and batch index `batch` $(N,)$, returning logits $(B, \text{num\_classes})$, or the
        pooled features $(B, \text{num\_features})$ when `num_classes=0`.
        """

    @abstractmethod
    def forward_features(self, *args: Any, **kwargs: Any) -> Any:
        r"""Encode a packed point cloud into features for `forward_head`.

        Canonical signature: `forward_features(x, pos, batch)`, returning the encoded features (often with
        the downsampled `pos` / `batch`).
        """

    @abstractmethod
    def forward_head(self, *args: Any, **kwargs: Any) -> Any:
        r"""Map encoded features to class logits.

        Canonical signature: `forward_head(x, batch, pre_logits=False)`, returning logits
        $(B, \text{num\_classes})$, or the pooled features $(B, \text{num\_features})$ when `pre_logits=True`.
        """


class SegmentationModel(nn.Module, metaclass=ABCMeta):
    r"""Base class for point cloud semantic segmentation models.

    Subclasses implement `forward_features` (encoder, keeping the skip intermediates), `forward_head`
    (per-point logits), `configure_head` and `reset_classifier`, and compose them in `forward`, usually
    through `forward_decoder` (upsampling path). Building a model with `num_classes=0` makes
    `configure_head` return `nn.Identity`, so `forward` returns the per-point features
    $(N, \text{num\_features})$; `forward_head(..., pre_logits=True)` returns those same features for any
    `num_classes`.

    Args:
        in_channels: Number of input feature channels.
        num_classes: Number of output classes; `0` drops the head.
    """

    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

    @property
    @abstractmethod
    def num_features(self) -> int:
        r"""Width $C$ of the per-point features entering the head.

        The last dimension of `forward(...)` when `num_classes=0` and of `forward_head(..., pre_logits=True)`.
        Read-only: it is derived from the instantiated submodules.
        """

    @abstractmethod
    def configure_head(self) -> nn.Module:
        r"""Build and return the segmentation head for the current `num_classes` (`nn.Identity` when 0).

        Called from both `__init__` and `reset_classifier` so the two always build the same module. The head
        is returned in the module's current training mode, so rebuilding it on an `eval()` model does not
        put BatchNorm and dropout back into train mode.
        """

    @abstractmethod
    def reset_classifier(self, num_classes: int) -> None:
        r"""Replace the segmentation head for `num_classes` outputs (`0` drops it).

        Concrete models may accept extra keyword options (e.g. `global_pool`, `head_channels`, `dropout`).
        """

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        r"""Run the model on a packed point cloud.

        Canonical signature: `forward(x, pos, batch)` with features `x` $(N, C)$ (or `None`), positions
        `pos` $(N, 3)$ and batch index `batch` $(N,)$, returning per-point logits
        $(N, \text{num\_classes})$, or the per-point features $(N, \text{num\_features})$ when
        `num_classes=0`.
        """

    @abstractmethod
    def forward_features(self, *args: Any, **kwargs: Any) -> Any:
        r"""Encode a packed point cloud, keeping what `forward_decoder` needs for the skip connections.

        Canonical signature: `forward_features(x, pos, batch)`, returning the encoder output together with
        the per-stage intermediates.
        """

    def forward_decoder(self, *args: Any, **kwargs: Any) -> Any:
        r"""Decode encoder features back to per-point resolution.

        Canonical signature: `forward_decoder(x, ..., intermediates)`, consuming the output of
        `forward_features` and returning per-point features $(N, C)$. Models whose encoder already emits
        per-point features (DGCNN, PointNet) have no decoder and raise `NotImplementedError`.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not implement `forward_decoder`.")

    @abstractmethod
    def forward_head(self, *args: Any, **kwargs: Any) -> Any:
        r"""Map per-point features to per-point logits.

        Canonical signature: `forward_head(x, pre_logits=False)`, returning logits $(N, \text{num\_classes})$,
        or the per-point features $(N, \text{num\_features})$ when `pre_logits=True`.
        """


class DetectionModel(nn.Module, metaclass=ABCMeta):
    r"""Base class for 3D object detection models.

    Subclasses implement `forward_features` (the backbone), `forward_head` (backbone features to the raw
    prediction output), `decode` (raw output to packed boxes) and `num_features`, and compose the first two
    in `forward`. `decode` is raw: score thresholding and NMS belong to the evaluation postprocess, not the
    model. Detectors have no headless mode: feature maps come from `forward_features`, and `reset_classifier`
    is only available where the class count can change without touching the box branches.

    Args:
        in_channels: Number of input feature channels.
        num_classes: Number of object classes.
    """

    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

    @property
    @abstractmethod
    def num_features(self) -> int:
        r"""Channel count $C$ of the `forward_features` output.

        The BEV map channels, the sparse voxel feature width, or the packed point feature width, depending
        on the backbone. Read-only: it is derived from the instantiated submodules.
        """

    def reset_classifier(self, num_classes: int) -> None:
        r"""Replace the classification branch of the detection head for `num_classes` outputs.

        Models whose head is not rebuildable in isolation raise `NotImplementedError`.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not implement `reset_classifier`.")

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        r"""Run the model and return its raw prediction output.

        Canonical signature: `forward(x, pos, batch)` for point-based detectors (voxel detectors take
        their voxelized layout instead), returning the prediction dict consumed by the training criterion
        and by `decode`.
        """

    @abstractmethod
    def forward_features(self, *args: Any, **kwargs: Any) -> Any:
        r"""Run the backbone on the packed input.

        Canonical signature: `forward_features(x, pos, batch)` for point-based detectors (voxel detectors
        take their voxelized layout instead), returning the backbone features fed to the detection head.
        """

    @abstractmethod
    def forward_head(self, *args: Any, **kwargs: Any) -> Any:
        r"""Map backbone features to the raw prediction output.

        Canonical signature: `forward_head(features)`, returning the same prediction output as `forward`.
        Heads that need more than the backbone output take it as extra arguments (VoteNet seed indices,
        3DETR scene extents, PointRCNN ground truth in train mode).
        """

    @abstractmethod
    def decode(self, *args: Any, **kwargs: Any) -> Any:
        r"""Decode the raw `forward` output into packed boxes.

        Canonical signature: `decode(output)`, returning a `Detection3D` with `boxes` $(K, 7)$, `scores`
        $(K,)$, `labels` $(K,)$ and `batch` $(K,)$. The decode is raw: no score thresholding and no NMS.
        """
