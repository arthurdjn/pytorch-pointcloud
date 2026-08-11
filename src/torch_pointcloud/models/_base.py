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
    `SegmentationModel` or `DetectionModel`. Those standardize the split into `forward_features`
    (encoder), `forward_decoder` (segmentation only) and `forward_head` (logits or raw
    predictions), plus `reset_classifier` for rebuilding the head (concrete models build it in a
    `configure_head` method called from both `__init__` and `reset_classifier`).

    Args:
        in_channels: Number of input feature channels.
    """

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels


class ClassificationModel(nn.Module, metaclass=ABCMeta):
    r"""Base class for point cloud classification models.

    Subclasses must implement `forward`. The canonical layout splits it into `forward_features` (encode
    the packed cloud) and `forward_head` (pool and map to logits); models built that way override both.

    Args:
        in_channels: Number of input feature channels.
        num_classes: Number of output classes.
    """

    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

    def reset_classifier(self, num_classes: int) -> None:
        r"""Replace the classification head for `num_classes` outputs.

        Concrete models may accept extra keyword options (e.g. `global_pool`, `head_channels`, `dropout`);
        models without a rebuildable head raise `NotImplementedError`.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not implement `reset_classifier`.")

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        r"""Run the model on a packed point cloud.

        Canonical signature: `forward(x, pos, batch)` with features `x` $(N, C)$ (or `None`), positions
        `pos` $(N, 3)$ and batch index `batch` $(N,)$, returning logits $(B, \text{num\_classes})$.
        """

    def forward_features(self, *args: Any, **kwargs: Any) -> Any:
        r"""Encode a packed point cloud into features for `forward_head`.

        Canonical signature: `forward_features(x, pos, batch)`, returning the encoded features (often with
        the downsampled `pos` / `batch`). Models that implement `forward` in one piece do not provide this
        split and raise `NotImplementedError`.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not implement `forward_features`.")

    def forward_head(self, *args: Any, **kwargs: Any) -> Any:
        r"""Map encoded features to class logits.

        Canonical signature: `forward_head(x, batch)`, returning logits $(B, \text{num\_classes})$. Models
        that implement `forward` in one piece do not provide this split and raise `NotImplementedError`.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not implement `forward_head`.")


class SegmentationModel(nn.Module, metaclass=ABCMeta):
    r"""Base class for point cloud semantic segmentation models.

    Subclasses must implement `forward`. The canonical layout splits it into `forward_features` (encoder,
    keeping the skip intermediates), `forward_decoder` (upsampling path) and `forward_head` (per-point
    logits); models built that way override all three.

    Args:
        in_channels: Number of input feature channels.
        num_classes: Number of output classes.
    """

    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

    def reset_classifier(self, num_classes: int) -> None:
        r"""Replace the segmentation head for `num_classes` outputs.

        Concrete models may accept extra keyword options (e.g. `global_pool`, `head_channels`, `dropout`);
        models without a rebuildable head raise `NotImplementedError`.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not implement `reset_classifier`.")

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        r"""Run the model on a packed point cloud.

        Canonical signature: `forward(x, pos, batch)` with features `x` $(N, C)$ (or `None`), positions
        `pos` $(N, 3)$ and batch index `batch` $(N,)$, returning per-point logits
        $(N, \text{num\_classes})$.
        """

    def forward_features(self, *args: Any, **kwargs: Any) -> Any:
        r"""Encode a packed point cloud, keeping what `forward_decoder` needs for the skip connections.

        Canonical signature: `forward_features(x, pos, batch)`, returning the encoder output together with
        the per-stage intermediates. Models that implement `forward` in one piece do not provide this split
        and raise `NotImplementedError`.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not implement `forward_features`.")

    def forward_decoder(self, *args: Any, **kwargs: Any) -> Any:
        r"""Decode encoder features back to per-point resolution.

        Canonical signature: `forward_decoder(x, ..., intermediates)`, consuming the output of
        `forward_features` and returning per-point features $(N, C)$. Models without a distinct decoder do
        not provide this split and raise `NotImplementedError`.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not implement `forward_decoder`.")

    def forward_head(self, *args: Any, **kwargs: Any) -> Any:
        r"""Map per-point features to per-point logits.

        Canonical signature: `forward_head(x)`, returning logits $(N, \text{num\_classes})$. Models that
        implement `forward` in one piece do not provide this split and raise `NotImplementedError`.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not implement `forward_head`.")


class DetectionModel(nn.Module, metaclass=ABCMeta):
    r"""Base class for 3D object detection models.

    Subclasses must implement `forward` (raw prediction output), `forward_features` (the backbone) and
    `decode` (raw output to packed boxes). `decode` is raw: score thresholding and NMS belong to the
    evaluation postprocess, not the model.

    Args:
        in_channels: Number of input feature channels.
        num_classes: Number of object classes.
    """

    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

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

    def forward_head(self, *args: Any, **kwargs: Any) -> Any:
        r"""Map backbone features to the raw prediction output.

        Canonical signature: `forward_head(features)`, returning the same prediction output as `forward`.
        Models that implement `forward` in one piece do not provide this split and raise
        `NotImplementedError`.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not implement `forward_head`.")

    @abstractmethod
    def decode(self, *args: Any, **kwargs: Any) -> Any:
        r"""Decode the raw `forward` output into packed boxes.

        Canonical signature: `decode(output)`, returning a `Detection3D` with `boxes` $(K, 7)$, `scores`
        $(K,)$, `labels` $(K,)$ and `batch` $(K,)$. The decode is raw: no score thresholding and no NMS.
        """
