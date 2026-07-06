from enum import Enum
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Sequence, Tuple, TypedDict, TypeVar, Union

import numpy as np
from torch import Tensor
from typing_extensions import NotRequired

PathLike = Union[str, Path]

T = TypeVar("T", bound=Any)

# Tuple of two elements of type T
Tuple2d = Tuple[T, T]

# Collection of single and sequential values (e.g. a float or a list of floats)
ValueCollection = Union[T, Sequence[T]]
# Collection of keys used in dict-based transforms (e.g. a string or a list of strings)
KeyCollection = ValueCollection[str]

# A shorthand for a dictionary with string keys and values of type T
DictStr = Dict[str, T]

# Array-like (numpy.ndarray or torch.Tensor)
NdarrayOrTensor = Union[np.ndarray, Tensor]

# Optional numpy array (numpy.ndarray or None)
OptNdarray = Union[np.ndarray, None]
# Optional tensor (torch.Tensor or None)
OptTensor = Union[Tensor, None]

# Pair of tensors
PairTensor = Tuple2d[Tensor]
PairOptTensor = Tuple2d[OptTensor]


class Boxes3D(TypedDict):
    """Packed 3D boxes with class and scene indices (e.g. detection ground truth, PyG batch layout).

    Args:
        boxes: Boxes $(N, 7)$ of $(c_x, c_y, c_z, d_x, d_y, d_z, \theta)$.
        labels: Per-box class, shape $(N,)$.
        batch: Per-box scene index, shape $(N,)$.
        ignore_mask: Per-box ignore mask, shape $(N,)$.
    """

    boxes: Tensor
    labels: Tensor
    batch: Tensor
    ignore_mask: NotRequired[Tensor]


class Detection3D(Boxes3D):
    """Packed 3D detections: `Boxes3D` plus a per-box confidence score (a model's `decode` output).

    Args:
        boxes: Boxes $(N, 7)$ of $(c_x, c_y, c_z, d_x, d_y, d_z, \theta)$.
        labels: Per-box class, shape $(N,)$.
        batch: Per-box scene index, shape $(N,)$.
        ignore_mask: Per-box ignore mask, shape $(N,)$.
        scores: Per-box confidence score, shape $(N,)$.
        class_probs: Per-box class probabilities, shape $(N, C)$, emitted by detectors whose eval protocol
            expands every kept box across all classes (e.g. VoteNet / 3DETR).
    """

    scores: Tensor
    class_probs: NotRequired[Tensor]


# Flow and aggregation types for message passing
FlowType = Literal["source_to_target", "target_to_source"]
AggrType = Literal["add", "mean", "max"]


class MessagePassingParams(TypedDict):
    aggr: NotRequired[AggrType]
    aggr_kwargs: NotRequired[Optional[Dict[str, Any]]]
    flow: NotRequired[FlowType]
    node_dim: NotRequired[int]
    decomposed_layers: NotRequired[int]


class StrEnum(str, Enum):
    def __str__(self) -> str:
        return self.value
