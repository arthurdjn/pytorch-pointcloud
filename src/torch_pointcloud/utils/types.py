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
    """Packed 3D boxes with class and scene indices (e.g. detection ground truth, PyG batch layout)."""

    boxes: Tensor  # (N, 7) as (cx, cy, cz, dx, dy, dz, heading)
    labels: Tensor  # (N,)
    batch: Tensor  # (N,) per-box scene index
    ignore_mask: NotRequired[Tensor]  # (N,) bool; True boxes are ignore regions (suppress FP, excluded from GT)


class Detection3D(Boxes3D):
    """Packed 3D detections: `Boxes3D` plus a per-box confidence score (a model's `decode` output)."""

    scores: Tensor  # (N,)


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
