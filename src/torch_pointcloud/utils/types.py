"""Shared type aliases, `TypedDict` definitions, and enums used across the package."""

from enum import Enum
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Sequence, Tuple, TypedDict, TypeVar, Union

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

# Optional tensor (torch.Tensor or None)
OptTensor = Union[Tensor, None]

# Pair of tensors
PairTensor = Tuple2d[Tensor]
PairOptTensor = Tuple2d[OptTensor]


class Boxes3D(TypedDict):
    r"""Packed 3D boxes with class and scene indices (e.g. detection ground truth, PyG batch layout).

    Attributes:
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
    r"""Packed 3D detections: `Boxes3D` plus a per-box confidence score (a model's `decode` output).

    Attributes:
        boxes: Boxes $(N, 7)$ of $(c_x, c_y, c_z, d_x, d_y, d_z, \theta)$.
        labels: Per-box class, shape $(N,)$.
        batch: Per-box scene index, shape $(N,)$.
        ignore_mask: Per-box ignore mask, shape $(N,)$.
        scores: Per-box confidence score, shape $(N,)$.
        class_probs: Per-box class probabilities, shape $(N, C)$, emitted by detectors whose eval protocol
            expands every kept box across all classes (e.g. VoteNet / 3DETR).
        velocity: Per-box BEV velocity $(v_x, v_y)$, shape $(N, 2)$, emitted by detectors whose head
            regresses velocity (the nuScenes heads).
    """

    scores: Tensor
    class_probs: NotRequired[Tensor]
    velocity: NotRequired[Tensor]


# Flow and aggregation types for message passing
FlowType = Literal["source_to_target", "target_to_source"]
AggrType = Literal["add", "mean", "max"]


class MessagePassingParams(TypedDict):
    """Keyword arguments a message-passing layer forwards to its `MessagePassing` base class."""

    aggr: NotRequired[AggrType]
    aggr_kwargs: NotRequired[Optional[Dict[str, Any]]]
    flow: NotRequired[FlowType]
    node_dim: NotRequired[int]
    decomposed_layers: NotRequired[int]


class StrEnum(str, Enum):
    """Enum whose members are plain strings, so they compare, print and hash like their value."""

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        # Members are used as dict keys next to plain strings (e.g. `collate` inserting `DataKeys.BATCH`);
        # the default `<DataKeys.BATCH: 'batch'>` repr makes such dicts print inconsistently.
        return repr(self.value)
