from pathlib import Path
from typing import Any, Dict, Literal, Optional, Sequence, Tuple, TypedDict, TypeVar, Union

from torch import Tensor
from typing_extensions import NotRequired

PathLike = Union[str, Path]

T = TypeVar("T", bound=Any)

Tuple2d = Tuple[T, T]
ValueCollection = Union[T, Sequence[T]]
KeyCollection = ValueCollection[str]
DictStr = Dict[str, T]

OptTensor = Union[Tensor, None]
PairTensor = Tuple2d[Tensor]
PairOptTensor = Tuple2d[OptTensor]

FlowType = Literal["source_to_target", "target_to_source"]
AggrType = Literal["add", "mean", "max"]


class MessagePassingParams(TypedDict):
    aggr: NotRequired[AggrType]
    aggr_kwargs: NotRequired[Optional[Dict[str, Any]]]
    flow: NotRequired[FlowType]
    node_dim: NotRequired[int]
    decomposed_layers: NotRequired[int]
