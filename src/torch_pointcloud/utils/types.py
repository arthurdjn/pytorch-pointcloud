from pathlib import Path
from typing import Any, Dict, Sequence, TypeVar, Union

from torch import Tensor

PathLike = Union[str, Path]

T = TypeVar("T", bound=Any)

ValueCollection = Union[T, Sequence[T]]
KeyCollection = ValueCollection[str]
DictStr = Dict[str, T]
OptTensor = Union[Tensor, None]
