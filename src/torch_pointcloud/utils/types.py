from pathlib import Path
from typing import Any, Dict, Optional, Sequence, TypeVar, Union

from torch import Tensor

PathLike = str | Path
KeyCollection = Union[str, Sequence[str]]

T = TypeVar("T", bound=Any)

DictStr = Dict[str, T]
OptTensor = Optional[Tensor]
