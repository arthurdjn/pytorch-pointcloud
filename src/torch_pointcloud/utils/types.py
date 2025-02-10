from pathlib import Path
from typing import Any, Dict, Sequence, TypeVar, Union

PathLike = str | Path
KeyCollection = Union[str, Sequence[str]]

T = TypeVar("T", bound=Any)

DictStr = Dict[str, T]
