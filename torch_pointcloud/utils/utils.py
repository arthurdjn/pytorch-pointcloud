import random
from typing import Any, List, Optional, Sequence, SupportsFloat, Union

import numpy as np
import torch
from torch import Tensor


def aslist(value: Any) -> List[Any]:
    if value is None:
        return []
    elif isinstance(value, (list, tuple)):
        return list(value)
    elif isinstance(value, Tensor):
        return value.tolist()
    elif isinstance(value, np.ndarray):
        return value.tolist()
    return [value]


def is_tensor(value: Any) -> bool:
    return isinstance(value, Tensor)


def to_tensor(value: Any) -> Tensor:
    if is_tensor(value):
        return value
    elif isinstance(value, (list, tuple)):
        return torch.tensor(value)
    elif isinstance(value, np.ndarray):
        return torch.from_numpy(value)
    return torch.tensor(value)


def default_vector(vector: Any, size: int = 1, default_value: int = 0) -> Tensor:
    vector = vector if vector is not None else default_value
    vector = aslist(vector)
    vector = torch.tensor(vector).flatten()
    vector = vector.repeat(size) if vector.size(0) == 1 else vector
    return vector


def sequence_tensor(
    value: Union[None, SupportsFloat, Sequence[SupportsFloat], torch.Tensor, np.ndarray],
    length: int = 1,
    default: Optional[Union[SupportsFloat, Sequence[SupportsFloat]]] = None,
) -> Tensor:
    if value is None and default is None:
        raise ValueError("Either value or default_value must be provided")
    elif value is None:
        value = default

    value = aslist(value)
    tensor = torch.tensor(value).flatten()
    tensor = tensor.repeat(length) if tensor.size(0) == 1 else tensor
    return tensor


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
