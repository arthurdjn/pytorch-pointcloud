from collections.abc import Iterable
from typing import Any, Tuple

import numpy as np
import torch


def ensure_tuple(value: Any) -> Tuple[Any, ...]:
    if isinstance(value, np.ndarray):
        if value.ndim > 0:
            return tuple(value.tolist())
        return tuple([value.item()])
    elif isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.ndim > 0:
            return tuple(value.tolist())
        return tuple([value.item()])
    elif isinstance(value, (str, bytes)):
        return tuple([value])
    elif isinstance(value, Iterable):
        return tuple(value)
    return tuple([value])
