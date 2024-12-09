import functools
from typing import Callable, Optional, Tuple

import torch
from torch_scatter import scatter


def create_pool(dim: int = 0, dim_size: Optional[int] = None, pool_type: str = "mean") -> functools.partial:
    return functools.partial(scatter, dim=dim, dim_size=dim_size, reduce=pool_type)


def create_classifier(num_features: int, num_classes: int, pool_type: str) -> Tuple[torch.nn.Module, Callable]:
    fc = torch.nn.Linear(num_features, num_classes)
    pool = create_pool(pool_type=pool_type)
    return fc, pool
