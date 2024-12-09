import functools
from typing import Optional

from torch_scatter import scatter


def create_pool(dim: int = 0, dim_size: Optional[int] = None, reduce: str = "mean") -> functools.partial:
    return functools.partial(scatter, dim=dim, dim_size=dim_size, reduce=reduce)
