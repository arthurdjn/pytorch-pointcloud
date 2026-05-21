from typing import Any, Callable, Dict, Optional, Sequence, Union

import torch
import torch.nn as nn
from torch_geometric.nn import MLP


def create_cls_head(num_features: int, num_classes: int) -> torch.nn.Module:
    if num_classes == 0:
        return nn.Identity()

    return nn.Linear(num_features, num_classes)


def create_seg_head(
    dims: Sequence[int],
    num_classes: int,
    act: Union[str, Callable, None] = "relu",
    act_kwargs: Optional[Dict[str, Any]] = None,
    norm: Union[str, Callable, None] = "batch_norm",
    norm_kwargs: Optional[Dict[str, Any]] = None,
    dropout: float = 0.0,
) -> torch.nn.Module:
    if not dims or num_classes == 0:
        return nn.Identity()

    return MLP(
        [*dims[:-1], num_classes],
        act=act,
        act_kwargs=act_kwargs,
        norm=norm,
        norm_kwargs=norm_kwargs,
        dropout=dropout,
        act_first=True,
        plain_last=True,
    )
