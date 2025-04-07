import itertools
from typing import List, Sequence

import torch
import torch.nn as nn

from .activations import ActLike
from .blocks import LinearBlockOrderLike, linear_block
from .norms import NormLike


def create_cls_head(num_features: int, num_classes: int) -> torch.nn.Module:
    if num_classes == 0:
        return nn.Identity()

    return nn.Linear(num_features, num_classes)


def create_seg_head(
    dims: Sequence[int],
    num_classes: int,
    act: ActLike = "relu",
    norm: NormLike = "batch_norm1d",
    dropout: float = 0.0,
    order: LinearBlockOrderLike = "land",
) -> torch.nn.Module:
    if not dims or num_classes == 0:
        return nn.Identity()

    blocks: List[nn.Module] = []
    for in_features, out_features in itertools.pairwise(dims[:-1]):
        blocks.append(linear_block(in_features, out_features, act=act, norm=norm, dropout=dropout, order=order))
    blocks.append(nn.Linear(dims[-2], num_classes))
    return nn.Sequential(*blocks)
