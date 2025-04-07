import itertools
from typing import TYPE_CHECKING, Any, Optional, Sequence, Tuple, Union, overload

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torch_pointcloud.layers import ActLike, NormLike, create_cls_head, create_pool, create_seg_head, linear_block
from torch_pointcloud.utils.conversion import ensure_tuple, ensure_tuple_size
from torch_pointcloud.utils.imports import optional_import

if TYPE_CHECKING:
    from torch_cluster import fps, knn, radius


fps, _ = optional_import("torch_cluster", "fps")
knn, _ = optional_import("torch_cluster", "knn")
radius, _ = optional_import("torch_cluster", "radius")


class PointNetSA(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        ratio: float,
        radius: Union[float, Sequence[float]],
        k: Union[int, Sequence[int]],
        pool: str = "max",
        act: ActLike = "relu",
        norm: NormLike = "batch_norm1d",
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.ratio = ratio
        self.radius = radius  # ensure_tuple(radius)
        self.k = k  # ensure_tuple(k)
        self.mlp = linear_block(in_channels + 3, out_channels, act=act, norm=norm, bias=bias)
        self.pool = create_pool(pool)

    def forward(
        self,
        features: Tensor,
        coords: Tensor,
        batch: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        idx = fps(coords, batch, ratio=self.ratio)
        new_coords = coords[idx]
        new_batch = batch[idx]

        # for r, k, mlp in zip(self.radius, self.k, self.mlps):
        row, col = radius(coords, new_coords, self.radius, batch, batch[idx], max_num_neighbors=self.k)
        # row: Tensor of shape (N,)
        # col: Tensor of shape (N,)
        rel_coords = coords[col] - new_coords[row]
        new_features = features[col]
        new_features = torch.cat([new_features, rel_coords], dim=1)
        new_features = self.mlp(new_features)
        new_features = self.pool(new_features, row)

        return new_coords, new_features, new_batch
