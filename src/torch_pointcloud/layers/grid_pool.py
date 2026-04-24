"""Grid-based point cloud pooling (downsampling).

Clusters points by quantized grid coordinates and reduces features
via scatter operations.  This is an alternative to code-space pooling
(`SerializedPooling`) used in PTV3 Mode 2 (Sonata) and Mode 3 (Utonia).
"""

from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn.resolver import activation_resolver, normalization_resolver

from torch_pointcloud.utils.imports import optional_import

if TYPE_CHECKING:
    import torch_scatter

torch_scatter, _ = optional_import("torch_scatter")


class GridPool(nn.Module):
    """Grid-based downsampling that clusters points by quantized coordinates.

    Divides `pos` by `stride`, groups unique voxels via
    `torch.unique`, and reduces features with `torch_scatter.segment_csr`.

    Args:
        in_channels: Number of input feature channels.
        out_channels: Number of output feature channels.
        stride: Spatial stride for grid quantization.
        act: Activation layer applied after projection.
        act_kwargs: Extra arguments for the activation function.
        act_first: Apply activation before normalization.
        norm: Normalization layer applied after projection.
        norm_kwargs: Extra arguments for the normalization layer.
        reduce: Scatter reduction (`"max"`, `"mean"`, `"sum"`, `"min"`).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 2,
        bias: bool = True,
        act: Union[str, Callable, None] = None,
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = None,
        norm_kwargs: Optional[Dict[str, Any]] = None,
        reduce: str = "max",
    ):
        super().__init__()
        act_kwargs = act_kwargs or {}
        norm_kwargs = norm_kwargs or {}
        if reduce not in ("sum", "mean", "min", "max"):
            raise ValueError(f"Invalid reduce operation: {reduce}.")

        self.stride = stride
        self.reduce = reduce
        self.act_first = act_first
        self.proj = nn.Linear(in_channels, out_channels, bias=bias)
        self.norm = normalization_resolver(norm, out_channels, **norm_kwargs) if norm is not None else None
        self.act = activation_resolver(act, **act_kwargs) if act is not None else None

    def forward(self, x: Tensor, pos: Tensor, batch: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Downsample points by grid quantization.

        Args:
            x: Point features of shape $(N, C_{in})$.
            pos: Integer grid coordinates of shape $(N, 3)$.
            batch: Batch indices of shape $(N,)$.

        Returns:
            Tuple of `(x_pooled, pos_pooled, batch_pooled, pooling_inverse)`
            where `pooling_inverse` maps each input point to its pooled cluster index.
        """
        pos_pooled = torch.div(pos, self.stride, rounding_mode="trunc")
        tagged = pos_pooled | batch.view(-1, 1) << 48
        tagged, cluster, counts = torch.unique(tagged, sorted=True, return_inverse=True, return_counts=True, dim=0)
        pos_pooled = tagged & ((1 << 48) - 1)

        _, indices = torch.sort(cluster)
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        head_indices = indices[idx_ptr[:-1]]

        x_pooled = torch_scatter.segment_csr(self.proj(x)[indices], idx_ptr, reduce=self.reduce)
        batch_pooled = batch[head_indices]

        if self.act_first:
            if self.act is not None:
                x_pooled = self.act(x_pooled)
            if self.norm is not None:
                x_pooled = self.norm(x_pooled)
        else:
            if self.norm is not None:
                x_pooled = self.norm(x_pooled)
            if self.act is not None:
                x_pooled = self.act(x_pooled)

        return x_pooled, pos_pooled, batch_pooled, cluster
