r"""Normalization factory wrapper with spatial dimensionality support.

PyG's `normalization_resolver` covers graph-aware norms and `nn.BatchNorm1d`
but does not register the `nn.BatchNorm2d` / `nn.BatchNorm3d` variants needed
by convolutional stacks. `create_norm` adds a `dim` parameter that selects the
matching `nn.*Nd` class for the common families when `dim > 1`, and defers to
PyG's resolver otherwise.
"""

from typing import Any, Callable, Dict, Optional, Sequence, Type, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn.resolver import normalization_resolver
from torch_geometric.utils import scatter

_BATCH_NORM_NDS: Dict[int, Type[nn.Module]] = {
    1: nn.BatchNorm1d,
    2: nn.BatchNorm2d,
    3: nn.BatchNorm3d,
}
_INSTANCE_NORM_NDS: Dict[int, Type[nn.Module]] = {
    1: nn.InstanceNorm1d,
    2: nn.InstanceNorm2d,
    3: nn.InstanceNorm3d,
}


def create_norm(
    norm: Union[str, Callable, None],
    channels: int,
    *,
    dim: int = 1,
    conditions: Optional[Sequence[str]] = None,
    **norm_kwargs: Any,
) -> Optional[nn.Module]:
    r"""Resolve a normalization layer with a spatial dimensionality hint.

    For `dim == 1`, defers to PyG's `normalization_resolver` (graph-aware norms
    plus `BatchNorm1d`), except `"group_norm"`, which resolves to the packed
    `GroupNorm`. For `dim` $\in \{2, 3\}$, maps common norm names to
    the matching `nn.*Nd` variant. Pass a class or an existing instance to
    bypass string resolution. When `conditions` is given, the resolved norm is
    wrapped in a per-condition `PDNorm` for multi-dataset (prompt-driven) training.

    Args:
        norm: Norm name (`"batch_norm"`, `"instance_norm"`, `"group_norm"`,
            `"layer_norm"`), a class, an instance, or `None`.
        channels: Number of feature channels.
        dim: Spatial dimensionality of the feature map. $1$ for packed / graph
            tensors $(N, C)$, $2$ for $(B, C, H, W)$, $3$ for $(B, C, H, W, D)$.
        conditions: Ordered dataset condition names. When set (and `norm` is not
            `None`), returns a `PDNorm` holding one inner `norm` per condition.
        **norm_kwargs: Forwarded to the norm constructor.

    Returns:
        The instantiated norm module, or `None` if `norm is None`.
    """
    if norm is None:
        return None

    if conditions is not None:
        # NOTE: import pdnorm here to avoid circular import
        from .pdnorm import PDNorm

        return PDNorm(channels, conditions=conditions, norm=norm, dim=dim, **norm_kwargs)

    if isinstance(norm, nn.Module):
        return norm

    if isinstance(norm, str):
        key = norm.lower().replace("-", "_")
        if dim == 1:
            if key in {"group_norm", "groupnorm", "gn"}:
                return GroupNorm(channels, **norm_kwargs)
            return normalization_resolver(norm, channels, **norm_kwargs)

        if key in {"batch_norm", "batchnorm", "bn"}:
            return _BATCH_NORM_NDS[dim](channels, **norm_kwargs)

        if key in {"instance_norm", "instancenorm", "in"}:
            return _INSTANCE_NORM_NDS[dim](channels, **norm_kwargs)

        if key in {"group_norm", "groupnorm", "gn"}:
            return nn.GroupNorm(num_channels=channels, **norm_kwargs)

        if key in {"layer_norm", "layernorm", "ln"}:
            return nn.LayerNorm(channels, **norm_kwargs)

        raise ValueError(
            f"Unknown norm string {norm!r} for dim={dim}. Use 'batch_norm', 'instance_norm', "
            "'group_norm', 'layer_norm', or pass a class / callable."
        )

    return norm(channels, **norm_kwargs)


class GroupNorm(nn.Module):
    r"""Group normalization of packed point features, with statistics per batch element.

    The channels are split into `num_groups` groups and every group is normalized by the mean and variance of its
    channels over the points of one batch element, then scaled and shifted per channel. `torch.nn.GroupNorm` has no
    batch index and would pool the statistics over every cloud of the batch; called without `batch`, this layer
    does the same. It takes the batch index as the PyG normalizations do, `norm(x, batch, batch_size)`, and
    `create_norm("group_norm", C, num_groups=G)` builds it. PyG's `MLP` does not resolve it by name, and shares an
    instance passed as its `norm` across all its layers.

    Args:
        in_channels: Number of channels $C$.
        num_groups: Number of channel groups $G$; `in_channels` must be a multiple of it.
        eps: Added to the variance for numerical stability.

    Shape:
        - Input: $(N, C)$ features, an optional $(N,)$ batch index and the optional number of batch elements.
        - Output: $(N, C)$.

    Example:
        ```pycon
        >>> import torch
        >>> from torch_pointcloud.layers.norms import GroupNorm
        >>> norm = GroupNorm(8, num_groups=4)
        >>> x = torch.randn(10, 8)
        >>> norm(x, torch.tensor([0] * 6 + [1] * 4), batch_size=2).shape
        torch.Size([10, 8])

        ```
    """

    def __init__(self, in_channels: int, num_groups: int, eps: float = 1e-5) -> None:
        super().__init__()
        if in_channels % num_groups != 0:
            raise ValueError(f"`in_channels` ({in_channels}) must be a multiple of `num_groups` ({num_groups}).")

        self.in_channels = in_channels
        self.num_groups = num_groups
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(in_channels))
        self.bias = nn.Parameter(torch.zeros(in_channels))

    def forward(self, x: Tensor, batch: Optional[Tensor] = None, batch_size: Optional[int] = None) -> Tensor:
        grouped = x.view(x.size(0), self.num_groups, self.in_channels // self.num_groups)
        if batch is None:
            mean = grouped.mean(dim=(0, 2), keepdim=True)
            var = grouped.var(dim=(0, 2), unbiased=False, keepdim=True)
        else:
            if batch_size is None:
                batch_size = int(batch.max()) + 1 if batch.numel() else 0
            group_mean = grouped.mean(dim=2)  # (N, G)
            mean = scatter(group_mean, batch, dim=0, dim_size=batch_size, reduce="mean")[batch].unsqueeze(2)
            group_var = (grouped - mean).pow(2).mean(dim=2)  # (N, G)
            var = scatter(group_var, batch, dim=0, dim_size=batch_size, reduce="mean")[batch].unsqueeze(2)

        x = ((grouped - mean) / torch.sqrt(var + self.eps)).view_as(x)
        return x * self.weight + self.bias

    def extra_repr(self) -> str:
        return f"{self.in_channels}, num_groups={self.num_groups}, eps={self.eps}"
