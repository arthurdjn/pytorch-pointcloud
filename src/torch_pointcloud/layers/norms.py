r"""Normalization factory wrapper with spatial dimensionality support.

PyG's `normalization_resolver` covers graph-aware norms and `nn.BatchNorm1d` but
does not register `nn.BatchNorm2d` / `nn.BatchNorm3d`. For convolutional stacks
operating on 2D feature maps or 3D voxel grids we want those Nd variants from
the same string name. `create_norm` adds a `dim` parameter (1, 2, or 3) that
selects the corresponding `nn.*Nd` class for the common families:

| `norm`             | `dim=1`          | `dim=2`               | `dim=3`               |
| ------------------ | ---------------- | --------------------- | --------------------- |
| `"batch_norm"`     | (PyG `BatchNorm`)| `nn.BatchNorm2d`      | `nn.BatchNorm3d`      |
| `"instance_norm"`  | (PyG resolver)   | `nn.InstanceNorm2d`   | `nn.InstanceNorm3d`   |
| `"group_norm"`     | `nn.GroupNorm`   | `nn.GroupNorm`        | `nn.GroupNorm`        |
| `"layer_norm"`     | `nn.LayerNorm`   | `nn.LayerNorm`        | `nn.LayerNorm`        |

Inspired by MONAI's `Norm[name, dim]` factory pattern.
"""

from typing import Any, Callable, Dict, Optional, Type, Union

import torch.nn as nn
from torch_geometric.nn.resolver import normalization_resolver

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
    **norm_kwargs: Any,
) -> Optional[nn.Module]:
    r"""Resolve a normalization layer with a spatial dimensionality hint.

    For `dim == 1`, defers to PyG's `normalization_resolver` (graph-aware norms
    plus `BatchNorm1d`). For `dim` $\in \{2, 3\}$, maps common norm names to
    the matching `nn.*Nd` variant. Pass a class or an existing instance to
    bypass string resolution.

    Args:
        norm: Norm name (`"batch_norm"`, `"instance_norm"`, `"group_norm"`,
            `"layer_norm"`), a class, an instance, or `None`.
        channels: Number of feature channels.
        dim: Spatial dimensionality of the feature map. $1$ for packed / graph
            tensors $(N, C)$, $2$ for $(B, C, H, W)$, $3$ for $(B, C, H, W, D)$.
        **norm_kwargs: Forwarded to the norm constructor.

    Returns:
        The instantiated norm module, or `None` if `norm is None`.
    """
    if norm is None:
        return None
    if isinstance(norm, nn.Module):
        return norm
    if isinstance(norm, str):
        if dim == 1:
            return normalization_resolver(norm, channels, **norm_kwargs)
        key = norm.lower().replace("-", "_")
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
