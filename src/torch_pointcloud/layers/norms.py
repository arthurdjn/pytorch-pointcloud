r"""Normalization factory wrapper with spatial dimensionality support.

PyG's `normalization_resolver` covers graph-aware norms and `nn.BatchNorm1d`
but does not register the `nn.BatchNorm2d` / `nn.BatchNorm3d` variants needed
by convolutional stacks. `create_norm` adds a `dim` parameter that selects the
matching `nn.*Nd` class for the common families when `dim > 1`, and defers to
PyG's resolver otherwise.
"""

from typing import Any, Callable, Dict, Optional, Sequence, Type, Union

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


def is_layer_norm_name(norm: Union[str, Callable, None]) -> bool:
    """Whether `norm` is a name that `create_norm` resolves to the graph `LayerNorm` for `dim == 1`.

    That layer normalizes over every node of a graph unless it is built with `mode="node"`.

    Args:
        norm: Norm name, class, instance, or `None`.

    Returns:
        `True` for the spellings of the layer normalization name, such as `"layer_norm"` or `"LayerNorm"`.
    """
    if not isinstance(norm, str):
        return False

    key = norm.lower().replace("_", "").replace("-", "").replace(" ", "")
    return key in ("layernorm", "layer")


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
    plus `BatchNorm1d`). For `dim` $\in \{2, 3\}$, maps common norm names to
    the matching `nn.*Nd` variant. Pass a class or an existing instance to
    bypass string resolution. When `conditions` is given, the resolved norm is
    wrapped in a per-condition `PDNorm` for multi-dataset (prompt-driven) training.

    Args:
        norm: Norm name (`"batch_norm"`, `"instance_norm"`, `"group_norm"`,
            `"layer_norm"`), a class, an instance, or `None`. `"layer_norm"` is only resolved for `dim == 1`.
        channels: Number of feature channels.
        dim: Spatial dimensionality of the feature map. $1$ for packed / graph
            tensors $(N, C)$, $2$ for $(B, C, H, W)$, $3$ for $(B, C, H, W, D)$.
        conditions: Ordered dataset condition names. When set (and `norm` is not
            `None`), returns a `PDNorm` holding one inner `norm` per condition.
        **norm_kwargs: Forwarded to the norm constructor.

    Returns:
        The instantiated norm module, or `None` if `norm is None`.

    Raises:
        ValueError: If `norm` is a name that has no variant for `dim`.
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
            raise ValueError(
                f"{norm!r} is not supported for dim={dim}: `nn.LayerNorm` normalizes the last axis, not the channel "
                "axis of a channel-first feature map. Pass a class / callable instead."
            )

        raise ValueError(
            f"Unknown norm string {norm!r} for dim={dim}. Use 'batch_norm', 'instance_norm', "
            "'group_norm', or pass a class / callable."
        )

    return norm(channels, **norm_kwargs)
