from typing import Any, Dict, Literal

from torch import nn as nn

from ._modules import ModuleLike, RegisteredModuleLike, create_module

NormName = Literal[
    "batch_norm1d",
    "batch_norm2d",
    "batch_norm3d",
    "group_norm",
    "instance_norm1d",
    "instance_norm2d",
    "instance_norm3d",
    "layer_norm",
    "local_response_norm",
    "identity",
]

NormLike = ModuleLike[NormName]

_NORM_REGISTRY: Dict[NormName, RegisteredModuleLike] = dict(
    batch_norm1d=nn.BatchNorm1d,
    batch_norm2d=nn.BatchNorm2d,
    batch_norm3d=nn.BatchNorm3d,
    group_norm=nn.GroupNorm,
    instance_norm1d=nn.InstanceNorm1d,
    instance_norm2d=nn.InstanceNorm2d,
    instance_norm3d=nn.InstanceNorm3d,
    layer_norm=nn.LayerNorm,
    local_response_norm=nn.LocalResponseNorm,
    identity=nn.Identity,
)


def create_norm(name: NormLike, *args: Any, **kwargs: Any) -> nn.Module:
    return create_module(name, *args, registry=_NORM_REGISTRY, **kwargs)
