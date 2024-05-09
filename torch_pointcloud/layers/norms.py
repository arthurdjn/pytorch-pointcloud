from typing import Any, Dict

from torch import nn as nn

from ._modules import MODULE_TYPE, REGISTERED_MODULE_TYPE, get_module

_NORM_LAYERS: Dict[str, REGISTERED_MODULE_TYPE] = dict(
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


def get_norm(name: MODULE_TYPE, *args: Any, **kwargs: Any) -> nn.Module:
    return get_module(name, *args, registry=_NORM_LAYERS, **kwargs)
