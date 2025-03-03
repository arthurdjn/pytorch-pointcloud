from typing import Any, Dict, Literal, Union

from torch import nn as nn

from ._modules import ModuleLike, RegisteredModuleLike, get_module

DropoutName = Literal["dropout", "dropout2d", "dropout3d", "alpha_dropout", "feature_alpha_dropout"]
DropoutLike = ModuleLike[DropoutName]

_DROPOUT_REGISTRY: Dict[DropoutName, RegisteredModuleLike] = dict(
    dropout=nn.Dropout,
    dropout2d=nn.Dropout2d,
    dropout3d=nn.Dropout3d,
    alpha_dropout=nn.AlphaDropout,
    feature_alpha_dropout=nn.FeatureAlphaDropout,
)


def get_dropout(name: Union[DropoutLike, float], *args: Any, **kwargs: Any) -> nn.Module:
    if isinstance(name, (int, float)):
        return get_module("dropout", p=name, *args, registry=_DROPOUT_REGISTRY, **kwargs)
    return get_module(name, *args, registry=_DROPOUT_REGISTRY, **kwargs)
