from typing import Any, Dict, Literal

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


def get_dropout(name: DropoutLike, *args: Any, **kwargs: Any) -> nn.Module:
    return get_module(name, *args, registry=_DROPOUT_REGISTRY, **kwargs)
