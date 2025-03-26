from typing import Any, Dict, Literal, Union

from torch import nn as nn

from ._modules import ModuleLike, RegisteredModuleLike, get_module

"""

"""

import torch.nn as nn
from torch import Tensor


def drop_path(x: Tensor, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True) -> Tensor:
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks),
    as described in the paper [Deep Networks with Stochastic Depth](https://arxiv.org/abs/1603.09382)
    by Gao Huang, Yu Sun, Zhuang Liu, Daniel Sedra, Kilian Weinberger.

    Implementation is taken from original implementation by Ross Wightman in
    [pytorch-image-models](https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/drop.py).
    """
    if drop_prob == 0.0 or not training:
        return x

    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)

    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)

    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x: Tensor) -> Tensor:
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self) -> str:
        return f"drop_prob={round(self.drop_prob, 3):0.3f}"


DropoutName = Literal["dropout", "dropout2d", "dropout3d", "alpha_dropout", "feature_alpha_dropout", "drop_path"]
DropoutLike = ModuleLike[DropoutName]

_DROPOUT_REGISTRY: Dict[DropoutName, RegisteredModuleLike] = dict(
    dropout=nn.Dropout,
    dropout2d=nn.Dropout2d,
    dropout3d=nn.Dropout3d,
    alpha_dropout=nn.AlphaDropout,
    feature_alpha_dropout=nn.FeatureAlphaDropout,
    drop_path=DropPath,
)


def get_dropout(name: Union[DropoutLike, float], *args: Any, **kwargs: Any) -> nn.Module:
    if isinstance(name, (int, float)):
        return get_module("dropout", p=name, *args, registry=_DROPOUT_REGISTRY, **kwargs)
    return get_module(name, *args, registry=_DROPOUT_REGISTRY, **kwargs)
