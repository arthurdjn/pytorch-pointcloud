"""Dropout layers: stochastic depth (`DropPath`) and the `create_dropout` factory."""

from typing import Any, Dict, Literal, Optional, Union, overload

import torch.nn as nn
from torch import Tensor

from ._modules import ModuleLike, RegisteredModuleLike, create_module


def drop_path(x: Tensor, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True) -> Tensor:
    r"""Drop paths (Stochastic Depth): zero whole entries of the first dimension of `x`, as described in the paper
    :arxiv: [Deep Networks with Stochastic Depth](https://arxiv.org/abs/1603.09382)
    by Gao Huang, Yu Sun, Zhuang Liu, Daniel Sedra, Kilian Weinberger.

    One Bernoulli mask value is drawn per entry of the first dimension and broadcast over the others. On a padded
    $(B, \ldots)$ tensor that drops the residual branch of whole samples; on packed $(N, C)$ features every row is a
    point, so the branch is dropped per point, not per sample.

    Implementation is taken from original implementation by Ross Wightman in
    [pytorch-image-models](https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/drop.py).

    Args:
        x: Output of the residual branch.
        drop_prob: Probability of zeroing an entry of the first dimension.
        training: Whether to drop; `x` is returned unchanged when `False`.
        scale_by_keep: Whether to divide the kept entries by $1 - \text{drop\_prob}$, which keeps the expectation.

    Returns:
        The masked tensor, same shape as `x`.

    Shape:
        - x: $(N, *)$
        - output: $(N, *)$

    Example:
        ```pycon
        >>> import torch
        >>> from torch_pointcloud.layers.dropouts import drop_path
        >>> x = torch.ones(6, 4)
        >>> drop_path(x, drop_prob=0.5, training=True).shape
        torch.Size([6, 4])

        ```
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
    r"""Drop paths (Stochastic Depth) on the main path of a residual block, the module form of `drop_path`.

    The mask holds one value per entry of the first dimension: whole samples of a padded $(B, \ldots)$ tensor, single
    points (rows) of packed $(N, C)$ features.

    Args:
        drop_prob: Probability of zeroing an entry of the first dimension.
        scale_by_keep: Whether to divide the kept entries by $1 - \text{drop\_prob}$, which keeps the expectation.

    Shape:
        - Input: $(N, *)$
        - Output: $(N, *)$

    Example:
        ```pycon
        >>> import torch
        >>> from torch_pointcloud.layers import DropPath
        >>> drop = DropPath(drop_prob=0.5)
        >>> drop(torch.ones(6, 4)).shape
        torch.Size([6, 4])

        ```
    """

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


@overload
def create_dropout(name: None, *args: Any, **kwargs: Any) -> None: ...


@overload
def create_dropout(name: Union[DropoutLike, float], *args: Any, **kwargs: Any) -> nn.Module: ...


def create_dropout(name: Union[DropoutLike, float, None], *args: Any, **kwargs: Any) -> Optional[nn.Module]:
    """Resolve a dropout layer from a name, a class, an instance, or a probability, returning `None` for `name=None`.

    Args:
        name: Dropout name (e.g., `"dropout"`, `"drop_path"`), a class, an instance, a float
            probability, which is a shorthand for `nn.Dropout(p=name)`, or `None`.
        *args: Forwarded to the dropout constructor.
        **kwargs: Forwarded to the dropout constructor (ignored if `name` is already an instance).

    Returns:
        The instantiated dropout module, or `None` if `name is None`.
    """
    if name is None:
        return None
    if isinstance(name, (int, float)):
        return create_module("dropout", p=name, *args, registry=_DROPOUT_REGISTRY, **kwargs)
    return create_module(name, *args, registry=_DROPOUT_REGISTRY, **kwargs)
