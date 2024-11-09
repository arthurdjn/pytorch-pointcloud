# Mostly from https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/activations.py

from functools import partial
from typing import Any, Dict, Literal

import torch
from torch import nn as nn
from torch.nn import functional as F

from ._modules import ModuleLike, RegisteredModuleLike, get_module


def swish(x: torch.Tensor, inplace: bool = False) -> torch.Tensor:
    """Swish - Described in: https://arxiv.org/abs/1710.05941"""
    return x.mul_(x.sigmoid()) if inplace else x.mul(x.sigmoid())


class Swish(nn.Module):
    def __init__(self, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return swish(x, self.inplace)


def mish(x: torch.Tensor, inplace: bool = False) -> torch.Tensor:
    """Mish: A Self Regularized Non-Monotonic Neural Activation Function - https://arxiv.org/abs/1908.08681"""
    inner = F.softplus(x).tanh()
    return x.mul_(inner) if inplace else x.mul(inner)


class Mish(nn.Module):
    """Mish: A Self Regularized Non-Monotonic Neural Activation Function - https://arxiv.org/abs/1908.08681"""

    def __init__(self, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return mish(x, inplace=self.inplace)


def sigmoid(x: torch.Tensor, inplace: bool = False) -> torch.Tensor:
    return x.sigmoid_() if inplace else x.sigmoid()


# PyTorch has this, but not with a consistent inplace argument interface
class Sigmoid(nn.Module):
    def __init__(self, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.sigmoid_() if self.inplace else x.sigmoid()


def tanh(x: torch.Tensor, inplace: bool = False) -> torch.Tensor:
    return x.tanh_() if inplace else x.tanh()


# PyTorch has this, but not with a consistent inplace argument interface
class Tanh(nn.Module):
    def __init__(self, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.tanh_() if self.inplace else x.tanh()


def hard_swish(x: torch.Tensor, inplace: bool = False) -> torch.Tensor:
    inner = F.relu6(x + 3.0).div_(6.0)
    return x.mul_(inner) if inplace else x.mul(inner)


class HardSwish(nn.Module):
    def __init__(self, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return hard_swish(x, self.inplace)


def hard_sigmoid(x: torch.Tensor, inplace: bool = False) -> torch.Tensor:
    if inplace:
        return x.add_(3.0).clamp_(0.0, 6.0).div_(6.0)
    return F.relu6(x + 3.0) / 6.0


class HardSigmoid(nn.Module):
    def __init__(self, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return hard_sigmoid(x, self.inplace)


def hard_mish(x: torch.Tensor, inplace: bool = False) -> torch.Tensor:
    """Hard Mish
    Experimental, based on notes by Mish author Diganta Misra at
      https://github.com/digantamisra98/H-Mish/blob/0da20d4bc58e696b6803f2523c58d3c8a82782d0/README.md
    """
    if inplace:
        return x.mul_(0.5 * (x + 2).clamp(min=0, max=2))
    return 0.5 * x * (x + 2).clamp(min=0, max=2)


class HardMish(nn.Module):
    def __init__(self, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return hard_mish(x, self.inplace)


def quick_gelu(x: torch.Tensor, inplace: bool = False) -> torch.Tensor:
    inner = torch.sigmoid(1.702 * x)
    return x.mul_(inner) if inplace else x.mul(inner)


class QuickGELU(nn.Module):
    """Applies the Gaussian Error Linear Units function (w/ dummy inplace arg)"""

    def __init__(self, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return quick_gelu(x, inplace=self.inplace)


# From https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/create_act.py
# PyTorch has an optimized, native 'silu' (aka 'swish') operator as of PyTorch 1.7.
# Also hardsigmoid, hardswish, and soon mish. This code will use native version if present.
# Eventually, the custom SiLU, Mish, Hard*, layers will be removed and only native variants will be used.
_has_silu = "silu" in dir(torch.nn.functional)
_has_hardswish = "hardswish" in dir(torch.nn.functional)
_has_hardsigmoid = "hardsigmoid" in dir(torch.nn.functional)
_has_mish = "mish" in dir(torch.nn.functional)

ActName = Literal[
    "silu",
    "swish",
    "mish",
    "relu",
    "relu6",
    "leaky_relu",
    "elu",
    "prelu",
    "celu",
    "selu",
    "gelu",
    "gelu_tanh",
    "quick_gelu",
    "sigmoid",
    "tanh",
    "hard_sigmoid",
    "hard_swish",
    "hard_mish",
    "identity",
]

ActLike = ModuleLike[ActName]

_ACT_REGISTRY: Dict[ActName, RegisteredModuleLike] = dict(
    silu=nn.SiLU if _has_silu else Swish,
    swish=nn.SiLU if _has_silu else Swish,
    mish=nn.Mish if _has_mish else Mish,
    relu=nn.ReLU,
    relu6=nn.ReLU6,
    leaky_relu=nn.LeakyReLU,
    elu=nn.ELU,
    prelu=nn.PReLU,
    celu=nn.CELU,
    selu=nn.SELU,
    gelu=nn.GELU,
    gelu_tanh=partial(nn.GELU, approximate="tanh"),
    quick_gelu=QuickGELU,
    sigmoid=Sigmoid,
    tanh=Tanh,
    hard_sigmoid=nn.Hardsigmoid if _has_hardsigmoid else HardSigmoid,
    hard_swish=nn.Hardswish if _has_hardswish else HardSwish,
    hard_mish=HardMish,
    identity=nn.Identity,
)


def get_act(name: ActLike, *args: Any, **kwargs: Any) -> nn.Module:
    return get_module(name, *args, registry=_ACT_REGISTRY, **kwargs)
