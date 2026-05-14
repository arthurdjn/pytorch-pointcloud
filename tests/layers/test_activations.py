import pytest
import torch
import torch.nn as nn

from torch_pointcloud.layers.activations import (
    HardMish,
    HardSigmoid,
    HardSwish,
    Mish,
    QuickGELU,
    Sigmoid,
    Swish,
    Tanh,
    create_act,
    hard_mish,
    hard_sigmoid,
    hard_swish,
    mish,
    quick_gelu,
    sigmoid,
    swish,
    tanh,
)

_ALL_NAMES = [
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


@pytest.mark.parametrize("name", _ALL_NAMES)
def test_create_act(name: str) -> None:
    layer = create_act(name)
    assert isinstance(layer, nn.Module)
    out = layer(torch.randn(8, 16))
    assert out.shape == (8, 16)


@pytest.mark.parametrize("cls", [Swish, Mish, Sigmoid, Tanh, HardSwish, HardSigmoid, HardMish, QuickGELU])
def test_activation_modules(cls: type) -> None:
    x = torch.randn(8, 16)
    out = cls(inplace=False)(x)
    assert out.shape == x.shape


@pytest.mark.parametrize("fn", [swish, mish, sigmoid, tanh, hard_swish, hard_sigmoid, hard_mish, quick_gelu])
def test_activation_functions(fn) -> None:  # type: ignore[no-untyped-def]
    x = torch.randn(8, 16)
    out = fn(x, inplace=False)
    assert out.shape == x.shape


def test_create_act_passes_module() -> None:
    given = nn.ReLU()
    assert create_act(given) is given
