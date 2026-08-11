from functools import partial

import pytest
import torch.nn as nn

from torch_pointcloud.layers._modules import ModuleRegistryDict, create_module

_REGISTRY: ModuleRegistryDict = {
    "linear": nn.Linear,
    "relu6": partial(nn.ReLU6, inplace=True),
}


def test_create_module_by_name() -> None:
    module = create_module("linear", 4, 8, registry=_REGISTRY)
    assert isinstance(module, nn.Linear)
    assert module.in_features == 4
    assert module.out_features == 8


def test_create_module_from_partial_entry() -> None:
    module = create_module("relu6", registry=_REGISTRY)
    assert isinstance(module, nn.ReLU6)
    assert module.inplace is True


def test_create_module_instance_passthrough() -> None:
    instance = nn.Linear(2, 3)
    assert create_module(instance, registry=_REGISTRY) is instance


def test_create_module_from_class() -> None:
    module = create_module(nn.Identity, registry=_REGISTRY)
    assert isinstance(module, nn.Identity)


def test_create_module_from_partial() -> None:
    module = create_module(partial(nn.Dropout, p=0.25), registry=_REGISTRY)
    assert isinstance(module, nn.Dropout)
    assert module.p == 0.25


def test_create_module_unknown_name_lists_available() -> None:
    with pytest.raises(ValueError, match="linear, relu6"):
        create_module("conv", registry=_REGISTRY)
