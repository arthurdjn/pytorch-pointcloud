import pytest
import torch
import torch.nn as nn

from torch_pointcloud.layers.act import create_act


def test_none_returns_none() -> None:
    assert create_act(None) is None


def test_none_or_identity_idiom() -> None:
    act = create_act(None) or nn.Identity()
    assert isinstance(act, nn.Identity)


@pytest.mark.parametrize(
    "act,expected",
    [
        pytest.param("relu", nn.ReLU, id="relu"),
        pytest.param("gelu", nn.GELU, id="gelu"),
        pytest.param("leaky_relu", nn.LeakyReLU, id="leaky_relu"),
    ],
)
def test_string_resolves_to_module(act: str, expected: type) -> None:
    assert type(create_act(act)) is expected


def test_string_applies_activation() -> None:
    act = create_act("relu")
    assert act is not None
    x = torch.tensor([-1.0, 0.0, 2.0])
    assert torch.equal(act(x), torch.tensor([0.0, 0.0, 2.0]))


def test_kwargs_forwarded_to_constructor() -> None:
    act = create_act("leaky_relu", negative_slope=0.2)
    assert isinstance(act, nn.LeakyReLU)
    assert act.negative_slope == 0.2


def test_module_instance_returned_unchanged() -> None:
    instance = nn.SiLU()
    assert create_act(instance) is instance


def test_module_class_is_instantiated() -> None:
    act = create_act(nn.GELU)
    assert isinstance(act, nn.GELU)
