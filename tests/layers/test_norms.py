import pytest
import torch.nn as nn

from torch_pointcloud.layers.norms import create_norm


@pytest.mark.parametrize(
    "name,expected_cls",
    [
        ("batch_norm1d", nn.BatchNorm1d),
        ("batch_norm2d", nn.BatchNorm2d),
        ("batch_norm3d", nn.BatchNorm3d),
        ("instance_norm1d", nn.InstanceNorm1d),
        ("instance_norm2d", nn.InstanceNorm2d),
        ("instance_norm3d", nn.InstanceNorm3d),
        ("layer_norm", nn.LayerNorm),
        ("identity", nn.Identity),
    ],
)
def test_create_norm_by_name(name: str, expected_cls: type) -> None:
    layer = create_norm(name, 16)  # type: ignore[arg-type]
    assert isinstance(layer, expected_cls)


def test_create_norm_group_norm() -> None:
    layer = create_norm("group_norm", 4, 16)
    assert isinstance(layer, nn.GroupNorm)
    assert layer.num_groups == 4
    assert layer.num_channels == 16


def test_create_norm_passes_module() -> None:
    given = nn.BatchNorm1d(8)
    assert create_norm(given) is given
