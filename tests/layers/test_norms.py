import pytest
import torch
import torch.nn as nn
from torch_geometric.nn.norm import BatchNorm as GraphBatchNorm
from torch_geometric.nn.norm import InstanceNorm as GraphInstanceNorm
from torch_geometric.nn.norm import LayerNorm as GraphLayerNorm

from torch_pointcloud.layers.norms import create_norm
from torch_pointcloud.layers.pdnorm import PDNorm


def test_none_returns_none() -> None:
    assert create_norm(None, 8) is None


def test_none_or_identity_idiom() -> None:
    norm = create_norm(None, 8) or nn.Identity()
    assert isinstance(norm, nn.Identity)


@pytest.mark.parametrize(
    "norm,expected",
    [
        pytest.param("batch_norm", GraphBatchNorm, id="batch_norm"),
        pytest.param("instance_norm", GraphInstanceNorm, id="instance_norm"),
        pytest.param("layer_norm", GraphLayerNorm, id="layer_norm"),
    ],
)
def test_dim1_defers_to_graph_resolver(norm: str, expected: type) -> None:
    assert type(create_norm(norm, 8)) is expected


def test_layer_norm_dim1_is_graph_layer_norm_not_torch() -> None:
    """`create_norm("layer_norm", dim=1)` resolves to PyG's graph `LayerNorm`, which normalizes over all
    nodes of a graph, not to `nn.LayerNorm`; dense $(B, N, C)$ blocks must pass the `nn.LayerNorm` class."""
    norm = create_norm("layer_norm", 8)
    assert type(norm) is GraphLayerNorm
    assert not isinstance(norm, nn.LayerNorm)


@pytest.mark.parametrize(
    "norm,dim,expected",
    [
        pytest.param("batch_norm", 2, nn.BatchNorm2d, id="batch_norm-2d"),
        pytest.param("batch_norm", 3, nn.BatchNorm3d, id="batch_norm-3d"),
        pytest.param("bn", 2, nn.BatchNorm2d, id="bn-alias-2d"),
        pytest.param("instance_norm", 2, nn.InstanceNorm2d, id="instance_norm-2d"),
        pytest.param("instance_norm", 3, nn.InstanceNorm3d, id="instance_norm-3d"),
        pytest.param("layer_norm", 2, nn.LayerNorm, id="layer_norm-2d"),
    ],
)
def test_dim_aware_dispatch(norm: str, dim: int, expected: type) -> None:
    assert type(create_norm(norm, 8, dim=dim)) is expected


def test_dim2_batch_norm_forward_shape() -> None:
    norm = create_norm("batch_norm", 8, dim=2)
    assert norm is not None
    x = torch.randn(2, 8, 4, 4)
    assert norm(x).shape == x.shape


def test_group_norm_forwards_kwargs() -> None:
    norm = create_norm("group_norm", 8, dim=2, num_groups=2)
    assert isinstance(norm, nn.GroupNorm)
    assert norm.num_groups == 2
    assert norm.num_channels == 8


def test_module_class_is_instantiated() -> None:
    norm = create_norm(nn.BatchNorm1d, 8)
    assert isinstance(norm, nn.BatchNorm1d)
    assert norm.num_features == 8


def test_module_instance_returned_unchanged() -> None:
    instance = nn.LayerNorm(8)
    assert create_norm(instance, 8) is instance


def test_unknown_string_raises_for_spatial_dim() -> None:
    with pytest.raises(ValueError, match="Unknown norm string"):
        create_norm("graph_norm", 8, dim=2)


def test_conditions_wrap_in_pdnorm() -> None:
    norm = create_norm("batch_norm", 8, conditions=["scannet", "s3dis"])
    assert isinstance(norm, PDNorm)
