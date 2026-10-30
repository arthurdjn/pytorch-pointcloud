import pytest
import torch
import torch.nn as nn
from torch_geometric.nn import MLP
from torch_geometric.nn.norm import BatchNorm as GraphBatchNorm
from torch_geometric.nn.norm import InstanceNorm as GraphInstanceNorm
from torch_geometric.nn.norm import LayerNorm as GraphLayerNorm

from torch_pointcloud.layers.norms import GroupNorm, create_norm
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


def test_group_norm_matches_torch_group_norm_on_one_cloud() -> None:
    torch.manual_seed(0)
    norm = GroupNorm(16, num_groups=4)
    reference = nn.GroupNorm(4, 16)
    with torch.no_grad():
        norm.weight.copy_(torch.randn(16))
        norm.bias.copy_(torch.randn(16))
        reference.weight.copy_(norm.weight)
        reference.bias.copy_(norm.bias)
    x = torch.randn(50, 16) * 3 + 1
    expected = reference(x.t().unsqueeze(0)).squeeze(0).t()
    assert torch.allclose(norm(x), expected, atol=1e-5)
    assert torch.allclose(norm(x, torch.zeros(50, dtype=torch.long)), expected, atol=1e-5)


def test_group_norm_is_per_batch_element() -> None:
    torch.manual_seed(0)
    norm = GroupNorm(8, num_groups=2)
    a, b = torch.randn(30, 8), torch.randn(20, 8) * 5 + 2
    batch = torch.cat([torch.zeros(30), torch.ones(20)]).long()
    out = norm(torch.cat([a, b]), batch)
    assert torch.allclose(out[:30], norm(a), atol=1e-5)
    assert torch.allclose(out[30:], norm(b), atol=1e-5)
    assert torch.equal(norm(torch.cat([a, b]), batch, batch_size=2), out)


def test_group_norm_rejects_indivisible_channels() -> None:
    with pytest.raises(ValueError, match="multiple"):
        GroupNorm(8, num_groups=3)


def test_group_norm_empty_input() -> None:
    norm = GroupNorm(8, num_groups=2)
    out = norm(torch.zeros(0, 8), torch.zeros(0, dtype=torch.long))
    assert out.shape == (0, 8)


def test_create_norm_resolves_group_norm_for_packed_features() -> None:
    norm = create_norm("group_norm", 16, num_groups=4)
    assert isinstance(norm, GroupNorm) and norm.num_groups == 4
    assert isinstance(create_norm(GroupNorm, 16, num_groups=4), GroupNorm)
    assert isinstance(create_norm("group_norm", 16, dim=2, num_groups=4), nn.GroupNorm)


def test_group_norm_in_pyg_mlp_is_per_batch_element() -> None:
    torch.manual_seed(0)
    mlp = MLP([4, 8], norm=GroupNorm(8, num_groups=2), plain_last=False)
    a, b = torch.randn(30, 4), torch.randn(20, 4) * 5 + 2
    batch = torch.cat([torch.zeros(30), torch.ones(20)]).long()
    out = mlp(torch.cat([a, b]), batch, batch_size=2)
    assert torch.allclose(out[:30], mlp(a), atol=1e-5)
    assert torch.allclose(out[30:], mlp(b), atol=1e-5)
