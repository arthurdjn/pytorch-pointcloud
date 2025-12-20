from typing import Dict, Literal

import pytest
import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import MLP, knn, knn_graph

from torch_pointcloud.layers.geometric_affine import GeometricAffineConv
from torch_pointcloud.utils.imports import _CUDA_AVAILABLE, _TORCH_CLUSTER_AVAILABLE, _TORCH_SCATTER_AVAILABLE

DEVICES = [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA not available")),
]


@pytest.fixture
def data() -> Dict[str, Tensor]:
    torch.manual_seed(42)
    x = torch.randn(1_000, 3)
    pos = torch.randn(1_000, 3)
    batch = torch.cat([torch.zeros(250), torch.ones(750)]).long()
    edge_index = knn_graph(pos, k=16, batch=batch)
    return dict(x=x, pos=pos, batch=batch, edge_index=edge_index)


@pytest.mark.skipif(
    not _TORCH_CLUSTER_AVAILABLE and not _TORCH_SCATTER_AVAILABLE,
    reason="torch-cluster or torch-scatter is not installed",
)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("use_pos", [True, False])
@pytest.mark.parametrize("normalize", ["center", "anchor"])
def test_geometric_affine_conv_basic(
    data: Dict[str, Tensor],
    device: Literal["cpu", "cuda"],
    use_pos: bool,
    normalize: Literal["center", "anchor"],
) -> None:
    """Test basic GeometricAffineConv functionality."""
    spatial_dim = 3 if use_pos else 0
    local_nn = MLP([2 * 3 + spatial_dim, 32])
    conv = GeometricAffineConv(
        local_nn=local_nn,
        channels=3,
        spatial_dim=spatial_dim,
        use_pos=use_pos,
        normalize=normalize,
    )

    conv = conv.to(device)
    data = {k: v.to(device) for k, v in data.items()}

    output = conv(data["x"], data["pos"], data["batch"], data["edge_index"])
    assert output.shape == (len(data["pos"]), 32)
    assert output.dtype == data["x"].dtype


@pytest.mark.skipif(
    not _TORCH_CLUSTER_AVAILABLE and not _TORCH_SCATTER_AVAILABLE,
    reason="torch-cluster or torch-scatter is not installed",
)
@pytest.mark.parametrize("device", DEVICES)
def test_geometric_affine_conv_with_self_loops(data: Dict[str, Tensor], device: Literal["cpu", "cuda"]) -> None:
    """Test GeometricAffineConv with self loops."""
    local_nn = MLP([2 * 3 + 3, 32])
    conv = GeometricAffineConv(
        local_nn=local_nn,
        channels=3,
        spatial_dim=3,
        use_pos=True,
        normalize="center",
        add_self_loops=True,
    )

    conv = conv.to(device)
    data = {k: v.to(device) for k, v in data.items()}

    output = conv(data["x"], data["pos"], data["batch"], data["edge_index"])
    assert output.shape == (len(data["pos"]), 32)


@pytest.mark.skipif(
    not _TORCH_CLUSTER_AVAILABLE and not _TORCH_SCATTER_AVAILABLE,
    reason="torch-cluster or torch-scatter is not installed",
)
def test_geometric_affine_conv_pair_tensor(data: Dict[str, Tensor]) -> None:
    """Test GeometricAffineConv with pair tensor inputs (different source and destination)."""
    local_nn = MLP([2 * 3 + 3, 32])
    conv = GeometricAffineConv(
        local_nn=local_nn,
        channels=3,
        spatial_dim=3,
        use_pos=True,
        normalize="center",
    )

    num_dst = len(data["pos"]) // 2
    pos_dst = data["pos"][:num_dst]
    x_dst = data["x"][:num_dst]
    batch_dst = data["batch"][:num_dst]

    row, col = knn(x=data["pos"], y=pos_dst, k=8, batch_x=data["batch"], batch_y=batch_dst)
    edge_index = torch.stack([col, row], dim=0)

    output = conv(
        x=(data["x"], x_dst),
        pos=(data["pos"], pos_dst),
        batch=(data["batch"], batch_dst),
        edge_index=edge_index,
    )

    assert output.shape == (num_dst, 32)


@pytest.mark.skipif(
    not _TORCH_CLUSTER_AVAILABLE and not _TORCH_SCATTER_AVAILABLE,
    reason="torch-cluster or torch-scatter is not installed",
)
def test_geometric_affine_conv_reset_parameters() -> None:
    """Test GeometricAffineConv reset_parameters."""
    conv = GeometricAffineConv(
        local_nn=nn.Identity(),
        channels=3,
        spatial_dim=3,
        use_pos=True,
        normalize="center",
    )

    conv.alpha.data.fill_(2.0)
    conv.beta.data.fill_(1.0)

    conv.reset_parameters()

    torch.testing.assert_close(conv.alpha, torch.ones_like(conv.alpha))
    torch.testing.assert_close(conv.beta, torch.zeros_like(conv.beta))
