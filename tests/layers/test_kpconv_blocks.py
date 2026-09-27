from pathlib import Path
from typing import Dict

import pytest
import torch
from torch import Tensor
from torch_geometric.utils import scatter

from torch_pointcloud.layers.kpconv_blocks import KPConv, KPConvBlock, KPResidualBlock, create_kernel_points


@pytest.fixture
def data() -> Dict[str, Tensor]:
    torch.manual_seed(42)
    lengths = torch.tensor([256, 512])
    pos = torch.randn(int(lengths.sum()), 3)
    features = torch.randn(int(lengths.sum()), 3)
    batch = torch.repeat_interleave(torch.arange(len(lengths)), lengths)

    # Dummy `[source, target]` edge_index connecting each point to 16 indices of its own cloud
    target = torch.arange(len(pos)).repeat_interleave(16)
    cumsum = torch.cat([torch.tensor([0]), torch.cumsum(lengths, dim=0)])
    source = torch.cat([torch.arange(int(lengths[i])).repeat(16) + cumsum[i] for i in range(len(lengths))])
    edge_index = torch.stack([source, target])

    return dict(
        features=features,
        pos=pos,
        batch=batch,
        edge_index=edge_index,
    )


def test_kpconv_module(data: Dict[str, Tensor]) -> None:
    conv = KPConv(
        spatial_dim=3,
        in_channels=3,
        out_channels=32,
        kernel_size=15,
        kp_radius=0.1,
        kp_sigma=0.1,
    )

    x = data["features"]
    pos = data["pos"]
    output = conv(x, pos, data["edge_index"])
    assert output.shape == (len(data["pos"]), 32)

    conv = KPConv(
        spatial_dim=3,
        in_channels=3,
        out_channels=32,
        kernel_size=15,
        kp_radius=0.1,
        kp_sigma=0.1,
        deformable=True,
        modulated=True,
    )

    x = data["features"]
    pos = data["pos"]
    output = conv(x, pos, data["edge_index"])
    assert output.shape == (len(data["pos"]), 32)


def test_kpconv_running_stats_are_not_buffers(data: Dict[str, Tensor]) -> None:
    conv = KPConv(
        spatial_dim=3,
        in_channels=3,
        out_channels=32,
        kernel_size=15,
        kp_radius=0.1,
        kp_sigma=0.1,
        deformable=True,
        modulated=True,
    )
    conv(data["features"], data["pos"], data["edge_index"])

    running_names = ("running_min_d2", "running_deformed_kernel", "running_offset_features")
    assert all(name not in conv.state_dict() for name in running_names)
    assert all(name not in dict(conv.named_buffers()) for name in running_names)
    assert all(getattr(conv, name) is not None for name in running_names)


def test_kpconv_block_layer(data: Dict[str, Tensor]) -> None:
    block = KPConvBlock(
        spatial_dim=3,
        in_channels=3,
        out_channels=32,
        kernel_size=15,
        kp_radius=0.1,
        kp_sigma=0.1,
    )

    output = block(data["features"], data["pos"], data["edge_index"])
    assert output.shape == (len(data["pos"]), 32)


def test_kpconv_residual_block(data: Dict[str, Tensor]) -> None:
    block = KPResidualBlock(
        spatial_dim=3,
        in_channels=3,
        out_channels=32,
        kernel_size=15,
        kp_radius=0.1,
        kp_sigma=0.1,
    )

    output = block(data["features"], data["pos"], data["edge_index"])
    assert output.shape == (len(data["pos"]), 32)

    block = KPResidualBlock(
        spatial_dim=3,
        in_channels=3,
        out_channels=32,
        kernel_size=15,
        kp_radius=0.1,
        kp_sigma=0.1,
        strided=True,
    )

    output = block(data["features"], data["pos"], data["edge_index"])
    assert output.shape == (len(data["pos"]), 32)


def test_kpconv_matches_per_edge_reference(data: Dict[str, Tensor]) -> None:
    conv = KPConv(spatial_dim=3, in_channels=3, out_channels=32, kernel_size=15, kp_radius=0.1, kp_sigma=0.1, bias=True)
    source, target = data["edge_index"]
    with torch.no_grad():
        out = conv(data["features"], data["pos"], data["edge_index"])
        weights = conv.message(data["pos"][target], data["pos"][source], None, None, target, len(data["pos"]))  # (E, K)
        x_j = data["features"][source]
        expected = conv.bias.clone()
        for k in range(conv.kernel_size):
            pooled = scatter(x_j * weights[:, k : k + 1], target, dim=0, dim_size=len(data["pos"]), reduce="sum")
            expected = expected + pooled @ conv.weight[k]
    assert torch.allclose(out, expected, atol=1e-5)


def test_kpconv_bias_is_added(data: Dict[str, Tensor]) -> None:
    conv = KPConv(spatial_dim=3, in_channels=3, out_channels=32, kernel_size=15, kp_radius=0.1, kp_sigma=0.1, bias=True)
    assert conv.bias is not None
    with torch.no_grad():
        conv.bias.fill_(2.0)
        output = conv(data["features"], data["pos"], data["edge_index"])
        conv.bias.zero_()
        output_no_bias = conv(data["features"], data["pos"], data["edge_index"])
    assert torch.allclose(output - output_no_bias, torch.full_like(output, 2.0))


def test_kpconv_bipartite(data: Dict[str, Tensor]) -> None:
    conv = KPConv(spatial_dim=3, in_channels=3, out_channels=32, kernel_size=15, kp_radius=0.1, kp_sigma=0.1)
    pos_target = data["pos"][::4]
    source = torch.arange(len(data["pos"]))
    target = torch.arange(len(pos_target)).repeat_interleave(4)
    edge_index = torch.stack([source, target])

    output = conv(data["features"], (data["pos"], pos_target), edge_index)
    assert output.shape == (len(pos_target), 32)

    block = KPResidualBlock(
        spatial_dim=3, in_channels=3, out_channels=32, kernel_size=15, kp_radius=0.1, kp_sigma=0.1, strided=True
    )
    output = block(data["features"], (data["pos"], pos_target), edge_index)
    assert output.shape == (len(pos_target), 32)


def test_create_kernel_points_gradient(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("torch_pointcloud.layers.kpconv_blocks.CACHE_DIR", tmp_path)
    torch.manual_seed(0)
    kernel_points = create_kernel_points(radius=0.05, num_points=7, method="gradient")
    assert kernel_points.shape == (7, 3)
    assert float(kernel_points.norm(dim=-1).max()) < 0.1
