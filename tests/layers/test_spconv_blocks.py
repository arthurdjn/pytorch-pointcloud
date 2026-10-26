from typing import Any, TypeVar

import pytest
import torch
import torch.nn.functional as F

from torch_pointcloud.layers.spconv_blocks import (
    SparseBasicBlock,
    SparseConvBlock,
    StridedSparseConv3d,
    SubMConv3dBlock,
)
from torch_pointcloud.utils.conversion import convert_to_spconv_tensor
from torch_pointcloud.utils.imports import _CUDA_AVAILABLE, _SPCONV_AVAILABLE, _SPCONV_GITHUB_URL, optional_import

spconv, _ = optional_import("spconv.pytorch", url=_SPCONV_GITHUB_URL)
ConvAlgo, _ = optional_import("spconv.core", "ConvAlgo", url=_SPCONV_GITHUB_URL)

pytestmark = pytest.mark.skipif(not _SPCONV_AVAILABLE, reason="spconv is not installed")
CUDA = pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA is not available")
DEVICES = pytest.mark.parametrize("device", [pytest.param("cpu"), pytest.param("cuda", marks=CUDA)])

M = TypeVar("M", bound=torch.nn.Module)


def _to_device(module: M, device: str) -> M:
    """Move `module` to `device`; on CPU a CUDA build of spconv runs only its native algorithm."""
    if device == "cpu":
        for layer in module.modules():
            if hasattr(layer, "algo"):
                layer.algo = ConvAlgo.Native
    return module.to(device)


@CUDA
def test_submconv3d_block_forward() -> None:
    block = SubMConv3dBlock(
        in_channels=8,
        out_channels=16,
        kernel_size=3,
        padding=1,
        norm="batch_norm",
        act="relu",
        act_kwargs=None,
        norm_kwargs=None,
        bias=True,
        stem_indice_key=None,
    ).cuda()

    x = torch.randn(20, 8).cuda()
    pos = torch.randint(0, 32, (20, 3)).cuda()
    batch = torch.cat([torch.zeros(8), torch.ones(12)]).long().cuda()

    out = block(x, pos, batch)
    assert out.shape == (20, 16)


def test_sparse_conv_block_state_dict_positions() -> None:
    block = SparseConvBlock(4, 8, 3, indice_key="s1")
    keys = set(block.state_dict().keys())
    assert "0.weight" in keys
    assert "1.module.weight" in keys and "1.module.bias" in keys


def test_sparse_conv_block_none_norm_act_builds_identity() -> None:
    block = SparseConvBlock(4, 8, 3, indice_key="s1", norm=None, act=None)
    modules = dict(block.named_modules())
    assert isinstance(modules["1"], torch.nn.Identity)
    assert isinstance(modules["2"], torch.nn.Identity)


@CUDA
def test_sparse_conv_block_none_norm_act_forward() -> None:
    import spconv.pytorch as spconv

    block = SparseConvBlock(4, 8, 3, indice_key="s1", norm=None, act=None).cuda()
    x = torch.randn(20, 4).cuda()
    indices = (
        torch.cat(
            [torch.cat([torch.zeros(8), torch.ones(12)]).long().unsqueeze(-1), torch.randint(0, 16, (20, 3))],
            dim=1,
        )
        .int()
        .cuda()
    )
    sparse_x = spconv.SparseConvTensor(x, indices, spatial_shape=[16, 16, 16], batch_size=2)
    out = block(sparse_x)
    assert out.features.shape == (20, 8)


def test_sparse_conv_block_invalid_conv_type_raises() -> None:
    with pytest.raises(ValueError, match="Unknown conv_type"):
        SparseConvBlock(4, 8, 3, indice_key="s1", conv_type="invalid")


@CUDA
def test_sparse_basic_block_projects_skip() -> None:
    torch.manual_seed(0)
    x = torch.randn(20, 8).cuda()
    pos = torch.randint(0, 32, (20, 3)).cuda()
    batch = torch.cat([torch.zeros(8), torch.ones(12)]).long().cuda()
    x_sparse = convert_to_spconv_tensor(x, pos, batch)

    block = SparseBasicBlock(8, 16, indice_key="res").cuda()
    assert block(x_sparse).features.shape == (20, 16)

    same = SparseBasicBlock(8, 8, bias=True, indice_key="res").cuda()
    assert not any(p.numel() for p in same.proj.parameters())
    assert same(x_sparse).features.shape == (20, 8)


def _random_sparse_tensor(size: int = 12, channels: int = 3, seed: int = 0, device: str = "cuda") -> Any:
    generator = torch.Generator().manual_seed(seed)
    voxels = (torch.rand(size, size, size, generator=generator) < 0.2).nonzero().int()
    indices = torch.cat([torch.zeros(len(voxels), 1, dtype=torch.int32), voxels], 1).contiguous().to(device)
    features = torch.randn(len(voxels), channels, generator=generator).to(device)
    return spconv.SparseConvTensor(features, indices, [size, size, size], 1)


@DEVICES
def test_strided_sparse_conv3d_outputs_the_parent_voxels(device: str) -> None:
    x = _random_sparse_tensor(device=device)
    out = _to_device(StridedSparseConv3d(3, 4, kernel_size=3, stride=2), device)(x)

    parents = torch.unique(torch.cat([x.indices[:, :1], x.indices[:, 1:] // 2], 1), dim=0)
    assert torch.equal(out.indices, parents.int())
    assert out.spatial_shape == [6, 6, 6]


@DEVICES
def test_strided_sparse_conv3d_matches_a_dense_convolution(device: str) -> None:
    x = _random_sparse_tensor(device=device)
    conv = _to_device(StridedSparseConv3d(3, 4, kernel_size=3, stride=2), device)
    out = conv(x)

    dense = F.conv3d(x.dense(), conv.weight.permute(0, 4, 1, 2, 3), stride=2, padding=1)
    index = out.indices.long()
    expected = dense[0, :, index[:, 1], index[:, 2], index[:, 3]].T
    torch.testing.assert_close(out.features, expected, atol=1e-4, rtol=1e-4)


@DEVICES
def test_strided_sparse_conv3d_pointwise_kernel_is_zero_without_an_even_child(device: str) -> None:
    indices = torch.tensor([[0, 0, 0, 0], [0, 3, 1, 1]], dtype=torch.int32, device=device)
    x = spconv.SparseConvTensor(torch.ones(2, 2, device=device), indices, [4, 4, 4], 1)
    conv = _to_device(StridedSparseConv3d(2, 1, kernel_size=1, stride=2), device)
    out = conv(x)

    assert out.indices.tolist() == [[0, 0, 0, 0], [0, 1, 0, 0]]
    assert out.features[0].item() == pytest.approx(conv.weight.sum().item(), rel=1e-5)
    assert out.features[1].item() == 0.0


@DEVICES
def test_sparse_basic_block_stride_two_halves_the_resolution(device: str) -> None:
    x = _random_sparse_tensor(channels=8, device=device)
    block = _to_device(SparseBasicBlock(8, 16, stride=2, indice_key="subm"), device).eval()
    out = block(x)

    parents = torch.unique(torch.cat([x.indices[:, :1], x.indices[:, 1:] // 2], 1), dim=0)
    assert torch.equal(out.indices, parents.int())
    assert out.features.shape == (len(parents), 16)
    assert set(block.state_dict()) >= {"conv1.weight", "proj.0.weight", "proj.1.module.weight", "conv2.weight"}


@DEVICES
def test_sparse_conv_block_transposeconv_generates_children(device: str) -> None:
    x = _random_sparse_tensor(size=6, channels=4, device=device)
    block = SparseConvBlock(4, 2, 2, stride=2, indice_key=None, conv_type="transposeconv")
    out = _to_device(block, device).eval()(x)

    offsets = torch.cartesian_prod(*[torch.arange(2)] * 3).int().to(device)
    children = (x.indices[:, None, 1:] * 2 + offsets).view(-1, 3)
    assert len(out.indices) == len(children)
    assert out.spatial_shape == [12, 12, 12]
