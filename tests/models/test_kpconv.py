from typing import Dict

import pytest
import torch
from torch import Tensor

from torch_pointcloud.models.kpconv import (
    EncoderBlock,
    GridPool,
    KPConv,
    KPConvBlock,
    KPConvNetClassification,
    KPConvNetSegmentation,
    KPResidualBlock,
)
from torch_pointcloud.utils.imports import _TORCH_CLUSTER_AVAILABLE, _TORCH_SCATTER_AVAILABLE


@pytest.fixture
def data() -> Dict[str, Tensor]:
    torch.manual_seed(42)
    lengths = torch.tensor([256, 512])
    pos = torch.randn(int(lengths.sum()), 3)
    features = torch.randn(int(lengths.sum()), 3)
    batch = torch.repeat_interleave(torch.arange(len(lengths)), lengths)

    # Dummy edge_index connecting each point to 16 nearest indices
    row = torch.arange(len(pos)).repeat_interleave(16)
    cumsum = torch.cat([torch.tensor([0]), torch.cumsum(lengths, dim=0)])
    col = torch.cat([torch.arange(int(lengths[i])).repeat(16) + cumsum[i] for i in range(len(lengths))])
    edge_index = torch.stack([row, col])

    return dict(
        features=features,
        pos=pos,
        batch=batch,
        edge_index=edge_index,
    )


@pytest.mark.skipif(
    not _TORCH_CLUSTER_AVAILABLE and not _TORCH_SCATTER_AVAILABLE,
    reason="torch-cluster or torch-scatter is not installed",
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
    output = conv(x, pos, pos, data["edge_index"])
    assert output.shape == (len(data["pos"]), 32)

    # Test with deformable and modulated options
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
    output = conv(x, pos, pos, data["edge_index"])
    assert output.shape == (len(data["pos"]), 32)


@pytest.mark.skipif(
    not _TORCH_CLUSTER_AVAILABLE and not _TORCH_SCATTER_AVAILABLE,
    reason="torch-cluster or torch-scatter is not installed",
)
def test_kpconv_block_layer(data: Dict[str, Tensor]) -> None:
    block = KPConvBlock(
        spatial_dim=3,
        in_channels=3,
        out_channels=32,
        kernel_size=15,
        kp_radius=0.1,
        kp_sigma=0.1,
    )

    output = block(data["features"], data["pos"], data["pos"], data["edge_index"])
    assert output.shape == (len(data["pos"]), 32)


@pytest.mark.skipif(
    not _TORCH_CLUSTER_AVAILABLE and not _TORCH_SCATTER_AVAILABLE,
    reason="torch-cluster or torch-scatter is not installed",
)
def test_kpconv_residual_block(data: Dict[str, Tensor]) -> None:
    block = KPResidualBlock(
        spatial_dim=3,
        in_channels=3,
        out_channels=32,
        kernel_size=15,
        kp_radius=0.1,
        kp_sigma=0.1,
    )

    output = block(data["features"], data["pos"], data["pos"], data["edge_index"])
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

    output = block(data["features"], data["pos"], data["pos"], data["edge_index"])
    assert output.shape == (len(data["pos"]), 32)


@pytest.mark.skipif(
    not _TORCH_CLUSTER_AVAILABLE and not _TORCH_SCATTER_AVAILABLE,
    reason="torch-cluster or torch-scatter is not installed",
)
def test_encoder_block(data: Dict[str, Tensor]) -> None:
    block = EncoderBlock(
        depth=2,
        radius=0.1,
        max_num_neighbors=16,
        spatial_dim=3,
        in_channels=3,
        out_channels=32,
        kernel_size=15,
        kp_radius=0.1,
        kp_sigma=0.1,
        downsample=None,
    )

    out_x, out_pos, out_batch = block(data["features"], data["pos"], data["batch"])
    assert out_x.shape == (len(data["pos"]), 32)
    assert out_pos.shape == data["pos"].shape
    assert out_batch.shape == data["batch"].shape

    block = EncoderBlock(
        depth=2,
        radius=0.1,
        max_num_neighbors=16,
        spatial_dim=3,
        in_channels=3,
        out_channels=32,
        kernel_size=15,
        kp_radius=0.1,
        kp_sigma=0.1,
        downsample=GridPool(grid_size=0.5),
    )

    out_x, out_pos, out_batch, inverse = block(
        data["features"],
        data["pos"],
        data["batch"],
        return_inverse=True,
    )
    assert out_x.shape[1] == 32
    assert out_x.shape[0] == out_pos.shape[0] == out_batch.shape[0]
    assert len(out_pos) < len(data["pos"])  # Should be downsampled
    assert inverse.shape == (len(data["pos"]),)


@pytest.fixture
def model_clf() -> KPConvNetClassification:
    return KPConvNetClassification(
        in_channels=3,
        num_classes=10,
        encoder_depths=[2, 2],
        encoder_channels=[32, 64],
        encoder_num_neighbors=[16, 16],
        grid_sizes=[0.1],
        radii=[0.1, 0.2],
        kernel_size=15,
        kp_radius=0.1,
        kp_sigma=0.1,
    )


@pytest.fixture
def model_seg() -> KPConvNetSegmentation:
    return KPConvNetSegmentation(
        in_channels=3,
        num_classes=10,
        encoder_depths=[2, 2],
        encoder_channels=[32, 64],
        encoder_num_neighbors=[16, 16],
        fp_channels=[[32], [16]],
        grid_sizes=[0.1],
        radii=[0.1, 0.2],
        kernel_size=15,
        kp_radius=0.1,
        kp_sigma=0.1,
    )


@pytest.mark.skipif(
    not _TORCH_CLUSTER_AVAILABLE and not _TORCH_SCATTER_AVAILABLE,
    reason="torch-cluster or torch-scatter is not installed",
)
def test_kpconv_clf_forward(model_clf: KPConvNetClassification, data: Dict[str, Tensor]) -> None:
    logits = model_clf(data["features"], data["pos"], data["batch"])
    assert logits.shape == (data["batch"].max() + 1, model_clf.num_classes)


@pytest.mark.skipif(
    not _TORCH_CLUSTER_AVAILABLE and not _TORCH_SCATTER_AVAILABLE,
    reason="torch-cluster or torch-scatter is not installed",
)
def test_kpconv_clf_reset_classifier(model_clf: KPConvNetClassification, data: Dict[str, Tensor]) -> None:
    new_num_classes = 20
    model_clf.reset_classifier(new_num_classes)

    assert model_clf.num_classes == new_num_classes
    assert model_clf.head.out_features == new_num_classes
    logits = model_clf(data["features"], data["pos"], data["batch"])
    assert logits.shape == (data["batch"].max() + 1, new_num_classes)


@pytest.mark.skipif(
    not _TORCH_CLUSTER_AVAILABLE and not _TORCH_SCATTER_AVAILABLE,
    reason="torch-cluster or torch-scatter is not installed",
)
def test_kpconv_clf_forward_features(model_clf: KPConvNetClassification, data: Dict[str, Tensor]) -> None:
    out_x, out_pos, out_batch = model_clf.forward_features(data["features"], data["pos"], data["batch"])
    assert out_x.dim() == 2
    assert out_pos.dim() == 2
    assert out_batch.dim() == 1

    # Test forward features with intermediates
    out_x, out_pos, out_batch, intermediates = model_clf.forward_features(
        data["features"],
        data["pos"],
        data["batch"],
        return_intermediates=True,
    )
    assert len(intermediates) == len(model_clf.encoder_blocks) - 1
    for intermediate in intermediates:
        assert "x" in intermediate
        assert "pos" in intermediate
        assert "batch" in intermediate
        if "pooling_inverse" in intermediate:
            assert intermediate["pooling_inverse"].dim() == 1


@pytest.mark.skipif(
    not _TORCH_CLUSTER_AVAILABLE and not _TORCH_SCATTER_AVAILABLE,
    reason="torch-cluster or torch-scatter is not installed",
)
def test_kpconv_clf_forward_features_and_head(model_clf: KPConvNetClassification, data: Dict[str, Tensor]) -> None:
    out_x, _, out_batch = model_clf.forward_features(data["features"], data["pos"], data["batch"])
    logits = model_clf.forward_head(out_x, out_batch)
    assert logits.shape == (data["batch"].max() + 1, model_clf.num_classes)


@pytest.mark.skipif(
    not _TORCH_CLUSTER_AVAILABLE and not _TORCH_SCATTER_AVAILABLE,
    reason="torch-cluster or torch-scatter is not installed",
)
def test_kpconv_seg_forward(model_seg: KPConvNetSegmentation, data: Dict[str, Tensor]) -> None:
    logits = model_seg(data["features"], data["pos"], data["batch"])
    assert logits.shape == (data["pos"].shape[0], model_seg.num_classes)


@pytest.mark.skipif(
    not _TORCH_CLUSTER_AVAILABLE and not _TORCH_SCATTER_AVAILABLE,
    reason="torch-cluster or torch-scatter is not installed",
)
def test_kpconv_seg_forward_features(model_seg: KPConvNetSegmentation, data: Dict[str, Tensor]) -> None:
    out_x, out_pos, out_batch = model_seg.forward_features(data["features"], data["pos"], data["batch"])
    assert out_x.shape[0] == out_pos.shape[0] == out_batch.shape[0]
    assert out_x.dim() == 2
    assert out_pos.dim() == 2
    assert out_batch.dim() == 1


@pytest.mark.skipif(
    not _TORCH_CLUSTER_AVAILABLE and not _TORCH_SCATTER_AVAILABLE,
    reason="torch-cluster or torch-scatter is not installed",
)
def test_kpconv_seg_forward_features_and_head(model_seg: KPConvNetSegmentation, data: Dict[str, Tensor]) -> None:
    out_x, out_pos, out_batch, intermediates = model_seg.forward_features(
        data["features"],
        data["pos"],
        data["batch"],
        return_intermediates=True,
    )

    out_x = model_seg.forward_decoder(out_x, out_pos, out_batch, intermediates)
    logits = model_seg.forward_head(out_x)
    assert logits.shape == (data["pos"].shape[0], model_seg.num_classes)
