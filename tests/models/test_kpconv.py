from pathlib import Path
from typing import Dict

import pytest
import torch
from torch import Tensor

from torch_pointcloud.models.kpconv import (
    EncoderBlock,
    GridPool,
    KPConv,
    KPConvBlock,
    KPFCNNClassification,
    KPFCNNSegmentation,
    KPResidualBlock,
    create_kernel_points,
)
from torch_pointcloud.utils.imports import _TORCH_CLUSTER_AVAILABLE, _TORCH_SCATTER_AVAILABLE

# See: https://docs.pytest.org/en/stable/how-to/skipping.html#summary
pytestmark = pytest.mark.skipif(
    not _TORCH_CLUSTER_AVAILABLE and not _TORCH_SCATTER_AVAILABLE,
    reason="torch-cluster or torch-scatter is not installed",
)


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
    conv(data["features"], data["pos"], data["pos"], data["edge_index"])

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

    output = block(data["features"], data["pos"], data["pos"], data["edge_index"])
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
def model_clf() -> KPFCNNClassification:
    return KPFCNNClassification(
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
def model_seg() -> KPFCNNSegmentation:
    return KPFCNNSegmentation(
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


def test_kpconv_clf_forward(model_clf: KPFCNNClassification, data: Dict[str, Tensor]) -> None:
    logits = model_clf(data["features"], data["pos"], data["batch"])
    assert logits.shape == (data["batch"].max() + 1, model_clf.num_classes)


def test_kpconv_clf_forward_x_none_uses_pos(model_clf: KPFCNNClassification, data: Dict[str, Tensor]) -> None:
    logits = model_clf(None, data["pos"], data["batch"])
    assert logits.shape == (data["batch"].max() + 1, model_clf.num_classes)


def test_kpconv_clf_forward_x_none_channel_mismatch_raises(data: Dict[str, Tensor]) -> None:
    model = KPFCNNClassification(
        in_channels=6,
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
    with pytest.raises(ValueError, match="in_channels=6"):
        model(None, data["pos"], data["batch"])


def test_kpconv_seg_forward_x_none_channel_mismatch_raises(data: Dict[str, Tensor]) -> None:
    model = KPFCNNSegmentation(
        in_channels=6,
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
    with pytest.raises(ValueError, match="in_channels=6"):
        model(None, data["pos"], data["batch"])


def test_kpconv_clf_reset_classifier(model_clf: KPFCNNClassification, data: Dict[str, Tensor]) -> None:
    new_num_classes = 20
    model_clf.reset_classifier(new_num_classes)

    assert model_clf.num_classes == new_num_classes
    assert model_clf.head.out_features == new_num_classes
    logits = model_clf(data["features"], data["pos"], data["batch"])
    assert logits.shape == (data["batch"].max() + 1, new_num_classes)


def test_kpconv_clf_reset_classifier_keeps_current_pooling(model_clf: KPFCNNClassification) -> None:
    model_clf.reset_classifier(10, global_pool="mean")
    pool = model_clf.global_pool
    model_clf.reset_classifier(7)
    assert model_clf.global_pool is pool
    assert type(pool).__name__ == "MeanPool"


def test_kpconv_clf_forward_features(model_clf: KPFCNNClassification, data: Dict[str, Tensor]) -> None:
    out_x, out_pos, out_batch = model_clf.forward_features(data["features"], data["pos"], data["batch"])
    assert out_x.dim() == 2
    assert out_pos.dim() == 2
    assert out_batch.dim() == 1

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


def test_kpconv_clf_forward_features_and_head(model_clf: KPFCNNClassification, data: Dict[str, Tensor]) -> None:
    out_x, _, out_batch = model_clf.forward_features(data["features"], data["pos"], data["batch"])
    logits = model_clf.forward_head(out_x, out_batch)
    assert logits.shape == (data["batch"].max() + 1, model_clf.num_classes)


def test_kpconv_seg_forward(model_seg: KPFCNNSegmentation, data: Dict[str, Tensor]) -> None:
    logits = model_seg(data["features"], data["pos"], data["batch"])
    assert logits.shape == (data["pos"].shape[0], model_seg.num_classes)


def test_kpconv_seg_forward_features(model_seg: KPFCNNSegmentation, data: Dict[str, Tensor]) -> None:
    out_x, out_pos, out_batch = model_seg.forward_features(data["features"], data["pos"], data["batch"])
    assert out_x.shape[0] == out_pos.shape[0] == out_batch.shape[0]
    assert out_x.dim() == 2
    assert out_pos.dim() == 2
    assert out_batch.dim() == 1


def test_kpconv_seg_forward_features_and_head(model_seg: KPFCNNSegmentation, data: Dict[str, Tensor]) -> None:
    out_x, out_pos, out_batch, intermediates = model_seg.forward_features(
        data["features"],
        data["pos"],
        data["batch"],
        return_intermediates=True,
    )

    out_x = model_seg.forward_decoder(out_x, out_pos, out_batch, intermediates)
    logits = model_seg.forward_head(out_x)
    assert logits.shape == (data["pos"].shape[0], model_seg.num_classes)


def test_create_kernel_points_gradient(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("torch_pointcloud.models.kpconv.CACHE_DIR", tmp_path)
    torch.manual_seed(0)
    kernel_points = create_kernel_points(radius=0.05, num_points=7, method="gradient")
    assert kernel_points.shape == (7, 3)
    assert float(kernel_points.norm(dim=-1).max()) < 0.1


def test_kpconv_seg_reset_classifier_keeps_head_channels() -> None:
    model = KPFCNNSegmentation(
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
        head_channels=[8],
    )
    model.reset_classifier(num_classes=7)
    assert isinstance(model.head, torch.nn.Sequential)
    assert model.head[-1].out_features == 7
    hidden = model.head[0]
    assert isinstance(hidden, torch.nn.Sequential)
    assert hidden[0].out_features == 8
