from typing import Dict

import pytest
import torch
from torch import Tensor

from torch_pointcloud.models.pointnet2 import PointNet2Classification, PointNet2Encoder, PointNet2Segmentation
from torch_pointcloud.utils.imports import _TORCH_CLUSTER_AVAILABLE

# See: https://docs.pytest.org/en/stable/how-to/skipping.html#summary
pytestmark = pytest.mark.skipif(
    not _TORCH_CLUSTER_AVAILABLE,
    reason="torch-cluster is not installed",
)


@pytest.fixture
def data() -> Dict[str, Tensor]:
    torch.manual_seed(42)
    lengths = torch.tensor([256, 512])
    pos = torch.randn(int(lengths.sum()), 3)
    x = torch.randn(int(lengths.sum()), 6)
    batch = torch.repeat_interleave(torch.arange(len(lengths)), lengths)
    return dict(x=x, pos=pos, batch=batch)


@pytest.fixture
def model_clf() -> PointNet2Classification:
    return PointNet2Classification(
        in_channels=6,
        num_classes=10,
        stem_channels=None,
        sa_channels=[[32, 64], [64, 128]],
        aggr_channels=None,
        ratios=[0.5, 0.5],
        radii=[0.2, 0.4],
        num_neighbors=[16, 16],
        spatial_dim=3,
        act="relu",
        act_kwargs=None,
        act_first=False,
        norm="batch_norm",
        norm_kwargs=None,
        bias=False,
        use_pos=True,
        pool="max",
        dropout=0.0,
        global_pool="max",
    )


@pytest.fixture
def model_seg() -> PointNet2Segmentation:
    return PointNet2Segmentation(
        in_channels=6,
        num_classes=10,
        stem_channels=None,
        sa_channels=[[32, 64], [64, 128]],
        aggr_channels=None,
        fp_channels=[[64, 64], [64, 32]],
        ratios=[0.5, 0.5],
        radii=[0.2, 0.4],
        num_neighbors=[16, 16],
        spatial_dim=3,
        act="relu",
        act_kwargs=None,
        act_first=False,
        norm="batch_norm",
        norm_kwargs=None,
        bias=False,
        use_pos=True,
        pool="max",
        dropout=0.0,
    )


def test_pointnet2_classification_head_per_layer_dropout() -> None:
    model = PointNet2Classification(
        in_channels=6,
        num_classes=10,
        sa_channels=[[32, 64], [64, 128]],
        ratios=[0.5, 0.5],
        radii=[0.2, 0.4],
        num_neighbors=[16, 16],
        head_channels=[64, 32],
        dropout=[0.4, 0.5],
    )
    assert model.head.dropout == [0.4, 0.5, 0.0]
    with pytest.raises(ValueError, match="one rate per head layer"):
        PointNet2Classification(
            in_channels=6,
            num_classes=10,
            sa_channels=[[32, 64], [64, 128]],
            ratios=[0.5, 0.5],
            radii=[0.2, 0.4],
            num_neighbors=[16, 16],
            head_channels=[64, 32],
            dropout=[0.4],
        )


def test_pointnet2_classification_forward(model_clf: PointNet2Classification, data: Dict[str, Tensor]) -> None:
    logits = model_clf(data["x"], data["pos"], data["batch"])
    assert logits.shape == (int(data["batch"].max()) + 1, model_clf.num_classes)
    assert logits.dtype == data["x"].dtype


def test_pointnet2_classification_reset_classifier(model_clf: PointNet2Classification, data: Dict[str, Tensor]) -> None:
    model_clf.reset_classifier(num_classes=42)
    logits = model_clf(data["x"], data["pos"], data["batch"])
    assert logits.shape == (int(data["batch"].max()) + 1, 42)


def test_pointnet2_segmentation_forward(model_seg: PointNet2Segmentation, data: Dict[str, Tensor]) -> None:
    logits = model_seg(data["x"], data["pos"], data["batch"])
    assert logits.shape == (data["pos"].shape[0], model_seg.num_classes)
    assert logits.dtype == data["x"].dtype


def test_pointnet2_segmentation_reset_classifier(model_seg: PointNet2Segmentation, data: Dict[str, Tensor]) -> None:
    model_seg.reset_classifier(num_classes=42)
    logits = model_seg(data["x"], data["pos"], data["batch"])
    assert logits.shape == (data["pos"].shape[0], 42)


def test_pointnet2_classification_forward_features_and_head(
    model_clf: PointNet2Classification, data: Dict[str, Tensor]
) -> None:
    x, _, batch = model_clf.forward_features(data["x"], data["pos"], data["batch"])
    assert x.shape[0] == batch.shape[0]
    logits = model_clf.forward_head(x, batch)
    assert logits.shape == (int(data["batch"].max()) + 1, model_clf.num_classes)


def test_pointnet2_segmentation_forward_features_decoder_head(
    model_seg: PointNet2Segmentation, data: Dict[str, Tensor]
) -> None:
    x, pos, batch, intermediates = model_seg.forward_features(
        data["x"], data["pos"], data["batch"], return_intermediates=True
    )
    assert len(intermediates) > 0
    x = model_seg.forward_decoder(x, pos, batch, intermediates)
    assert x.shape[0] == data["pos"].shape[0]
    logits = model_seg.forward_head(x)
    assert logits.shape == (data["pos"].shape[0], model_seg.num_classes)


def test_pointnet2_encoder_num_points_and_pos_first(data: Dict[str, Tensor]) -> None:
    encoder = PointNet2Encoder(
        in_channels=6,
        sa_channels=[[32, 64], [64, 128]],
        num_points=[128, 32],
        radii=[0.2, 0.4],
        num_neighbors=[16, 16],
        normalize_pos=False,
        pos_first=True,
    )
    x, pos, batch = encoder(data["x"], data["pos"], data["batch"])
    # `num_points` samples a fixed number of centroids per scene, regardless of the scene sizes
    assert x.shape == (2 * 32, 128)
    assert pos.shape == (2 * 32, 3)
    assert batch.shape == (2 * 32,)


def test_pointnet2_encoder_needs_ratios_or_num_points() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        PointNet2Encoder(in_channels=6, sa_channels=[[32, 64]], radii=[0.2], num_neighbors=[16])
    with pytest.raises(ValueError, match="exactly one"):
        PointNet2Encoder(
            in_channels=6, sa_channels=[[32, 64]], ratios=[0.5], num_points=[128], radii=[0.2], num_neighbors=[16]
        )


def test_pointnet2_classification_num_classes_zero_returns_features(data: Dict[str, Tensor]) -> None:
    model = PointNet2Classification(
        in_channels=6,
        num_classes=0,
        sa_channels=[[32, 64], [64, 128]],
        ratios=[0.5, 0.5],
        radii=[0.2, 0.4],
        num_neighbors=[16, 16],
        head_channels=[32],
    )
    assert isinstance(model.head, torch.nn.Identity)
    out = model(data["x"], data["pos"], data["batch"])
    assert out.shape == (int(data["batch"].max()) + 1, model.num_features)


def test_pointnet2_reset_classifier_keeps_current_pooling(model_clf: PointNet2Classification) -> None:
    model_clf.reset_classifier(10, global_pool="mean")
    pool = model_clf.global_pool
    model_clf.reset_classifier(5)
    assert model_clf.global_pool is pool
    assert type(pool).__name__ == "MeanPool"
