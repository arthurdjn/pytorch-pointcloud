from typing import Dict

import pytest
import torch
from torch import Tensor

from torch_pointcloud.models.pointconv import (
    PointConvDensityClassification,
    PointConvDensityEncoder,
)
from torch_pointcloud.utils.imports import _TORCH_CLUSTER_AVAILABLE, _TORCH_SCATTER_AVAILABLE


@pytest.fixture
def data() -> Dict[str, Tensor]:
    torch.manual_seed(42)
    lengths = torch.tensor([256, 512])
    pos = torch.randn(int(lengths.sum()), 3)
    features = torch.randn(int(lengths.sum()), 64)  # Matches default in_channels for test
    batch = torch.repeat_interleave(torch.arange(len(lengths)), lengths)

    return dict(
        features=features,
        pos=pos,
        batch=batch,
    )


@pytest.fixture
def model_clf() -> PointConvDensityClassification:
    return PointConvDensityClassification(
        in_channels=64,
        num_classes=10,
        channels=[[64, 64], [64, 128]],
        num_neighbors=[16, 16],
        bandwidths=[0.1, 0.2],
        ratios=[0.5, 0.5],
        density_channels=[16, 8],
        weight_channels=[8, 8],
    )


@pytest.mark.skipif(
    not _TORCH_CLUSTER_AVAILABLE and not _TORCH_SCATTER_AVAILABLE,
    reason="torch-cluster or torch-scatter is not installed",
)
def test_pointconv_encoder_basic(data: Dict[str, Tensor]) -> None:
    """Test basic PointConvDensityEncoder functionality."""
    encoder = PointConvDensityEncoder(
        in_channels=64,
        channels=[[64, 64], [64, 128]],
        num_neighbors=[16, 16],
        bandwidths=[0.1, 0.2],
        ratios=[0.5, 0.5],
        density_channels=[16, 8],
        weight_channels=[8, 8],
        spatial_dim=3,
    )

    out_features, out_pos, out_batch = encoder(data["features"], data["pos"], data["batch"])

    assert out_features.shape[0] == out_pos.shape[0] == out_batch.shape[0]
    assert out_features.shape[1] == 128  # Last channel of last layer
    assert out_pos.shape[1] == 3
    assert out_batch.shape[0] < data["batch"].shape[0]  # Should be downsampled


@pytest.mark.skipif(
    not _TORCH_CLUSTER_AVAILABLE and not _TORCH_SCATTER_AVAILABLE,
    reason="torch-cluster or torch-scatter is not installed",
)
def test_pointconv_encoder_with_intermediates(data: Dict[str, Tensor]) -> None:
    """Test PointConvDensityEncoder with intermediate outputs."""
    encoder = PointConvDensityEncoder(
        in_channels=64,
        channels=[[64, 64], [64, 128]],
        num_neighbors=[16, 16],
        bandwidths=[0.1, 0.2],
        ratios=[0.5, 0.5],
        density_channels=[16, 8],
        weight_channels=[8, 8],
        spatial_dim=3,
    )

    _, _, _, intermediates = encoder(
        data["features"],
        data["pos"],
        data["batch"],
        return_intermediates=True,
    )

    assert len(intermediates) == 2  # Number of layers
    for intermediate in intermediates:
        assert hasattr(intermediate, "x")
        assert hasattr(intermediate, "pos")
        assert hasattr(intermediate, "batch")
        assert intermediate.x.shape[0] == intermediate.pos.shape[0] == intermediate.batch.shape[0]


@pytest.mark.skipif(
    not _TORCH_CLUSTER_AVAILABLE and not _TORCH_SCATTER_AVAILABLE,
    reason="torch-cluster or torch-scatter is not installed",
)
def test_pointconv_classification_forward(model_clf: PointConvDensityClassification, data: Dict[str, Tensor]) -> None:
    """Test PointConvDensityClassification forward pass."""
    logits = model_clf(data["features"], data["pos"], data["batch"])
    assert logits.shape == (data["batch"].max() + 1, model_clf.num_classes)
    assert logits.dtype == data["features"].dtype


@pytest.mark.skipif(
    not _TORCH_CLUSTER_AVAILABLE and not _TORCH_SCATTER_AVAILABLE,
    reason="torch-cluster or torch-scatter is not installed",
)
def test_pointconv_classification_reset_classifier(
    model_clf: PointConvDensityClassification, data: Dict[str, Tensor]
) -> None:
    """Test PointConvDensityClassification reset_classifier."""
    new_num_classes = 5
    model_clf.reset_classifier(num_classes=new_num_classes)
    assert model_clf.num_classes == new_num_classes

    logits = model_clf(data["features"], data["pos"], data["batch"])
    assert logits.shape == (data["batch"].max() + 1, new_num_classes)


@pytest.mark.skipif(
    not _TORCH_CLUSTER_AVAILABLE and not _TORCH_SCATTER_AVAILABLE,
    reason="torch-cluster or torch-scatter is not installed",
)
def test_pointconv_classification_forward_features(
    model_clf: PointConvDensityClassification, data: Dict[str, Tensor]
) -> None:
    """Test PointConvDensityClassification forward_features."""
    out_features, out_pos, out_batch = model_clf.forward_features(data["features"], data["pos"], data["batch"])

    assert out_features.dim() == 2
    assert out_pos.dim() == 2
    assert out_batch.dim() == 1

    # Test with intermediates
    out_features, out_pos, out_batch, intermediates = model_clf.forward_features(
        data["features"], data["pos"], data["batch"], return_intermediates=True
    )
    assert len(intermediates) == len(model_clf.channels)
