from typing import Dict

import pytest
import torch
from torch import Tensor

from torch_pointcloud.models.pointnext import (
    PointNeXtClassification,
    PointNeXtDecoder,
    PointNeXtEncoder,
    PointNeXtEncoderBlock,
    PointNeXtSegmentation,
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
    features = torch.randn(int(lengths.sum()), 6)
    batch = torch.repeat_interleave(torch.arange(len(lengths)), lengths)

    return dict(
        features=features,
        pos=pos,
        batch=batch,
    )


@pytest.fixture
def model_clf() -> PointNeXtClassification:
    return PointNeXtClassification(
        in_channels=6,
        num_classes=10,
        stem_channels=32,
        encoder_channels=[32, 64, 128],
        encoder_depths=[2, 2, 2],
        ratios=[0.5, 0.5, 0.5, 0.5],
        radiuses=[0.1, 0.2, 0.4, 0.8],
        num_neighbors=[16, 16, 16, 16],
    )


@pytest.fixture
def model_seg() -> PointNeXtSegmentation:
    return PointNeXtSegmentation(
        in_channels=6,
        num_classes=10,
        stem_channels=32,
        encoder_channels=[32, 64, 128],
        encoder_depths=[2, 2, 2],
        decoder_channels=[128, 64, 32],
        decoder_depths=[2, 2, 2],
        ratios=[0.5, 0.5, 0.5, 0.5],
        radiuses=[0.1, 0.2, 0.4, 0.8],
        num_neighbors=[16, 16, 16, 16],
    )


def test_pointnext_encoder_block_basic(data: Dict[str, Tensor]) -> None:
    """Test basic PointNeXtEncoderBlock functionality."""
    block = PointNeXtEncoderBlock(
        spatial_dim=3,
        channels=6,
        depth=2,
        expansion=4,
        ratio=0.5,
        radius=0.1,
        num_neighbors=16,
    )

    out_features, out_pos, out_batch = block(data["features"], data["pos"], data["batch"])

    assert out_features.shape[0] == out_pos.shape[0] == out_batch.shape[0]
    assert out_features.shape[1] == 6
    assert out_pos.shape[1] == 3
    assert out_batch.shape[0] <= data["batch"].shape[0]  # May be downsampled


def test_pointnext_encoder_basic(data: Dict[str, Tensor]) -> None:
    """Test basic PointNeXtEncoder functionality."""
    encoder = PointNeXtEncoder(
        channels=[6, 32, 64, 128],
        depths=[2, 2, 2],
        ratios=[0.5, 0.5, 0.5, 0.5],
        radiuses=[0.1, 0.2, 0.4, 0.8],
        num_neighbors=[16, 16, 16, 16],
    )

    out_features, out_pos, out_batch = encoder(data["features"], data["pos"], data["batch"])

    assert out_features.shape[0] == out_pos.shape[0] == out_batch.shape[0]
    assert out_features.shape[1] == 128  # Last channel
    assert out_pos.shape[1] == 3


def test_pointnext_encoder_with_intermediates(data: Dict[str, Tensor]) -> None:
    """Test PointNeXtEncoder with intermediate outputs."""
    encoder = PointNeXtEncoder(
        channels=[6, 32, 64, 128],
        depths=[2, 2, 2],
        ratios=[0.5, 0.5, 0.5, 0.5],
        radiuses=[0.1, 0.2, 0.4, 0.8],
        num_neighbors=[16, 16, 16, 16],
    )

    _, _, _, intermediates = encoder(
        data["features"],
        data["pos"],
        data["batch"],
        return_intermediates=True,
    )

    assert len(intermediates) == 3  # Number of blocks
    for intermediate in intermediates:
        assert hasattr(intermediate, "x")
        assert hasattr(intermediate, "pos")
        assert hasattr(intermediate, "batch")
        assert intermediate.x.shape[0] == intermediate.pos.shape[0] == intermediate.batch.shape[0]


def test_pointnext_encoder_decoder_basic(data: Dict[str, Tensor]) -> None:
    """Test basic PointNeXtDecoder functionality."""
    encoder = PointNeXtEncoder(
        channels=[6, 32, 64, 128],
        depths=[2, 2, 2],
        ratios=[0.5, 0.5, 0.5, 0.5],
        radiuses=[0.1, 0.2, 0.4, 0.8],
        num_neighbors=[16, 16, 16, 16],
    )

    decoder = PointNeXtDecoder(
        channels=[128, 128, 64, 32],
        skip_channels=[64, 32, 6],
        depths=[2, 2, 2],
    )

    x, pos, batch, intermediates = encoder(data["features"], data["pos"], data["batch"], return_intermediates=True)
    x, pos, batch = decoder(x, pos, batch, intermediates)

    assert x.shape[0] == pos.shape[0] == batch.shape[0]
    assert x.shape[1] == 32  # Final channel
    assert pos.shape[1] == 3


def test_pointnext_classification_forward(model_clf: PointNeXtClassification, data: Dict[str, Tensor]) -> None:
    """Test PointNeXtClassification forward pass."""
    logits = model_clf(data["features"], data["pos"], data["batch"])
    assert logits.shape == (data["batch"].max() + 1, model_clf.num_classes)
    assert logits.dtype == data["features"].dtype


def test_pointnext_segmentation_forward(model_seg: PointNeXtSegmentation, data: Dict[str, Tensor]) -> None:
    """Test PointNeXtSegmentation forward pass."""
    logits = model_seg(data["features"], data["pos"], data["batch"])
    assert logits.shape == (data["pos"].shape[0], model_seg.num_classes)
    assert logits.dtype == data["features"].dtype
