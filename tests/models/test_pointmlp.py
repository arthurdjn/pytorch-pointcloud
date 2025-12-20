from typing import Dict

import pytest
import torch
from torch import Tensor

from torch_pointcloud.models.pointmlp import (
    PointMLPClassification,
    PointMLPDecoder,
    PointMLPEncoder,
    PointMLPSegmentation,
)
from torch_pointcloud.utils.imports import _TORCH_CLUSTER_AVAILABLE, _TORCH_SCATTER_AVAILABLE


@pytest.fixture
def data() -> Dict[str, Tensor]:
    torch.manual_seed(42)
    lengths = torch.tensor([256, 512])
    x = torch.randn(int(lengths.sum()), 6)
    pos = torch.randn(int(lengths.sum()), 3)
    batch = torch.repeat_interleave(torch.arange(len(lengths)), lengths)

    return dict(
        x=x,
        pos=pos,
        batch=batch,
    )


@pytest.fixture
def model_clf() -> PointMLPClassification:
    return PointMLPClassification(
        in_channels=6,
        num_classes=10,
        channels=[32, 64, 128, 256],
        num_neighbors=[16, 16, 16],
        ratios=[0.5, 0.5, 0.5],
        num_pre_blocks=2,
        num_pos_blocks=2,
    )


@pytest.fixture
def model_seg() -> PointMLPSegmentation:
    return PointMLPSegmentation(
        in_channels=6,
        num_classes=10,
        encoder_channels=[32, 64, 128, 256],
        num_neighbors=[16, 16, 16],
        ratios=[0.5, 0.5, 0.5],
        decoder_channels=[256, 128, 64],
        decoder_blocks=[2, 2, 2],  # type: ignore # depths for decoder blocks (type annotation in source is incorrect)
        num_pre_blocks=2,
        num_pos_blocks=2,
    )


@pytest.mark.skipif(
    not _TORCH_CLUSTER_AVAILABLE and not _TORCH_SCATTER_AVAILABLE,
    reason="torch-cluster or torch-scatter is not installed",
)
def test_pointmlp_encoder_with_intermediates(data: Dict[str, Tensor]) -> None:
    """Test PointMLPEncoder with intermediate outputs."""
    encoder = PointMLPEncoder(
        channels=[6, 32, 64, 128],
        num_neighbors=[16, 16, 16],
        ratios=[0.5, 0.5, 0.5],
        num_pre_blocks=2,
        num_pos_blocks=2,
    )

    _, _, _, intermediates = encoder(
        data["x"],
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


@pytest.mark.skipif(
    not _TORCH_CLUSTER_AVAILABLE and not _TORCH_SCATTER_AVAILABLE,
    reason="torch-cluster or torch-scatter is not installed",
)
def test_pointmlp_encoder_decoder_basic(data: Dict[str, Tensor]) -> None:
    """Test basic PointMLPDecoder functionality."""
    encoder = PointMLPEncoder(
        channels=[6, 32, 64, 128],
        num_neighbors=[16, 16, 16],
        ratios=[0.5, 0.5, 0.5],
        num_pre_blocks=2,
        num_pos_blocks=2,
    )

    decoder = PointMLPDecoder(
        channels=[128, 64, 32],
        skip_channels=[64, 32, 6],
        depths=[2, 2],
    )

    x, pos, batch, intermediates = encoder(data["x"], data["pos"], data["batch"], return_intermediates=True)
    x, pos, batch = decoder(x, pos, batch, intermediates)

    assert x.shape[0] == pos.shape[0] == batch.shape[0]
    assert x.shape[1] == 32  # Final channel
    assert pos.shape[1] == 3


@pytest.mark.skipif(
    not _TORCH_CLUSTER_AVAILABLE and not _TORCH_SCATTER_AVAILABLE,
    reason="torch-cluster or torch-scatter is not installed",
)
def test_pointmlp_classification_forward(model_clf: PointMLPClassification, data: Dict[str, Tensor]) -> None:
    """Test PointMLPClassification forward pass."""
    logits = model_clf(data["x"], data["pos"], data["batch"])
    assert logits.shape == (data["batch"].max() + 1, model_clf.num_classes)
    assert logits.dtype == data["x"].dtype


@pytest.mark.skipif(
    not _TORCH_CLUSTER_AVAILABLE and not _TORCH_SCATTER_AVAILABLE,
    reason="torch-cluster or torch-scatter is not installed",
)
def test_pointmlp_classification_reset_classifier(
    model_clf: PointMLPClassification,
    data: Dict[str, Tensor],
) -> None:
    """Test PointMLPClassification reset_classifier."""
    new_num_classes = 20
    model_clf.reset_classifier(new_num_classes)

    assert model_clf.num_classes == new_num_classes
    assert model_clf.head.out_features == new_num_classes
    logits = model_clf(data["x"], data["pos"], data["batch"])
    assert logits.shape == (data["batch"].max() + 1, new_num_classes)


@pytest.mark.skipif(
    not _TORCH_CLUSTER_AVAILABLE and not _TORCH_SCATTER_AVAILABLE,
    reason="torch-cluster or torch-scatter is not installed",
)
def test_pointmlp_classification_forward_features(
    model_clf: PointMLPClassification,
    data: Dict[str, Tensor],
) -> None:
    """Test PointMLPClassification forward_features."""
    x, pos, batch = model_clf.forward_features(data["x"], data["pos"], data["batch"])
    assert x.dim() == 2
    assert pos.dim() == 2
    assert batch.dim() == 1

    # Test forward features with intermediates
    x, pos, batch, intermediates = model_clf.forward_features(
        data["x"],
        data["pos"],
        data["batch"],
        return_intermediates=True,
    )
    assert len(intermediates) == len(model_clf.encoder.blocks)
    for intermediate in intermediates:
        assert hasattr(intermediate, "x")
        assert hasattr(intermediate, "pos")
        assert hasattr(intermediate, "batch")


@pytest.mark.skipif(
    not _TORCH_CLUSTER_AVAILABLE and not _TORCH_SCATTER_AVAILABLE,
    reason="torch-cluster or torch-scatter is not installed",
)
def test_pointmlp_classification_forward_features_and_head(
    model_clf: PointMLPClassification,
    data: Dict[str, Tensor],
) -> None:
    """Test PointMLPClassification forward_features and forward_head."""
    x, _, batch = model_clf.forward_features(data["x"], data["pos"], data["batch"])
    logits = model_clf.forward_head(x, batch)
    assert logits.shape == (data["batch"].max() + 1, model_clf.num_classes)


@pytest.mark.skipif(
    not _TORCH_CLUSTER_AVAILABLE and not _TORCH_SCATTER_AVAILABLE,
    reason="torch-cluster or torch-scatter is not installed",
)
def test_pointmlp_segmentation_forward(model_seg: PointMLPSegmentation, data: Dict[str, Tensor]) -> None:
    """Test PointMLPSegmentation forward pass."""
    logits = model_seg(data["x"], data["pos"], data["batch"])
    assert logits.shape == (data["pos"].shape[0], model_seg.num_classes)
    assert logits.dtype == data["x"].dtype


@pytest.mark.skipif(
    not _TORCH_CLUSTER_AVAILABLE and not _TORCH_SCATTER_AVAILABLE,
    reason="torch-cluster or torch-scatter is not installed",
)
def test_pointmlp_segmentation_reset_classifier(
    model_seg: PointMLPSegmentation,
    data: Dict[str, Tensor],
) -> None:
    """Test PointMLPSegmentation reset_classifier."""
    new_num_classes = 20
    model_seg.reset_classifier(new_num_classes)

    assert model_seg.num_classes == new_num_classes
    assert model_seg.head.out_features == new_num_classes
    logits = model_seg(data["x"], data["pos"], data["batch"])
    assert logits.shape == (data["pos"].shape[0], new_num_classes)


@pytest.mark.skipif(
    not _TORCH_CLUSTER_AVAILABLE and not _TORCH_SCATTER_AVAILABLE,
    reason="torch-cluster or torch-scatter is not installed",
)
def test_pointmlp_segmentation_forward_features(
    model_seg: PointMLPSegmentation,
    data: Dict[str, Tensor],
) -> None:
    """Test PointMLPSegmentation forward_features."""
    x, pos, batch = model_seg.forward_features(data["x"], data["pos"], data["batch"])
    assert x.shape[0] == pos.shape[0] == batch.shape[0]
    assert x.dim() == 2
    assert pos.dim() == 2
    assert batch.dim() == 1


@pytest.mark.skipif(
    not _TORCH_CLUSTER_AVAILABLE and not _TORCH_SCATTER_AVAILABLE,
    reason="torch-cluster or torch-scatter is not installed",
)
def test_pointmlp_segmentation_forward_features_and_head(
    model_seg: PointMLPSegmentation,
    data: Dict[str, Tensor],
) -> None:
    """Test PointMLPSegmentation forward_features, forward_decoder, and forward_head."""
    x, pos, batch, intermediates = model_seg.forward_features(
        data["x"],
        data["pos"],
        data["batch"],
        return_intermediates=True,
    )

    x, _, _ = model_seg.forward_decoder(x, pos, batch, intermediates)
    logits = model_seg.forward_head(x)
    assert logits.shape == (data["pos"].shape[0], model_seg.num_classes)
