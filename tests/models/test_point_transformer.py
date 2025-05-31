from typing import Dict

import pytest
import torch
from torch import Tensor

from torch_pointcloud.models.point_transformer import (
    PointTransformerClassification,
    PointTransformerConv,
    PointTransformerSegmentation,
)
from torch_pointcloud.utils.imports import _TORCH_CLUSTER_AVAILABLE, _TORCH_SCATTER_AVAILABLE


@pytest.fixture
def data() -> Dict[str, Tensor]:
    torch.manual_seed(42)
    lengths = torch.tensor([256, 512])
    coords = torch.randn(int(lengths.sum()), 3)
    features = torch.randn(int(lengths.sum()), 3)
    batch = torch.repeat_interleave(torch.arange(len(lengths)), lengths)

    # Dummy edge_index connecting each point to 16 nearest indices
    row = torch.arange(len(coords)).repeat_interleave(16)
    cumsum = torch.cat([torch.tensor([0]), torch.cumsum(lengths, dim=0)])
    col = torch.cat([torch.arange(int(lengths[i])).repeat(16) + cumsum[i] for i in range(len(lengths))])
    edge_index = torch.stack([row, col])

    return dict(
        features=features,
        coords=coords,
        batch=batch,
        edge_index=edge_index,
    )


@pytest.mark.skipif(
    not _TORCH_CLUSTER_AVAILABLE and not _TORCH_SCATTER_AVAILABLE,
    reason="torch-cluster or torch-scatter is not installed",
)
def test_point_transformer_conv(data: Dict[str, Tensor]) -> None:
    conv = PointTransformerConv(
        spatial_dim=3,
        in_channels=3,
        out_channels=32,
    )

    output = conv(data["features"], data["coords"], data["edge_index"])
    assert output.shape == (len(data["coords"]), 32)


@pytest.fixture
def model_clf() -> PointTransformerClassification:
    return PointTransformerClassification(
        in_channels=3,
        num_classes=10,
        encoder_depths=[2, 2],
        encoder_channels=[32, 64],
        encoder_num_groups=[1, 1],
        encoder_num_neighbors=[16, 16],
        ratios=[0.25],
    )


@pytest.fixture
def model_seg() -> PointTransformerSegmentation:
    return PointTransformerSegmentation(
        in_channels=3,
        num_classes=10,
        encoder_depths=[2, 2],
        encoder_channels=[32, 64],
        encoder_num_groups=[1, 1],
        encoder_num_neighbors=[16, 16],
        decoder_depths=[2],
        decoder_channels=[32],
        decoder_num_groups=[1],
        decoder_num_neighbors=[16],
        ratios=[0.25],
    )


@pytest.mark.skipif(
    not _TORCH_CLUSTER_AVAILABLE and not _TORCH_SCATTER_AVAILABLE,
    reason="torch-cluster or torch-scatter is not installed",
)
def test_point_transformer_clf_forward(model_clf: PointTransformerClassification, data: Dict[str, Tensor]) -> None:
    logits = model_clf(data["features"], data["coords"], data["batch"])
    assert logits.shape == (data["batch"].max() + 1, model_clf.num_classes)


@pytest.mark.skipif(
    not _TORCH_CLUSTER_AVAILABLE and not _TORCH_SCATTER_AVAILABLE,
    reason="torch-cluster or torch-scatter is not installed",
)
def test_point_transformer_clf_reset_classifier(
    model_clf: PointTransformerClassification,
    data: Dict[str, Tensor],
) -> None:
    new_num_classes = 20
    model_clf.reset_classifier(new_num_classes)

    assert model_clf.num_classes == new_num_classes
    assert model_clf.head.out_features == new_num_classes
    logits = model_clf(data["features"], data["coords"], data["batch"])
    assert logits.shape == (data["batch"].max() + 1, new_num_classes)


@pytest.mark.skipif(
    not _TORCH_CLUSTER_AVAILABLE and not _TORCH_SCATTER_AVAILABLE,
    reason="torch-cluster or torch-scatter is not installed",
)
def test_point_transformer_clf_forward_encoder(
    model_clf: PointTransformerClassification,
    data: Dict[str, Tensor],
) -> None:
    x, pos, batch = model_clf.forward_encoder(data["features"], data["coords"], data["batch"])
    assert x.dim() == 2
    assert pos.dim() == 2
    assert batch.dim() == 1

    # Test forward features with intermediates
    x, pos, batch, intermediates = model_clf.forward_encoder(
        data["features"],
        data["coords"],
        data["batch"],
        return_intermediates=True,
    )
    assert len(intermediates) == len(model_clf.encoder.blocks)
    for intermediate in intermediates:
        assert "features" in intermediate
        assert "pos" in intermediate
        assert "batch" in intermediate


@pytest.mark.skipif(
    not _TORCH_CLUSTER_AVAILABLE and not _TORCH_SCATTER_AVAILABLE,
    reason="torch-cluster or torch-scatter is not installed",
)
def test_point_transformer_clf_forward_encoder_and_head(
    model_clf: PointTransformerClassification,
    data: Dict[str, Tensor],
) -> None:
    x, _, batch = model_clf.forward_encoder(data["features"], data["coords"], data["batch"])
    logits = model_clf.forward_head(x, batch)
    assert logits.shape == (data["batch"].max() + 1, model_clf.num_classes)


@pytest.mark.skipif(
    not _TORCH_CLUSTER_AVAILABLE and not _TORCH_SCATTER_AVAILABLE,
    reason="torch-cluster or torch-scatter is not installed",
)
def test_point_transformer_seg_forward(model_seg: PointTransformerSegmentation, data: Dict[str, Tensor]) -> None:
    logits = model_seg(data["features"], data["coords"], data["batch"])
    assert logits.shape == (data["coords"].shape[0], model_seg.num_classes)


@pytest.mark.skipif(
    not _TORCH_CLUSTER_AVAILABLE and not _TORCH_SCATTER_AVAILABLE,
    reason="torch-cluster or torch-scatter is not installed",
)
def test_point_transformer_seg_reset_classifier(
    model_seg: PointTransformerSegmentation,
    data: Dict[str, Tensor],
) -> None:
    new_num_classes = 20
    model_seg.reset_classifier(new_num_classes)

    assert model_seg.num_classes == new_num_classes
    assert model_seg.head.out_features == new_num_classes
    logits = model_seg(data["features"], data["coords"], data["batch"])
    assert logits.shape == (data["coords"].shape[0], new_num_classes)


@pytest.mark.skipif(
    not _TORCH_CLUSTER_AVAILABLE and not _TORCH_SCATTER_AVAILABLE,
    reason="torch-cluster or torch-scatter is not installed",
)
def test_point_transformer_seg_forward_encoder(
    model_seg: PointTransformerSegmentation,
    data: Dict[str, Tensor],
) -> None:
    x, pos, batch = model_seg.forward_encoder(data["features"], data["coords"], data["batch"])
    assert x.shape[0] == pos.shape[0] == batch.shape[0]
    assert x.dim() == 2
    assert pos.dim() == 2
    assert batch.dim() == 1


@pytest.mark.skipif(
    not _TORCH_CLUSTER_AVAILABLE and not _TORCH_SCATTER_AVAILABLE,
    reason="torch-cluster or torch-scatter is not installed",
)
def test_point_transformer_seg_forward_features_and_head(
    model_seg: PointTransformerSegmentation,
    data: Dict[str, Tensor],
) -> None:
    x, pos, batch, intermediates = model_seg.forward_encoder(
        data["features"],
        data["coords"],
        data["batch"],
        return_intermediates=True,
    )

    x, _, _ = model_seg.forward_decoder(x, pos, batch, intermediates)
    logits = model_seg.forward_head(x)
    assert logits.shape == (data["coords"].shape[0], model_seg.num_classes)
