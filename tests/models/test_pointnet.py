from typing import Dict

import pytest
import torch
from torch import Tensor

from torch_pointcloud.models.pointnet import PointNetClassification, PointNetSegmentation
from torch_pointcloud.utils.imports import _TORCH_SCATTER_AVAILABLE

# See: https://docs.pytest.org/en/stable/how-to/skipping.html#summary
pytestmark = pytest.mark.skipif(
    not _TORCH_SCATTER_AVAILABLE,
    reason="torch-scatter is not installed",
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
def model_clf() -> PointNetClassification:
    return PointNetClassification(
        num_classes=10,
        in_channels=6,
        spatial_dim=3,
        dropout=0.0,
        global_pool="max",
        mlp1_dims=(32,),
        mlp2_dims=(64, 128),
        act="relu",
        norm="batch_norm1d",
        use_features_transform=True,
        tnet_mlp1_dims=(32, 64, 128),
        tnet_mlp2_dims=(64, 32),
        tnet_act="relu",
        tnet_norm="batch_norm1d",
    )


@pytest.fixture
def model_seg() -> PointNetSegmentation:
    return PointNetSegmentation(
        num_classes=10,
        spatial_dim=3,
        in_channels=6,
        dropout=0.3,
        mlp1_dims=(32,),
        mlp2_dims=(64, 128),
        act="relu",
        norm="batch_norm1d",
        global_pool="max",
        use_features_transform=True,
        tnet_mlp1_dims=(32, 64, 128),
        tnet_mlp2_dims=(64, 32),
        tnet_act="relu",
        tnet_norm="batch_norm1d",
        seg_head_dims=(64, 32),
    )


def test_pointnet_classification_forward(model_clf: PointNetClassification, data: Dict[str, Tensor]) -> None:
    logits = model_clf(data["x"], data["pos"], data["batch"])
    assert logits.shape == (int(data["batch"].max()) + 1, model_clf.num_classes)
    assert logits.dtype == data["x"].dtype


def test_pointnet_classification_reset_classifier(model_clf: PointNetClassification, data: Dict[str, Tensor]) -> None:
    model_clf.reset_classifier(num_classes=42)
    logits = model_clf(data["x"], data["pos"], data["batch"])
    assert logits.shape == (int(data["batch"].max()) + 1, 42)


def test_pointnet_segmentation_forward(model_seg: PointNetSegmentation, data: Dict[str, Tensor]) -> None:
    logits = model_seg(data["x"], data["pos"], data["batch"])
    assert logits.shape == (data["pos"].shape[0], model_seg.num_classes)
    assert logits.dtype == data["x"].dtype


def test_pointnet_classification_forward_features_and_head(
    model_clf: PointNetClassification, data: Dict[str, Tensor]
) -> None:
    x = model_clf.forward_features(data["x"], data["pos"], data["batch"])
    assert x.dim() == 2
    logits = model_clf.forward_head(x, data["batch"])
    assert logits.shape == (int(data["batch"].max()) + 1, model_clf.num_classes)


def test_pointnet_segmentation_forward_features_and_head(
    model_seg: PointNetSegmentation, data: Dict[str, Tensor]
) -> None:
    x, point_features = model_seg.forward_features(data["x"], data["pos"], data["batch"])
    assert x.shape[0] == point_features.shape[0] == data["pos"].shape[0]
    logits = model_seg.forward_head(x, point_features, data["batch"])
    assert logits.shape == (data["pos"].shape[0], model_seg.num_classes)
