from typing import Dict

import pytest
import torch
from torch import Tensor

from torch_pointcloud.models.pointcnn import PointCNNClassification, PointCNNSegmentation
from torch_pointcloud.utils.imports import _TORCH_CLUSTER_AVAILABLE

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
def model_clf() -> PointCNNClassification:
    return PointCNNClassification(
        in_channels=6,
        num_classes=10,
        spatial_dim=3,
        channels=[32, 64, 128],
        kernel_sizes=[8, 8, 8],
        ratios=[0.0, 0.5, 0.5],
        hidden_channels=[16, 32, 64],
        dilations=[1, 1, 1],
        bias=True,
        act="relu",
        act_kwargs=None,
        act_first=False,
        norm="batch_norm",
        norm_kwargs=None,
        add_self_loops=False,
        dropout=0.0,
        head_channels=[64, 32],
        global_pool="max",
    )


@pytest.fixture
def model_seg() -> PointCNNSegmentation:
    return PointCNNSegmentation(
        in_channels=6,
        num_classes=10,
        spatial_dim=3,
        channels=[32, 64, 128],
        hidden_channels=[16, 32, 64],
        kernel_sizes=[8, 8, 8],
        dilations=[1, 1, 1],
        ratios=[0.0, 0.5, 0.5],
        act="relu",
        act_kwargs=None,
        act_first=False,
        norm="batch_norm",
        norm_kwargs=None,
        bias=True,
        add_self_loops=False,
        dropout=0.0,
        head_channels=[64, 32],
    )


def test_pointcnn_classification_forward(model_clf: PointCNNClassification, data: Dict[str, Tensor]) -> None:
    logits = model_clf(data["x"], data["pos"], data["batch"])
    assert logits.shape == (int(data["batch"].max()) + 1, model_clf.num_classes)
    assert logits.dtype == data["x"].dtype


def test_pointcnn_classification_reset_classifier(model_clf: PointCNNClassification, data: Dict[str, Tensor]) -> None:
    model_clf.reset_classifier(num_classes=42)
    logits = model_clf(data["x"], data["pos"], data["batch"])
    assert logits.shape == (int(data["batch"].max()) + 1, 42)


def test_pointcnn_segmentation_forward(model_seg: PointCNNSegmentation, data: Dict[str, Tensor]) -> None:
    logits = model_seg(data["x"], data["pos"], data["batch"])
    assert logits.shape == (data["pos"].shape[0], model_seg.num_classes)
    assert logits.dtype == data["x"].dtype


def test_pointcnn_segmentation_reset_classifier(model_seg: PointCNNSegmentation, data: Dict[str, Tensor]) -> None:
    model_seg.reset_classifier(num_classes=42)
    logits = model_seg(data["x"], data["pos"], data["batch"])
    assert logits.shape == (data["pos"].shape[0], 42)
