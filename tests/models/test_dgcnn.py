from typing import Dict

import pytest
import torch
from torch import Tensor

from torch_pointcloud.models.dgcnn import DGCNNClassification, DGCNNPartSegmentation, DGCNNSegmentation
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
    x = torch.randn(int(lengths.sum()), 3)
    batch = torch.repeat_interleave(torch.arange(len(lengths)), lengths)
    return dict(x=x, pos=pos, batch=batch)


@pytest.fixture
def model_clf() -> DGCNNClassification:
    return DGCNNClassification(
        in_channels=3,
        num_classes=10,
        spatial_dim=3,
        stnet_local_channels=None,
        stnet_global_channels=None,
        head_channels=[32],
        channels=[32, 64],
        proj_channels=64,
        num_neighbors=8,
        act="relu",
        act_kwargs=None,
        act_first=False,
        norm="batch_norm",
        norm_kwargs=None,
        bias=True,
        dropout=0.0,
        global_pool="max",
    )


@pytest.fixture
def model_seg() -> DGCNNSegmentation:
    return DGCNNSegmentation(
        in_channels=3,
        num_classes=10,
        spatial_dim=3,
        stnet_local_channels=None,
        stnet_global_channels=None,
        proj_channels=64,
        channels=[32, 64],
        head_channels=[32],
        num_neighbors=8,
        act="relu",
        act_kwargs=None,
        act_first=False,
        norm="batch_norm",
        norm_kwargs=None,
        bias=True,
        dropout=0.0,
    )


def test_dgcnn_classification_forward(model_clf: DGCNNClassification, data: Dict[str, Tensor]) -> None:
    logits = model_clf(data["x"], data["pos"], data["batch"])
    assert logits.shape == (int(data["batch"].max()) + 1, model_clf.num_classes)
    assert logits.dtype == data["x"].dtype


def test_dgcnn_classification_reset_classifier(model_clf: DGCNNClassification, data: Dict[str, Tensor]) -> None:
    model_clf.reset_classifier(num_classes=42)
    logits = model_clf(data["x"], data["pos"], data["batch"])
    assert logits.shape == (int(data["batch"].max()) + 1, 42)


def test_dgcnn_segmentation_forward(model_seg: DGCNNSegmentation, data: Dict[str, Tensor]) -> None:
    logits = model_seg(data["x"], data["pos"], data["batch"])
    assert logits.shape == (data["pos"].shape[0], model_seg.num_classes)
    assert logits.dtype == data["x"].dtype


def test_dgcnn_segmentation_reset_classifier(model_seg: DGCNNSegmentation, data: Dict[str, Tensor]) -> None:
    model_seg.reset_classifier(num_classes=42)
    logits = model_seg(data["x"], data["pos"], data["batch"])
    assert logits.shape == (data["pos"].shape[0], 42)


def test_dgcnn_classification_forward_features_and_head(
    model_clf: DGCNNClassification, data: Dict[str, Tensor]
) -> None:
    x, _, batch = model_clf.forward_features(data["x"], data["pos"], data["batch"])
    assert x.shape[0] == batch.shape[0]
    logits = model_clf.forward_head(x, batch)
    assert logits.shape == (int(data["batch"].max()) + 1, model_clf.num_classes)


def test_dgcnn_segmentation_forward_features_and_head(model_seg: DGCNNSegmentation, data: Dict[str, Tensor]) -> None:
    x, _, batch = model_seg.forward_features(data["x"], data["pos"], data["batch"])
    assert x.shape[0] == batch.shape[0] == data["pos"].shape[0]
    logits = model_seg.forward_head(x, batch)
    assert logits.shape == (data["pos"].shape[0], model_seg.num_classes)


@pytest.fixture
def model_partseg() -> DGCNNPartSegmentation:
    return DGCNNPartSegmentation(
        in_channels=3,
        num_classes=10,
        num_categories=4,
        cat_embed_channels=16,
        spatial_dim=3,
        stnet_edge_channels=None,
        stnet_local_channels=None,
        stnet_global_channels=None,
        stnet_num_neighbors=20,
        proj_channels=64,
        channels=[32, 64],
        head_channels=[32],
        num_neighbors=8,
        act="relu",
        act_kwargs=None,
        act_first=False,
        norm="batch_norm",
        norm_kwargs=None,
        bias=True,
        dropout=0.0,
    )


@pytest.fixture
def partseg_category(data: Dict[str, Tensor]) -> Tensor:
    num_batches = int(data["batch"].max()) + 1
    return torch.nn.functional.one_hot(torch.arange(num_batches) % 4, num_classes=4).float()


def test_dgcnn_part_segmentation_forward(
    model_partseg: DGCNNPartSegmentation, data: Dict[str, Tensor], partseg_category: Tensor
) -> None:
    logits = model_partseg(data["x"], data["pos"], data["batch"], partseg_category)
    assert logits.shape == (data["pos"].shape[0], model_partseg.num_classes)
    assert logits.dtype == data["x"].dtype


def test_dgcnn_part_segmentation_reset_classifier(
    model_partseg: DGCNNPartSegmentation, data: Dict[str, Tensor], partseg_category: Tensor
) -> None:
    model_partseg.reset_classifier(num_classes=42)
    logits = model_partseg(data["x"], data["pos"], data["batch"], partseg_category)
    assert logits.shape == (data["pos"].shape[0], 42)


def test_dgcnn_part_segmentation_forward_features_and_head(
    model_partseg: DGCNNPartSegmentation, data: Dict[str, Tensor], partseg_category: Tensor
) -> None:
    x, _, batch = model_partseg.forward_features(data["x"], data["pos"], data["batch"], partseg_category)
    assert x.shape[0] == data["pos"].shape[0]
    logits = model_partseg.forward_head(x, batch)
    assert logits.shape == (data["pos"].shape[0], model_partseg.num_classes)
