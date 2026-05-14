from typing import Dict

import pytest
import torch
from torch import Tensor

from torch_pointcloud.models.point_transformer_v2 import (
    PointTransformerV2Classification,
    PointTransformerV2Segmentation,
)
from torch_pointcloud.utils.imports import _TORCH_CLUSTER_AVAILABLE, _TORCH_SCATTER_AVAILABLE

# See: https://docs.pytest.org/en/stable/how-to/skipping.html#summary
pytestmark = pytest.mark.skipif(
    not (_TORCH_CLUSTER_AVAILABLE and _TORCH_SCATTER_AVAILABLE),
    reason="torch-cluster or torch-scatter is not installed",
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
def model_clf() -> PointTransformerV2Classification:
    return PointTransformerV2Classification(
        in_channels=6,
        num_classes=10,
        grid_sizes=(0.1, 0.2),
        encoder_depths=(1, 1, 1),
        encoder_channels=(32, 64, 128),
        encoder_num_groups=(2, 4, 8),
        encoder_num_neighbors=(8, 8, 8),
        norm="batch_norm1d",
        act="relu",
        qkv_bias=True,
        attn_drop=0.0,
        pe_multiplier=False,
        pe_bias=True,
        drop_path=0.0,
        dropout=0.0,
        global_pool="max",
    )


@pytest.fixture
def model_seg() -> PointTransformerV2Segmentation:
    return PointTransformerV2Segmentation(
        in_channels=6,
        num_classes=10,
        grid_sizes=(0.1, 0.2),
        encoder_depths=(1, 1, 1),
        encoder_channels=(32, 64, 128),
        encoder_num_groups=(2, 4, 8),
        encoder_num_neighbors=(8, 8, 8),
        decoder_depths=(1, 1),
        decoder_channels=(64, 32),
        decoder_num_groups=(4, 2),
        decoder_num_neighbors=(8, 8),
        norm="batch_norm1d",
        act="relu",
        qkv_bias=True,
        attn_drop=0.0,
        pe_multiplier=False,
        pe_bias=True,
        drop_path=0.0,
        dropout=0.0,
    )


def test_pt_v2_classification_forward(model_clf: PointTransformerV2Classification, data: Dict[str, Tensor]) -> None:
    logits = model_clf(data["x"], data["pos"], data["batch"])
    assert logits.shape == (int(data["batch"].max()) + 1, model_clf.num_classes)
    assert logits.dtype == data["x"].dtype


def test_pt_v2_classification_reset_classifier(
    model_clf: PointTransformerV2Classification, data: Dict[str, Tensor]
) -> None:
    model_clf.reset_classifier(num_classes=42)
    logits = model_clf(data["x"], data["pos"], data["batch"])
    assert logits.shape == (int(data["batch"].max()) + 1, 42)


def test_pt_v2_segmentation_forward(model_seg: PointTransformerV2Segmentation, data: Dict[str, Tensor]) -> None:
    logits = model_seg(data["x"], data["pos"], data["batch"])
    assert logits.shape == (data["pos"].shape[0], model_seg.num_classes)
    assert logits.dtype == data["x"].dtype
