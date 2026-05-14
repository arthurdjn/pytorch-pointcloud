from typing import Dict

import pytest
import torch
from torch import Tensor

from torch_pointcloud.models.pvcnn2 import PVCNN2Classification, PVCNN2Segmentation
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
def model_clf() -> PVCNN2Classification:
    return PVCNN2Classification(
        in_channels=6,
        num_classes=10,
        ratios=[0.5, 0.5],
        radii=[0.2, 0.4],
        num_neighbors=[16, 16],
        sa_channels=[[32, 64], [64, 64]],
        encoder_channels=[6, 32, 64],
        encoder_depths=[1, 1],
        encoder_resolutions=[8, 8],
        encoder_kernel_sizes=[3, 3],
        with_se=False,
        normalize=True,
        act="relu",
        act_kwargs=None,
        act_first=False,
        norm="batch_norm",
        norm_kwargs=None,
        dropout=0.0,
        global_pool="max",
    )


@pytest.fixture
def model_seg() -> PVCNN2Segmentation:
    return PVCNN2Segmentation(
        in_channels=6,
        num_classes=10,
        ratios=[0.5, 0.5],
        radii=[0.2, 0.4],
        num_neighbors=[16, 16],
        sa_channels=[[32, 64], [64, 64]],
        encoder_channels=[6, 32, 64],
        encoder_depths=[1, 1],
        encoder_resolutions=[8, 8],
        encoder_kernel_sizes=[3, 3],
        fp_channels=[[64, 32], [32, 32]],
        decoder_channels=[32, 32],
        decoder_depths=[1, 1],
        decoder_resolutions=[8, 8],
        decoder_kernel_sizes=[3, 3],
        with_se=False,
        normalize=True,
        act="relu",
        act_kwargs=None,
        act_first=False,
        norm="batch_norm",
        norm_kwargs=None,
        dropout=0.0,
    )


def test_pvcnn2_classification_forward(model_clf: PVCNN2Classification, data: Dict[str, Tensor]) -> None:
    logits = model_clf(data["x"], data["pos"], data["batch"])
    assert logits.shape == (int(data["batch"].max()) + 1, model_clf.num_classes)
    assert logits.dtype == data["x"].dtype


def test_pvcnn2_classification_reset_head(model_clf: PVCNN2Classification, data: Dict[str, Tensor]) -> None:
    model_clf.reset_head(num_classes=42)
    logits = model_clf(data["x"], data["pos"], data["batch"])
    assert logits.shape == (int(data["batch"].max()) + 1, 42)


def test_pvcnn2_segmentation_forward(model_seg: PVCNN2Segmentation, data: Dict[str, Tensor]) -> None:
    logits = model_seg(data["x"], data["pos"], data["batch"])
    assert logits.shape == (data["pos"].shape[0], model_seg.num_classes)
    assert logits.dtype == data["x"].dtype
