from typing import Dict

import pytest
import torch
from torch import Tensor

from torch_pointcloud.models.pvcnn import PVCNNClassification, PVCNNSegmentation


@pytest.fixture
def data() -> Dict[str, Tensor]:
    torch.manual_seed(42)
    lengths = torch.tensor([256, 512])
    pos = torch.randn(int(lengths.sum()), 3)
    x = torch.randn(int(lengths.sum()), 6)
    batch = torch.repeat_interleave(torch.arange(len(lengths)), lengths)
    return dict(x=x, pos=pos, batch=batch)


@pytest.fixture
def model_clf() -> PVCNNClassification:
    return PVCNNClassification(
        in_channels=6,
        num_classes=10,
        channels=[32, 64, 128],
        global_channels=[64, 32],
        depths=[1, 1, 1],
        kernel_sizes=[3, 3, 3],
        resolutions=[8, 8, 8],
        with_se=False,
        normalize=True,
        dropout=0.0,
        global_pool="max",
        act="relu",
        act_kwargs=None,
        act_first=False,
        norm="batch_norm",
        norm_kwargs=None,
    )


@pytest.fixture
def model_seg() -> PVCNNSegmentation:
    return PVCNNSegmentation(
        in_channels=6,
        num_classes=10,
        channels=[32, 64, 128],
        global_channels=[64, 32],
        depths=[1, 1, 1],
        kernel_sizes=[3, 3, 3],
        resolutions=[8, 8, 8],
        spatial_dim=3,
        with_se=False,
        normalize=True,
        dropout=0.0,
        global_pool="max",
        act="relu",
        act_kwargs=None,
        act_first=False,
        norm="batch_norm",
        norm_kwargs=None,
    )


def test_pvcnn_classification_forward(model_clf: PVCNNClassification, data: Dict[str, Tensor]) -> None:
    logits = model_clf(data["x"], data["pos"], data["batch"])
    assert logits.shape == (int(data["batch"].max()) + 1, model_clf.num_classes)
    assert logits.dtype == data["x"].dtype


def test_pvcnn_segmentation_forward(model_seg: PVCNNSegmentation, data: Dict[str, Tensor]) -> None:
    logits = model_seg(data["x"], data["pos"], data["batch"])
    assert logits.shape == (data["pos"].shape[0], model_seg.num_classes)
    assert logits.dtype == data["x"].dtype
