from typing import Dict

import pytest
import torch
from torch import Tensor

from torch_pointcloud.models.spvcnn import SPVCNNClassification, SPVCNNSegmentation
from torch_pointcloud.utils.imports import _CUDA_AVAILABLE, _TORCHSPARSE_AVAILABLE

# See: https://docs.pytest.org/en/stable/how-to/skipping.html#summary
pytestmark = [
    pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA is not available"),
    pytest.mark.skipif(not _TORCHSPARSE_AVAILABLE, reason="torchsparse is not installed"),
]


@pytest.fixture
def data() -> Dict[str, Tensor]:
    torch.manual_seed(42)
    lengths = torch.tensor([512, 768])
    pos = torch.randn(int(lengths.sum()), 3) * 20.0
    x = torch.randn(int(lengths.sum()), 6)
    batch = torch.repeat_interleave(torch.arange(len(lengths)), lengths)
    return dict(
        x=x.cuda(),
        pos=pos.cuda(),
        batch=batch.cuda(),
    )


@pytest.fixture
def model_seg() -> SPVCNNSegmentation:
    return SPVCNNSegmentation(
        in_channels=6,
        num_classes=10,
        spatial_dim=3,
        stem_channels=16,
        encoder_channels=(16, 32, 64),
        encoder_depths=(1, 1, 1),
        encoder_fusion_stages=(False, False, True),
        decoder_channels=(32, 16, 16),
        decoder_depths=(1, 1, 1),
        decoder_fusion_stages=(False, True, False),
        kernel_size=3,
        stride=1,
        dilation=1,
        drop_path=0.3,
        act="relu",
        act_kwargs=None,
        norm="batch_norm",
        norm_kwargs=None,
    ).cuda()


def test_spvcnn_segmentation_forward(model_seg: SPVCNNSegmentation, data: Dict[str, Tensor]) -> None:
    logits = model_seg(data["x"], data["pos"], data["batch"])
    assert logits.shape == (data["pos"].shape[0], model_seg.num_classes)


@pytest.fixture
def model_clf() -> SPVCNNClassification:
    return SPVCNNClassification(
        in_channels=6,
        num_classes=10,
        spatial_dim=3,
        stem_channels=16,
        encoder_channels=(16, 32, 64),
        encoder_depths=(1, 1, 1),
        encoder_fusion_stages=(False, False, True),
        kernel_size=3,
        stride=1,
        dilation=1,
        drop_path=0.3,
        global_pool="max",
        dropout=0.0,
        act="relu",
        act_kwargs=None,
        norm="batch_norm",
        norm_kwargs=None,
    ).cuda()


def test_spvcnn_classification_forward(model_clf: SPVCNNClassification, data: Dict[str, Tensor]) -> None:
    logits = model_clf(data["x"], data["pos"], data["batch"])
    assert logits.shape == (int(data["batch"].max()) + 1, model_clf.num_classes)


def test_spvcnn_classification_reset_classifier(model_clf: SPVCNNClassification, data: Dict[str, Tensor]) -> None:
    model_clf.reset_classifier(num_classes=42)
    model_clf.cuda()
    logits = model_clf(data["x"], data["pos"], data["batch"])
    assert logits.shape == (int(data["batch"].max()) + 1, 42)
