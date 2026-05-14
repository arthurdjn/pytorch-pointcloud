from typing import Dict

import pytest
import torch
from torch import Tensor

from torch_pointcloud.models.spunet import SparseUNetSegmentation
from torch_pointcloud.utils.imports import _CUDA_AVAILABLE, _SPCONV_AVAILABLE

# See: https://docs.pytest.org/en/stable/how-to/skipping.html#summary
pytestmark = [
    pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA is not available"),
    pytest.mark.skipif(not _SPCONV_AVAILABLE, reason="spconv is not installed"),
]


@pytest.fixture
def data() -> Dict[str, Tensor]:
    torch.manual_seed(42)
    lengths = torch.tensor([256, 512])
    pos_grid = torch.randint(0, 64, (int(lengths.sum()), 3))
    x = torch.randn(int(lengths.sum()), 6)
    batch = torch.repeat_interleave(torch.arange(len(lengths)), lengths)
    return dict(
        x=x.cuda(),
        pos_grid=pos_grid.cuda(),
        batch=batch.cuda(),
    )


@pytest.fixture
def model_seg() -> SparseUNetSegmentation:
    return SparseUNetSegmentation(
        in_channels=6,
        num_classes=10,
        base_channels=16,
        channels=(16, 32, 64, 128, 64, 32, 16, 16),
        layers=(1, 1, 1, 1, 1, 1, 1, 1),
        stem_kernel_size=5,
        kernel_size=3,
        spatial_padding=64,
        act="relu",
        act_kwargs=None,
        norm="batch_norm",
        norm_kwargs=None,
    ).cuda()


def test_spunet_segmentation_forward(model_seg: SparseUNetSegmentation, data: Dict[str, Tensor]) -> None:
    logits = model_seg(data["x"], data["pos_grid"], data["batch"])
    assert logits.shape == (data["pos_grid"].shape[0], model_seg.num_classes)


def test_spunet_segmentation_reset_classifier(model_seg: SparseUNetSegmentation, data: Dict[str, Tensor]) -> None:
    model_seg.reset_classifier(num_classes=42)
    model_seg.cuda()
    logits = model_seg(data["x"], data["pos_grid"], data["batch"])
    assert logits.shape == (data["pos_grid"].shape[0], 42)
