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
spconv = pytest.importorskip("spconv.pytorch")


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


def test_spunet_reset_classifier_initializes_head(model_seg: SparseUNetSegmentation) -> None:
    model_seg.reset_classifier(num_classes=42)
    head = model_seg.head
    assert isinstance(head, spconv.SubMConv3d)
    assert head.bias is not None
    assert torch.all(head.bias == 0)
    assert head.weight.std().item() < 0.05


def test_spunet_segmentation_forward_features_decoder_head(
    model_seg: SparseUNetSegmentation, data: Dict[str, Tensor]
) -> None:
    sparse_x, skips = model_seg.forward_features(data["x"], data["pos_grid"], data["batch"])
    assert len(skips) > 0
    sparse_x = model_seg.forward_decoder(sparse_x, skips)
    logits = model_seg.forward_head(sparse_x)
    assert logits.shape == (data["pos_grid"].shape[0], model_seg.num_classes)


def test_spunet_forward_head_pre_logits(model_seg: SparseUNetSegmentation, data: Dict[str, Tensor]) -> None:
    sparse_x, skips = model_seg.forward_features(data["x"], data["pos_grid"], data["batch"])
    sparse_x = model_seg.forward_decoder(sparse_x, skips)
    feats = model_seg.forward_head(sparse_x, pre_logits=True)
    assert torch.equal(feats, sparse_x.features)
    assert feats.shape[1] == model_seg.channels[-1]
