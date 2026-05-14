from typing import Dict

import pytest
import torch
from torch import Tensor

from torch_pointcloud.models.point_transformer_v3 import (
    PointTransformerV3Classification,
    PointTransformerV3Segmentation,
)
from torch_pointcloud.utils.imports import _CUDA_AVAILABLE, _SPCONV_AVAILABLE, _TORCH_SCATTER_AVAILABLE

# See: https://docs.pytest.org/en/stable/how-to/skipping.html#summary
pytestmark = [
    pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA is not available"),
    pytest.mark.skipif(not _SPCONV_AVAILABLE, reason="spconv is not installed"),
    pytest.mark.skipif(not _TORCH_SCATTER_AVAILABLE, reason="torch-scatter is not installed"),
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
def model_clf() -> PointTransformerV3Classification:
    return PointTransformerV3Classification(
        in_channels=6,
        num_classes=10,
        serialization_orders=("hilbert", "hilbert-trans"),
        shuffle_serialization_orders=True,
        strides=(2, 2),
        encoder_depths=(1, 1, 1),
        encoder_channels=(16, 32, 64),
        encoder_num_head=(2, 4, 8),
        encoder_patch_size=(16, 16, 16),
        norm="batch_norm",
        act="gelu",
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        attention="default",
        use_flash_attn=False,
        upcast_attention=False,
        upcast_softmax=False,
        rope_base=10.0,
        dropout=0.0,
        global_pool="max",
        pooling="serialized",
        stem_type="sparse_conv",
        act_kwargs=None,
        norm_kwargs=None,
    ).cuda()


@pytest.fixture
def model_seg() -> PointTransformerV3Segmentation:
    return PointTransformerV3Segmentation(
        in_channels=6,
        num_classes=10,
        serialization_orders=("hilbert", "hilbert-trans"),
        strides=(2, 2),
        encoder_depths=(1, 1, 1),
        encoder_channels=(16, 32, 64),
        encoder_num_head=(2, 4, 8),
        encoder_patch_size=(16, 16, 16),
        decoder_depths=(1, 1),
        decoder_channels=(32, 16),
        decoder_num_head=(4, 2),
        decoder_patch_size=(16, 16),
        norm="batch_norm",
        act="gelu",
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_serialization_orders=True,
        attention="default",
        rope_base=10.0,
        use_flash_attn=False,
        upcast_attention=False,
        upcast_softmax=False,
        dropout=0.0,
        pooling="serialized",
        stem_type="sparse_conv",
        act_kwargs=None,
        norm_kwargs=None,
    ).cuda()


def test_pt_v3_classification_forward(model_clf: PointTransformerV3Classification, data: Dict[str, Tensor]) -> None:
    logits = model_clf(data["x"], data["pos_grid"], data["batch"])
    assert logits.shape == (int(data["batch"].max()) + 1, model_clf.num_classes)


def test_pt_v3_classification_reset_classifier(
    model_clf: PointTransformerV3Classification, data: Dict[str, Tensor]
) -> None:
    model_clf.reset_classifier(num_classes=42)
    model_clf.cuda()
    logits = model_clf(data["x"], data["pos_grid"], data["batch"])
    assert logits.shape == (int(data["batch"].max()) + 1, 42)


def test_pt_v3_classification_forward_features_and_head(
    model_clf: PointTransformerV3Classification, data: Dict[str, Tensor]
) -> None:
    x, _, batch = model_clf.forward_features(data["x"], data["pos_grid"], data["batch"])
    assert x.shape[0] == batch.shape[0]
    logits = model_clf.forward_head(x, batch)
    assert logits.shape == (int(data["batch"].max()) + 1, model_clf.num_classes)


def test_pt_v3_segmentation_forward(model_seg: PointTransformerV3Segmentation, data: Dict[str, Tensor]) -> None:
    logits = model_seg(data["x"], data["pos_grid"], data["batch"])
    assert logits.shape == (data["pos_grid"].shape[0], model_seg.num_classes)


def test_pt_v3_segmentation_forward_features_decoder_head(
    model_seg: PointTransformerV3Segmentation, data: Dict[str, Tensor]
) -> None:
    x, _, _, intermediates = model_seg.forward_features(
        data["x"], data["pos_grid"], data["batch"], return_intermediates=True
    )
    assert len(intermediates) > 0
    x, _, _ = model_seg.forward_decoder(x, intermediates)
    logits = model_seg.forward_head(x)
    assert logits.shape == (data["pos_grid"].shape[0], model_seg.num_classes)
