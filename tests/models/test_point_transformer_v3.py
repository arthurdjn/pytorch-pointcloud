from typing import Dict

import pytest
import torch
from torch import Tensor

from torch_pointcloud.models.point_transformer_v3 import (
    PointTransformerV3Classification,
    PointTransformerV3Segmentation,
    serialize,
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
        encoder_num_heads=(2, 4, 8),
        encoder_patch_size=(16, 16, 16),
        norm="batch_norm",
        act="gelu",
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        attn_kind="default",
        use_flash_attn=False,
        upcast_attn=False,
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
        encoder_num_heads=(2, 4, 8),
        encoder_patch_size=(16, 16, 16),
        decoder_depths=(1, 1),
        decoder_channels=(32, 16),
        decoder_num_heads=(4, 2),
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
        attn_kind="default",
        rope_base=10.0,
        use_flash_attn=False,
        upcast_attn=False,
        upcast_softmax=False,
        dropout=0.0,
        pooling="serialized",
        stem_type="sparse_conv",
        act_kwargs=None,
        norm_kwargs=None,
    ).cuda()


@pytest.fixture
def model_seg_pdnorm() -> PointTransformerV3Segmentation:
    return PointTransformerV3Segmentation(
        in_channels=6,
        num_classes=10,
        strides=(2, 2),
        encoder_depths=(1, 1, 1),
        encoder_channels=(16, 32, 64),
        encoder_num_heads=(2, 4, 8),
        encoder_patch_size=(16, 16, 16),
        decoder_depths=(1, 1),
        decoder_channels=(32, 16),
        decoder_num_heads=(4, 2),
        decoder_patch_size=(16, 16),
        use_flash_attn=False,
        pdnorm_conditions=("ScanNet", "S3DIS"),
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


def test_pt_v3_segmentation_reset_classifier(
    model_seg: PointTransformerV3Segmentation, data: Dict[str, Tensor]
) -> None:
    model_seg.reset_classifier(num_classes=42)
    model_seg.cuda()
    logits = model_seg(data["x"], data["pos_grid"], data["batch"])
    assert logits.shape == (data["pos_grid"].shape[0], 42)


def test_pt_v3_condition_without_pdnorm_conditions_raises(
    model_seg: PointTransformerV3Segmentation, data: Dict[str, Tensor]
) -> None:
    with pytest.raises(ValueError, match="without conditional norms"):
        model_seg(data["x"], data["pos_grid"], data["batch"], condition="ScanNet")


def test_pt_v3_pdnorm_conditions_without_condition_raises(
    model_seg_pdnorm: PointTransformerV3Segmentation, data: Dict[str, Tensor]
) -> None:
    with pytest.raises(ValueError, match="pass `condition=`"):
        model_seg_pdnorm(data["x"], data["pos_grid"], data["batch"])


def test_pt_v3_pdnorm_forward_with_condition(
    model_seg_pdnorm: PointTransformerV3Segmentation, data: Dict[str, Tensor]
) -> None:
    logits = model_seg_pdnorm(data["x"], data["pos_grid"], data["batch"], condition="ScanNet")
    assert logits.shape == (data["pos_grid"].shape[0], model_seg_pdnorm.num_classes)


def test_serialize_single_voxel_scene() -> None:
    """An all-zero grid has `bit_length() == 0`; the depth floor keeps the encoders valid."""
    grid = torch.zeros(6, 3, dtype=torch.long)
    batch = torch.zeros(6, dtype=torch.long)
    code, order, inverse = serialize(grid, batch, orders=["hilbert"])
    assert code.shape == (1, 6)
    assert torch.equal(order.gather(1, inverse), torch.arange(6).unsqueeze(0))


def test_serialize_negative_grid_raises() -> None:
    """Negative grid coordinates would silently wrap to valid codes; serialize rejects them."""
    grid = torch.tensor([[-1, 0, 0], [1, 2, 3]], dtype=torch.long)
    batch = torch.zeros(2, dtype=torch.long)
    with pytest.raises(ValueError, match="non-negative"):
        serialize(grid, batch, orders=["hilbert"])
