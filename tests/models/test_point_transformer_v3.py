from typing import Dict

import pytest
import torch
from torch import Tensor, nn

from torch_pointcloud.models import model_info
from torch_pointcloud.models.point_transformer_v3 import (
    PointTransformerV3Classification,
    PointTransformerV3Segmentation,
    encoder_checkpoint_filter_fn,
    segmentation_checkpoint_filter_fn,
    serialize,
)
from torch_pointcloud.utils.data import DataKeys
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
    model_clf: PointTransformerV3Classification,
    data: Dict[str, Tensor],
) -> None:
    model_clf.reset_classifier(num_classes=42)
    model_clf.cuda()
    logits = model_clf(data["x"], data["pos_grid"], data["batch"])
    assert logits.shape == (int(data["batch"].max()) + 1, 42)


def test_pt_v3_classification_forward_features_and_head(
    model_clf: PointTransformerV3Classification,
    data: Dict[str, Tensor],
) -> None:
    x, _, batch = model_clf.forward_features(data["x"], data["pos_grid"], data["batch"])
    assert x.shape[0] == batch.shape[0]
    logits = model_clf.forward_head(x, batch)
    assert logits.shape == (int(data["batch"].max()) + 1, model_clf.num_classes)


def test_pt_v3_segmentation_forward(model_seg: PointTransformerV3Segmentation, data: Dict[str, Tensor]) -> None:
    logits = model_seg(data["x"], data["pos_grid"], data["batch"])
    assert logits.shape == (data["pos_grid"].shape[0], model_seg.num_classes)


def test_pt_v3_segmentation_forward_features_decoder_head(
    model_seg: PointTransformerV3Segmentation,
    data: Dict[str, Tensor],
) -> None:
    x, _, _, intermediates = model_seg.forward_features(
        data["x"], data["pos_grid"], data["batch"], return_intermediates=True
    )
    assert len(intermediates) > 0
    x, _, _ = model_seg.forward_decoder(x, intermediates)
    logits = model_seg.forward_head(x)
    assert logits.shape == (data["pos_grid"].shape[0], model_seg.num_classes)


def test_pt_v3_segmentation_reset_classifier(
    model_seg: PointTransformerV3Segmentation,
    data: Dict[str, Tensor],
) -> None:
    model_seg.reset_classifier(num_classes=42)
    model_seg.cuda()
    logits = model_seg(data["x"], data["pos_grid"], data["batch"])
    assert logits.shape == (data["pos_grid"].shape[0], 42)


def test_pt_v3_condition_without_pdnorm_conditions_raises(
    model_seg: PointTransformerV3Segmentation,
    data: Dict[str, Tensor],
) -> None:
    with pytest.raises(ValueError, match="without conditional norms"):
        model_seg(data["x"], data["pos_grid"], data["batch"], condition="ScanNet")


def test_pt_v3_pdnorm_conditions_without_condition_raises(
    model_seg_pdnorm: PointTransformerV3Segmentation,
    data: Dict[str, Tensor],
) -> None:
    with pytest.raises(ValueError, match="pass `condition=`"):
        model_seg_pdnorm(data["x"], data["pos_grid"], data["batch"])


def test_pt_v3_pdnorm_forward_with_condition(
    model_seg_pdnorm: PointTransformerV3Segmentation,
    data: Dict[str, Tensor],
) -> None:
    logits = model_seg_pdnorm(data["x"], data["pos_grid"], data["batch"], condition="ScanNet")
    assert logits.shape == (data["pos_grid"].shape[0], model_seg_pdnorm.num_classes)


def test_pt_v3_segmentation_grid_pooling_feeds_the_summed_upsampling_to_cpe(data: Dict[str, Tensor]) -> None:
    model = PointTransformerV3Segmentation(
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
        pooling="grid",
    ).cuda()
    assert not any(block.cpe_from_skip for block in model.decoder.blocks)

    logits = model(data["x"], data["pos_grid"], data["batch"])
    assert logits.shape == (data["x"].shape[0], 10)


def test_pt_v3_segmentation_condition_is_keyword_only(
    model_seg_pdnorm: PointTransformerV3Segmentation, data: Dict[str, Tensor]
) -> None:
    with pytest.raises(TypeError, match="positional argument"):
        model_seg_pdnorm(data["x"], data["pos_grid"], data["batch"], None, "ScanNet")


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


_RELEASED_ENCODER_KEYS = {
    "embedding.mask_token": torch.zeros(1, 4),
    "embedding.stem.linear.weight": torch.ones(4, 2),
    "enc.enc1.block0.mlp.0.fc2.bias": torch.ones(4),
    "enc.enc1.down.norm.0.weight": torch.ones(4),
    "seg_head.weight": torch.ones(3, 4),
}


def test_encoder_checkpoint_filter_fn_keys_a_bare_encoder() -> None:
    filtered = encoder_checkpoint_filter_fn(dict(_RELEASED_ENCODER_KEYS), nn.Linear(2, 4))
    assert sorted(filtered) == [
        "blocks.1.blocks.0.mlp.3.bias",
        "blocks.1.downsample.norm.weight",
        "head.weight",
        "stem.stem.weight",
    ]


def test_encoder_checkpoint_filter_fn_prefixes_a_wrapped_encoder() -> None:
    model = nn.ModuleDict({"encoder": nn.Linear(2, 4), "head": nn.Linear(4, 3)})
    filtered = encoder_checkpoint_filter_fn(dict(_RELEASED_ENCODER_KEYS), model)
    assert sorted(filtered) == [
        "encoder.blocks.1.blocks.0.mlp.3.bias",
        "encoder.blocks.1.downsample.norm.weight",
        "encoder.stem.stem.weight",
        "head.weight",
    ]


_RELEASED_SEGMENTATION_KEYS = {
    "backbone.embedding.stem.linear.weight": torch.ones(4, 2),
    "backbone.enc.enc1.down.norm.0.weight": torch.ones(4),
    "backbone.dec.dec1.up.proj_skip.1.bias": torch.ones(4),
    "backbone.dec.dec0.block0.cpe.0.weight": torch.ones(4, 3, 3, 3, 4),
    "backbone.dec.dec0.block0.attn.qkv.weight": torch.ones(12, 4),
    "seg_head.weight": torch.ones(3, 4),
}


def test_segmentation_checkpoint_filter_fn_numbers_the_decoder_in_execution_order(
    model_seg: PointTransformerV3Segmentation,
) -> None:
    assert len(model_seg.decoder.blocks) == 2
    filtered = segmentation_checkpoint_filter_fn(dict(_RELEASED_SEGMENTATION_KEYS), model_seg)
    assert sorted(filtered) == [
        "decoder.blocks.0.upsample.norm_skip.bias",
        "decoder.blocks.1.blocks.0.attn.qkv.weight",
        "decoder.blocks.1.blocks.0.cpe_conv.weight",
        "encoder.blocks.1.downsample.norm.weight",
        "encoder.stem.stem.weight",
        "head.weight",
    ]


def test_nuscenes_registered_transform_runs_without_labels() -> None:
    """The registered nuScenes transform runs on an unlabeled scan, as at inference."""
    transform = model_info("ptv3-base.nuscenes.pointcept")["transform"]
    assert transform is not None
    torch.manual_seed(0)
    data = {
        DataKeys.POS: torch.rand(500, 3) * 50.0,
        DataKeys.INTENSITY: torch.randint(0, 256, (500, 1)).float(),
    }

    out = transform(data)

    assert DataKeys.SEGMENT not in out
    assert out[DataKeys.X].size(0) == out[DataKeys.POS_GRID].size(0)
    assert out[DataKeys.INVERSE].shape == (500,)
