from typing import Dict, List, Optional, Tuple, Union

import pytest
import torch
from torch import Tensor, nn

from torch_pointcloud.layers.conditional_heads import ConditionalLinear, CosineSimilarityHead
from torch_pointcloud.layers.dropouts import DropPath
from torch_pointcloud.models import model_info
from torch_pointcloud.models.point_transformer_v3 import (
    PointTransformerV3Classification,
    PointTransformerV3Decoder,
    PointTransformerV3DecoderBlock,
    PointTransformerV3Segmentation,
    encoder_checkpoint_filter_fn,
    segmentation_checkpoint_filter_fn,
    serialize,
)
from torch_pointcloud.ops import serialization
from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.imports import _CUDA_AVAILABLE, _SPCONV_AVAILABLE

# See: https://docs.pytest.org/en/stable/how-to/skipping.html#summary
pytestmark = pytest.mark.skipif(not _SPCONV_AVAILABLE, reason="spconv is not installed")
requires_cuda = pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA is not available")
DEVICE = "cuda" if _CUDA_AVAILABLE else "cpu"


@pytest.fixture
def data() -> Dict[str, Tensor]:
    torch.manual_seed(42)
    lengths = torch.tensor([256, 512])
    pos_grid = torch.randint(0, 64, (int(lengths.sum()), 3))
    x = torch.randn(int(lengths.sum()), 6)
    batch = torch.repeat_interleave(torch.arange(len(lengths)), lengths)
    return dict(
        x=x.to(DEVICE),
        pos_grid=pos_grid.to(DEVICE),
        batch=batch.to(DEVICE),
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
    ).to(DEVICE)


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
    ).to(DEVICE)


@pytest.fixture
def model_seg_layer_norm() -> PointTransformerV3Segmentation:
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
        norm="layer_norm",
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
        stem_type="linear",
        act_kwargs=None,
        norm_kwargs=None,
    ).to(DEVICE)


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
    ).to(DEVICE)


@pytest.fixture
def model_seg_per_condition() -> PointTransformerV3Segmentation:
    return PointTransformerV3Segmentation(
        in_channels=6,
        num_classes=(10, 4),
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
        condition="ScanNet",
    ).to(DEVICE)


@pytest.fixture
def model_seg_language() -> PointTransformerV3Segmentation:
    return PointTransformerV3Segmentation(
        in_channels=6,
        num_classes=(10, 4),
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
        condition="S3DIS",
        text_embed_dim=8,
        text_scale=10.0,
    ).to(DEVICE)


@requires_cuda
def test_pt_v3_classification_forward(model_clf: PointTransformerV3Classification, data: Dict[str, Tensor]) -> None:
    logits = model_clf(data["x"], data["pos_grid"], data["batch"])
    assert logits.shape == (int(data["batch"].max()) + 1, model_clf.num_classes)


@requires_cuda
def test_pt_v3_classification_reset_classifier(
    model_clf: PointTransformerV3Classification,
    data: Dict[str, Tensor],
) -> None:
    model_clf.reset_classifier(num_classes=42)
    model_clf.to(DEVICE)
    logits = model_clf(data["x"], data["pos_grid"], data["batch"])
    assert logits.shape == (int(data["batch"].max()) + 1, 42)


@requires_cuda
def test_pt_v3_classification_forward_features_and_head(
    model_clf: PointTransformerV3Classification,
    data: Dict[str, Tensor],
) -> None:
    x, _, batch = model_clf.forward_features(data["x"], data["pos_grid"], data["batch"])
    assert x.shape[0] == batch.shape[0]
    logits = model_clf.forward_head(x, batch)
    assert logits.shape == (int(data["batch"].max()) + 1, model_clf.num_classes)


@requires_cuda
def test_pt_v3_segmentation_forward(model_seg: PointTransformerV3Segmentation, data: Dict[str, Tensor]) -> None:
    logits = model_seg(data["x"], data["pos_grid"], data["batch"])
    assert logits.shape == (data["pos_grid"].shape[0], model_seg.num_classes)


@requires_cuda
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(torch.float16, id="float16"),
        pytest.param(torch.bfloat16, id="bfloat16"),
    ],
)
def test_pt_v3_segmentation_inference_under_autocast(
    dtype: torch.dtype, model_seg: PointTransformerV3Segmentation, data: Dict[str, Tensor]
) -> None:
    """The half-precision pooled features reach the sparse convolutions of the next stage, whose weights are float32."""
    model_seg.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
        logits = model_seg(data["x"], data["pos_grid"], data["batch"])
    assert logits.shape == (data["x"].shape[0], 10)
    assert torch.isfinite(logits).all()


@requires_cuda
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


@requires_cuda
def test_pt_v3_segmentation_reset_classifier(
    model_seg: PointTransformerV3Segmentation,
    data: Dict[str, Tensor],
) -> None:
    model_seg.reset_classifier(num_classes=42)
    model_seg.to(DEVICE)
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


@requires_cuda
def test_pt_v3_pdnorm_forward_with_condition(
    model_seg_pdnorm: PointTransformerV3Segmentation,
    data: Dict[str, Tensor],
) -> None:
    logits = model_seg_pdnorm(data["x"], data["pos_grid"], data["batch"], condition="ScanNet")
    assert logits.shape == (data["pos_grid"].shape[0], model_seg_pdnorm.num_classes)


@requires_cuda
def test_pt_v3_segmentation_one_head_per_condition(
    model_seg_per_condition: PointTransformerV3Segmentation,
    data: Dict[str, Tensor],
) -> None:
    logits = model_seg_per_condition(data["x"], data["pos_grid"], data["batch"])
    assert logits.shape == (data["x"].shape[0], 10)
    logits = model_seg_per_condition(data["x"], data["pos_grid"], data["batch"], condition="S3DIS")
    assert logits.shape == (data["x"].shape[0], 4)


@requires_cuda
def test_pt_v3_segmentation_stages_default_to_the_model_condition(
    model_seg_per_condition: PointTransformerV3Segmentation,
    data: Dict[str, Tensor],
) -> None:
    model_seg_per_condition.eval()
    with torch.no_grad():
        x, _, _, intermediates = model_seg_per_condition.forward_features(
            data["x"], data["pos_grid"], data["batch"], return_intermediates=True
        )
        x, _, _ = model_seg_per_condition.forward_decoder(x, intermediates)
        expected = model_seg_per_condition(data["x"], data["pos_grid"], data["batch"], condition="ScanNet")
    assert torch.allclose(model_seg_per_condition.forward_head(x), expected)


@requires_cuda
def test_pt_v3_segmentation_cosine_similarity_head(
    model_seg_language: PointTransformerV3Segmentation,
    data: Dict[str, Tensor],
) -> None:
    logits = model_seg_language(data["x"], data["pos_grid"], data["batch"])
    assert logits.shape == (data["x"].shape[0], 4)
    assert logits.abs().max() <= 10.0 + 1e-4
    logits = model_seg_language(data["x"], data["pos_grid"], data["batch"], condition="ScanNet")
    assert logits.shape == (data["x"].shape[0], 10)

    x, _, _, intermediates = model_seg_language.forward_features(
        data["x"], data["pos_grid"], data["batch"], return_intermediates=True, condition="S3DIS"
    )
    x, _, _ = model_seg_language.forward_decoder(x, intermediates, condition="S3DIS")
    features = model_seg_language.forward_head(x, pre_logits=True)
    assert features.shape == (data["x"].shape[0], model_seg_language.num_features)


def test_pt_v3_segmentation_one_head_per_condition_routes_the_condition(
    model_seg_per_condition: PointTransformerV3Segmentation,
) -> None:
    head = model_seg_per_condition.head
    assert isinstance(head, ConditionalLinear)
    assert head.conditions == ("ScanNet", "S3DIS")
    assert model_seg_per_condition.num_classes == 10
    features = torch.randn(7, model_seg_per_condition.num_features, device=DEVICE)
    assert torch.equal(model_seg_per_condition.forward_head(features), head.layers[0](features))
    assert torch.equal(model_seg_per_condition.forward_head(features, condition="S3DIS"), head.layers[1](features))
    assert torch.equal(
        model_seg_per_condition.forward_head(features, condition=["S3DIS", "S3DIS"]),
        head.layers[1](features),
    )


def test_pt_v3_segmentation_mixed_conditions_in_one_batch_raise(
    model_seg_per_condition: PointTransformerV3Segmentation,
) -> None:
    features = torch.randn(7, model_seg_per_condition.num_features, device=DEVICE)
    with pytest.raises(ValueError, match=r"conditions \['S3DIS', 'ScanNet'\] in one batch"):
        model_seg_per_condition.forward_head(features, condition=["S3DIS", "ScanNet"])


def test_pt_v3_segmentation_head_rejects_a_condition_without_pdnorm_conditions(
    model_seg: PointTransformerV3Segmentation,
) -> None:
    features = torch.randn(7, model_seg.num_features, device=DEVICE)
    with pytest.raises(ValueError, match="without conditional norms"):
        model_seg.forward_head(features, condition="ScanNet")


def test_pt_v3_segmentation_cosine_similarity_head_routes_the_condition(
    model_seg_language: PointTransformerV3Segmentation,
) -> None:
    head = model_seg_language.head
    assert isinstance(head, CosineSimilarityHead)
    assert head.conditions == ("ScanNet", "S3DIS")
    assert model_seg_language.num_classes == 4
    features = torch.randn(7, model_seg_language.num_features, device=DEVICE)
    assert torch.equal(model_seg_language.forward_head(features), head(features, condition="S3DIS"))
    assert model_seg_language.forward_head(features, condition="ScanNet").shape == (7, 10)
    assert torch.equal(model_seg_language.forward_head(features, pre_logits=True), features)


@pytest.mark.parametrize(
    ("num_classes", "head_type", "logits_dim"),
    [
        pytest.param(3, nn.Linear, 3, id="one-count"),
        pytest.param((3, 5), ConditionalLinear, 5, id="one-count-per-condition"),
        pytest.param(0, nn.Identity, 16, id="headless"),
    ],
)
def test_pt_v3_segmentation_reset_classifier_builds_linear_heads(
    model_seg_language: PointTransformerV3Segmentation,
    num_classes: Union[int, Tuple[int, ...]],
    head_type: type,
    logits_dim: int,
) -> None:
    model_seg_language.reset_classifier(num_classes)
    model_seg_language.to(DEVICE)
    assert type(model_seg_language.head) is head_type
    assert model_seg_language.text_embed_dim is None
    features = torch.randn(7, model_seg_language.num_features, device=DEVICE)
    assert model_seg_language.forward_head(features).shape == (7, logits_dim)


def test_pt_v3_segmentation_reset_classifier_with_class_embedding_keeps_the_projection(
    model_seg_language: PointTransformerV3Segmentation,
) -> None:
    head = model_seg_language.head
    assert isinstance(head, CosineSimilarityHead)
    proj = head.proj.weight.clone()
    tables = [torch.randn(2, 8), torch.randn(3, 8)]
    model_seg_language.reset_classifier((2, 3), class_embedding=tables)
    assert model_seg_language.head is head
    assert torch.equal(head.proj.weight, proj)
    assert model_seg_language.num_classes == 3
    assert model_seg_language.num_classes_per_condition == (2, 3)
    features = torch.randn(7, model_seg_language.num_features, device=DEVICE)
    assert model_seg_language.forward_head(features).shape == (7, 3)
    assert model_seg_language.forward_head(features, condition="ScanNet").shape == (7, 2)

    model_seg_language.reset_classifier(5, class_embedding=torch.randn(5, 8))
    assert model_seg_language.num_classes == 5
    assert model_seg_language.num_classes_per_condition is None
    assert model_seg_language.forward_head(features, condition="ScanNet").shape == (7, 5)


@pytest.mark.parametrize(
    ("num_classes", "class_embedding", "match"),
    [
        pytest.param(4, torch.randn(5, 8), "for class tables of \\[5\\] rows", id="rows"),
        pytest.param(
            (4, 2),
            [torch.randn(4, 8), torch.randn(3, 8)],
            "for class tables of \\[4, 3\\] rows",
            id="rows-per-condition",
        ),
    ],
)
def test_pt_v3_segmentation_reset_classifier_with_class_embedding_checks_the_counts(
    model_seg_language: PointTransformerV3Segmentation,
    num_classes: Union[int, Tuple[int, ...]],
    class_embedding: Union[Tensor, List[Tensor]],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        model_seg_language.reset_classifier(num_classes, class_embedding=class_embedding)
    assert model_seg_language.num_classes == 4


def test_pt_v3_segmentation_reset_classifier_with_class_embedding_requires_a_text_head(
    model_seg_per_condition: PointTransformerV3Segmentation,
) -> None:
    with pytest.raises(ValueError, match="this model has a `ConditionalLinear` head"):
        model_seg_per_condition.reset_classifier(3, class_embedding=torch.randn(3, 8))


@pytest.mark.parametrize(
    ("num_classes", "pdnorm_conditions", "condition", "text_embed_dim", "text_scale"),
    [
        pytest.param((10, 4), None, None, None, None, id="counts-without-conditions"),
        pytest.param((10, 4), ("A",), "A", None, None, id="count-condition-mismatch"),
        pytest.param((10, 4), ("A", "B"), None, None, None, id="counts-without-default-condition"),
        pytest.param((10, 4), ("A", "B"), "C", None, None, id="counts-with-unknown-default-condition"),
        pytest.param(10, None, None, 8, None, id="text-embed-dim-without-scale"),
        pytest.param(10, None, None, None, 10.0, id="text-scale-without-embed-dim"),
    ],
)
def test_pt_v3_segmentation_validates_the_head_arguments(
    num_classes: Union[int, Tuple[int, ...]],
    pdnorm_conditions: Optional[Tuple[str, ...]],
    condition: Optional[str],
    text_embed_dim: Optional[int],
    text_scale: Optional[float],
) -> None:
    with pytest.raises(ValueError):
        PointTransformerV3Segmentation(
            in_channels=6,
            num_classes=num_classes,
            pdnorm_conditions=pdnorm_conditions,
            condition=condition,
            text_embed_dim=text_embed_dim,
            text_scale=text_scale,
        )


@requires_cuda
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
    ).to(DEVICE)
    assert not any(block.cpe_from_skip for block in model.decoder.blocks)

    logits = model(data["x"], data["pos_grid"], data["batch"])
    assert logits.shape == (data["x"].shape[0], 10)


def test_pt_v3_segmentation_condition_is_keyword_only(
    model_seg_pdnorm: PointTransformerV3Segmentation, data: Dict[str, Tensor]
) -> None:
    with pytest.raises(TypeError, match="positional argument"):
        model_seg_pdnorm(data["x"], data["pos_grid"], data["batch"], None, "ScanNet")


def test_serialize_is_deprecated() -> None:
    grid = torch.tensor([[0, 0, 0], [1, 2, 3]], dtype=torch.long)
    batch = torch.zeros(2, dtype=torch.long)
    with pytest.warns(FutureWarning, match="ops.serialization.serialize"):
        code, _, _ = serialize(grid, batch, orders=["hilbert"])
    assert torch.equal(code, serialization.serialize(grid, batch, orders=["hilbert"])[0])


@pytest.mark.parametrize(
    "decoder_depths,expected",
    [
        pytest.param((2, 1, 1, 1), [[0.3, 0.225], [0.15], [0.075], [0.0]], id="coarse-heavy"),
        pytest.param((1, 1, 1, 2), [[0.3], [0.225], [0.15], [0.075, 0.0]], id="fine-heavy"),
    ],
)
def test_pt_v3_decoder_drop_path_follows_the_reference_order(
    decoder_depths: Tuple[int, ...],
    expected: List[List[float]],
) -> None:
    """The reference spreads the rates over the stages finest first, each stage in reverse."""
    decoder = PointTransformerV3Decoder(
        encoder_channels=(12, 12, 12, 12, 12),
        decoder_depths=decoder_depths,
        decoder_channels=(12, 12, 12, 12),
        decoder_num_heads=(2, 2, 2, 2),
        decoder_patch_size=(16, 16, 16, 16),
        drop_path=0.3,
        use_flash_attn=False,
    )
    rates = []
    for stage in decoder.blocks:
        assert isinstance(stage, PointTransformerV3DecoderBlock)
        rates.append([b.drop_path.drop_prob if isinstance(b.drop_path, DropPath) else 0.0 for b in stage.blocks])
    assert rates == [pytest.approx(r) for r in expected]


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
    "backbone.embedding.stem.linear.weight": torch.ones(4, 6),
    "backbone.enc.enc1.down.norm.0.weight": torch.ones(4),
    "backbone.dec.dec1.up.proj_skip.1.bias": torch.ones(4),
    "backbone.dec.dec0.block0.cpe.0.weight": torch.ones(4, 3, 3, 3, 4),
    "backbone.dec.dec0.block0.attn.qkv.weight": torch.ones(12, 4),
    "seg_head.weight": torch.ones(3, 4),
}
_TRAINING_SEGMENTATION_KEYS = {
    "module.backbone.embedding.stem.conv.weight": torch.ones(4, 3, 3, 3, 6),
    "module.backbone.embedding.stem.norm.running_mean": torch.ones(4),
    "module.backbone.enc.enc1.down.norm.0.weight": torch.ones(4),
    "module.backbone.dec.dec1.up.proj_skip.1.bias": torch.ones(4),
    "module.backbone.dec.dec0.block0.cpe.0.weight": torch.ones(4, 3, 3, 3, 4),
    "module.backbone.dec.dec0.block0.attn.qkv.weight": torch.ones(12, 4),
    "module.seg_head.weight": torch.ones(3, 4),
}


def test_segmentation_checkpoint_filter_fn_numbers_the_decoder_in_execution_order(
    model_seg_layer_norm: PointTransformerV3Segmentation,
) -> None:
    """A released checkpoint of a linear stem, layer-norm model: the decoder stages are reversed and the norms keep
    their names."""
    assert len(model_seg_layer_norm.decoder.blocks) == 2
    filtered = segmentation_checkpoint_filter_fn(dict(_RELEASED_SEGMENTATION_KEYS), model_seg_layer_norm)
    assert sorted(filtered) == [
        "decoder.blocks.0.upsample.norm_skip.bias",
        "decoder.blocks.1.blocks.0.attn.qkv.weight",
        "decoder.blocks.1.blocks.0.cpe_conv.weight",
        "encoder.blocks.1.downsample.norm.weight",
        "encoder.stem.stem.weight",
        "head.weight",
    ]
    assert all(key in model_seg_layer_norm.state_dict() for key in filtered)


def test_segmentation_checkpoint_filter_fn_wraps_the_norms_of_a_batch_norm_training_checkpoint(
    model_seg: PointTransformerV3Segmentation,
) -> None:
    """A distributed training checkpoint of a sparse-conv stem, batch-norm model: the `module.` prefix goes, the
    decoder stages are reversed and the stem, pooling and unpooling norms land under their `module`."""
    assert len(model_seg.decoder.blocks) == 2
    filtered = segmentation_checkpoint_filter_fn(dict(_TRAINING_SEGMENTATION_KEYS), model_seg)
    assert sorted(filtered) == [
        "decoder.blocks.0.upsample.norm_skip.module.bias",
        "decoder.blocks.1.blocks.0.attn.qkv.weight",
        "decoder.blocks.1.blocks.0.cpe_conv.weight",
        "encoder.blocks.1.downsample.norm.module.weight",
        "encoder.stem.norm.module.running_mean",
        "encoder.stem.stem.weight",
        "head.weight",
    ]
    assert all(key in model_seg.state_dict() for key in filtered)


@pytest.mark.parametrize(
    "name",
    [
        pytest.param("ptv3-base.nuscenes.pointcept", id="nuscenes"),
        pytest.param("ptv3-base.s3dis-area5.pointcept", id="s3dis-area5"),
        pytest.param("ptv3-base.scannet20.pointcept", id="scannet20"),
        pytest.param("ptv3-base.scannet200.pointcept", id="scannet200"),
        pytest.param("ptv3.dales.jayakumar-pujar", id="dales"),
        pytest.param("ptv3.gridnet-hd.heig-vd-geo", id="gridnet-hd"),
        pytest.param("ptv3-ppt.scannet20.pointcept", id="ppt-scannet20"),
        pytest.param("ptv3-ppt.s3dis-area5.pointcept", id="ppt-s3dis-area5"),
        pytest.param("ptv3-ppt.scannet20.labelmaker", id="labelmaker-scannet20"),
        pytest.param("ptv3-ppt.scannet200.labelmaker", id="labelmaker-scannet200"),
    ],
)
def test_segmentation_registered_transform_runs_without_labels(name: str) -> None:
    """A registered transform runs on a cloud that carries the model inputs and no label, as at inference."""
    transform = model_info(name)["transform"]
    assert transform is not None
    torch.manual_seed(0)
    data = {
        DataKeys.POS: torch.rand(500, 3) * 2.0,
        DataKeys.COLOR: torch.randint(0, 256, (500, 3), dtype=torch.uint8),
        DataKeys.NORMAL: torch.nn.functional.normalize(torch.randn(500, 3), dim=1),
        DataKeys.INTENSITY: torch.rand(500, 1),
        DataKeys.RETURN_NUMBER: torch.randint(1, 4, (500, 1)),
        DataKeys.NUMBER_OF_RETURNS: torch.randint(1, 4, (500, 1)),
    }

    out = transform(data)

    assert DataKeys.SEGMENT not in out
    assert out[DataKeys.X].size(0) == out[DataKeys.POS_GRID].size(0)
    assert out[DataKeys.INVERSE].shape == (500,)


@pytest.mark.parametrize(
    "name",
    [
        pytest.param("ptv3-ppt.scannet20.pointcept"),
        pytest.param("ptv3-ppt.s3dis-area5.pointcept"),
        pytest.param("ptv3-ppt.scannet20.labelmaker"),
        pytest.param("ptv3-ppt.scannet200.labelmaker"),
    ],
)
def test_ptv3_per_condition_registration_lists_the_default_condition_classes(name: str) -> None:
    info = model_info(name)
    hparams = info["hparams"]
    weights = info["weights"]
    assert weights is not None and "classes" in weights
    index = list(hparams["pdnorm_conditions"]).index(hparams["condition"])
    assert len(weights["classes"]) == hparams["num_classes"][index]
