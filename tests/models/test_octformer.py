from typing import Any, Dict, Optional
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from torch_pointcloud.models.octformer import CPE, OctFormerBlock, OctFormerClassification, OctFormerSegmentation
from torch_pointcloud.ops.octree import build_octree
from torch_pointcloud.utils.imports import _DWCONV_AVAILABLE, _OCNN_AVAILABLE

# See: https://docs.pytest.org/en/stable/how-to/skipping.html#summary
pytestmark = pytest.mark.skipif(
    not (_OCNN_AVAILABLE and _DWCONV_AVAILABLE),
    reason="ocnn or dwconv is not installed",
)


@pytest.fixture
def data() -> Dict[str, Any]:
    torch.manual_seed(42)
    lengths = torch.tensor([256, 512])
    pos = torch.rand(int(lengths.sum()), 3) * 1.8 - 0.9  # [-0.9, 0.9]
    normal = torch.nn.functional.normalize(torch.randn(int(lengths.sum()), 3), dim=1)
    batch = torch.repeat_interleave(torch.arange(len(lengths)), lengths)

    depth = 5
    octree = build_octree(
        pos=pos,
        normal=normal,
        batch=batch,
        batch_size=int(len(lengths)),
        depth=depth,
        full_depth=2,
    )
    octree.construct_all_neigh()
    x = octree.get_input_feature("ND", nempty=False)

    return dict(x=x, octree=octree, depth=depth, pos=pos, batch=batch)


@pytest.fixture
def model_clf() -> OctFormerClassification:
    return OctFormerClassification(
        in_channels=4,
        num_classes=10,
        stem_channels=(8, 16),
        encoder_channels=(16, 32),
        head_channels=None,
        encoder_depths=(1, 1),
        num_heads=(2, 4),
        patch_size=4,
        dilation=2,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.5,
        nempty=False,
        use_checkpoint=False,
        use_rpe=True,
        use_dwconv=False,
        act="gelu",
        act_kwargs=None,
        act_first=False,
        norm="batch_norm",
        norm_kwargs=None,
        bias=True,
        dropout=0.0,
        global_pool="mean",
    )


@pytest.fixture
def model_seg() -> OctFormerSegmentation:
    return OctFormerSegmentation(
        in_channels=4,
        num_classes=10,
        stem_channels=(8, 16),
        encoder_channels=(16, 32),
        encoder_depths=(1, 1),
        num_heads=(2, 4),
        head_channels=None,
        fpn_channels=16,
        patch_size=4,
        dilation=2,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.5,
        nempty=False,
        use_checkpoint=False,
        use_rpe=True,
        use_dwconv=False,
        act="gelu",
        act_kwargs=None,
        act_first=False,
        norm="batch_norm",
        norm_kwargs=None,
        bias=True,
        dropout=0.5,
    )


def test_octformer_classification_forward(model_clf: OctFormerClassification, data: Dict[str, Any]) -> None:
    logits = model_clf(data["x"], data["octree"], data["depth"])
    assert logits.shape == (int(data["batch"].max()) + 1, model_clf.num_classes)
    assert logits.dtype == data["x"].dtype


def test_octformer_block_cpe_order(model_clf: OctFormerClassification, data: Dict[str, Any]) -> None:
    model_clf.eval()
    with torch.no_grad():
        first = model_clf(data["x"], data["octree"], data["depth"])
        for block in model_clf.encoder.modules():
            if isinstance(block, OctFormerBlock):
                block.cpe_first = False
        after = model_clf(data["x"], data["octree"], data["depth"])
    assert first.shape == after.shape
    assert not torch.allclose(first, after)


def test_octformer_classification_reset_classifier(model_clf: OctFormerClassification, data: Dict[str, Any]) -> None:
    model_clf.reset_classifier(num_classes=42)
    logits = model_clf(data["x"], data["octree"], data["depth"])
    assert logits.shape == (int(data["batch"].max()) + 1, 42)


def test_octformer_classification_num_classes_zero_returns_features(
    model_clf: OctFormerClassification,
    data: Dict[str, Any],
) -> None:
    model_clf.reset_classifier(num_classes=0)
    assert isinstance(model_clf.head, torch.nn.Identity)
    out = model_clf(data["x"], data["octree"], data["depth"])
    assert out.shape == (int(data["batch"].max()) + 1, model_clf.num_features)


def test_octformer_segmentation_forward(model_seg: OctFormerSegmentation, data: Dict[str, Any]) -> None:
    logits = model_seg(data["x"], data["octree"], data["depth"], data["pos"], data["batch"])
    assert logits.shape == (data["pos"].shape[0], model_seg.num_classes)
    assert logits.dtype == data["x"].dtype


def test_octformer_segmentation_reset_classifier(model_seg: OctFormerSegmentation, data: Dict[str, Any]) -> None:
    model_seg.reset_classifier(num_classes=42)
    logits = model_seg(data["x"], data["octree"], data["depth"], data["pos"], data["batch"])
    assert logits.shape == (data["pos"].shape[0], 42)


def test_octformer_segmentation_num_classes_zero_returns_features(
    model_seg: OctFormerSegmentation,
    data: Dict[str, Any],
) -> None:
    model_seg.reset_classifier(num_classes=0)
    assert isinstance(model_seg.head, torch.nn.Identity)
    out = model_seg(data["x"], data["octree"], data["depth"], data["pos"], data["batch"])
    assert out.shape == (data["pos"].shape[0], model_seg.num_features)


def test_octformer_classification_forward_features_and_head(
    model_clf: OctFormerClassification,
    data: Dict[str, Any],
) -> None:
    x = model_clf.forward_features(data["x"], data["octree"], data["depth"])
    assert x.dim() == 2
    logits = model_clf.forward_head(x, data["octree"], model_clf.get_head_depth(data["depth"]))
    assert logits.shape == (int(data["batch"].max()) + 1, model_clf.num_classes)


def test_octformer_segmentation_forward_features_decoder_head(
    model_seg: OctFormerSegmentation,
    data: Dict[str, Any],
) -> None:
    x, intermediates = model_seg.forward_features(data["x"], data["octree"], data["depth"], return_intermediates=True)
    assert len(intermediates) > 0
    x = model_seg.forward_decoder(x, data["octree"], data["depth"], intermediates)
    logits = model_seg.forward_head(x, data["octree"], data["depth"], data["pos"], data["batch"])
    assert logits.shape == (data["pos"].shape[0], model_seg.num_classes)


def test_octformer_reset_classifier_keeps_head_act(model_clf: OctFormerClassification) -> None:
    """The registered factories set `head_act="relu"`; a reset must rebuild the head with the same act."""
    model_clf.head_act = "relu"
    model_clf.reset_classifier(10)
    assert type(model_clf.head.act).__name__ == "ReLU"
    model_clf.reset_classifier(5)
    assert type(model_clf.head.act).__name__ == "ReLU"


@pytest.mark.parametrize("nempty", [pytest.param(True, id="non-empty-nodes"), pytest.param(False, id="all-nodes")])
def test_cpe_depthwise_conv_on_cpu_matches_neighbor_sum(data: Dict[str, Any], nempty: bool) -> None:
    torch.manual_seed(0)
    octree, depth = data["octree"], data["depth"]
    cpe = CPE(8, kernel_size=3, nempty=nempty, use_dwconv=True).eval()
    num_nodes = int(octree.nnum_nempty[depth] if nempty else octree.nnum[depth])
    x = torch.randn(num_nodes, 8)

    neighbors = octree.get_neigh(depth, cpe.conv.kernel, 1, nempty)
    padded = torch.cat([x, x.new_zeros(1, 8)])
    expected = (padded[neighbors] * cpe.conv.weights[:, 0, :]).sum(dim=1)
    with torch.no_grad():
        out = cpe(x, octree, depth)

    torch.testing.assert_close(out, cpe.norm(expected), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    "head_channels",
    [pytest.param(None, id="linear-head"), pytest.param(16, id="one-hidden-layer")],
)
def test_octformer_classification_head_applies_dropout_once(data: Dict[str, Any], head_channels: Optional[int]) -> None:
    model = OctFormerClassification(
        in_channels=4,
        num_classes=10,
        stem_channels=(8, 16),
        encoder_channels=(16, 32),
        head_channels=head_channels,
        encoder_depths=(1, 1),
        num_heads=(2, 4),
        patch_size=4,
        dilation=2,
        nempty=False,
        dropout=0.5,
    ).train()
    head_depth = model.get_head_depth(data["depth"])
    x = torch.randn(int(data["octree"].nnum[head_depth]), 32)

    with patch.object(F, "dropout", wraps=F.dropout) as dropout:
        model.forward_head(x, data["octree"], head_depth)

    rates = [call.kwargs["p"] for call in dropout.call_args_list]
    assert [rate for rate in rates if rate > 0] == [0.5]
