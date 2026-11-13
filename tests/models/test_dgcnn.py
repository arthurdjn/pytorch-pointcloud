from typing import Dict

import pytest
import torch
from torch import Tensor

from torch_pointcloud.models import model_info
from torch_pointcloud.models.dgcnn import (
    DGCNNClassification,
    DGCNNEncoderBlock,
    DGCNNPartSegmentation,
    DGCNNSegmentation,
)
from torch_pointcloud.utils.data import DataKeys, collate, select_inputs
from torch_pointcloud.utils.imports import _PYG_LIB_AVAILABLE

# See: https://docs.pytest.org/en/stable/how-to/skipping.html#summary
pytestmark = pytest.mark.skipif(
    not _PYG_LIB_AVAILABLE,
    reason="pyg-lib is not installed",
)


@pytest.fixture
def data() -> Dict[str, Tensor]:
    torch.manual_seed(42)
    lengths = torch.tensor([256, 512])
    pos = torch.randn(int(lengths.sum()), 3)
    x = torch.randn(int(lengths.sum()), 3)
    batch = torch.repeat_interleave(torch.arange(len(lengths)), lengths)
    return dict(x=x, pos=pos, batch=batch)


@pytest.fixture
def model_clf() -> DGCNNClassification:
    return DGCNNClassification(
        in_channels=3,
        num_classes=10,
        spatial_dim=3,
        stnet_local_channels=None,
        stnet_global_channels=None,
        head_channels=[32],
        channels=[32, 64],
        proj_channels=64,
        num_neighbors=8,
        act="relu",
        act_kwargs=None,
        act_first=False,
        norm="batch_norm",
        norm_kwargs=None,
        bias=True,
        dropout=0.0,
        global_pool="max",
    )


@pytest.fixture
def model_seg() -> DGCNNSegmentation:
    return DGCNNSegmentation(
        in_channels=3,
        num_classes=10,
        spatial_dim=3,
        stnet_local_channels=None,
        stnet_global_channels=None,
        proj_channels=64,
        channels=[32, 64],
        head_channels=[32],
        num_neighbors=8,
        act="relu",
        act_kwargs=None,
        act_first=False,
        norm="batch_norm",
        norm_kwargs=None,
        bias=True,
        dropout=0.0,
    )


def test_dgcnn_head_dropout_placement_matches_reference() -> None:
    """Classification regularizes every hidden layer, semantic segmentation only its last hidden layer,
    part segmentation every hidden layer except the last one."""
    clf = DGCNNClassification(
        in_channels=3, num_classes=10, channels=[32, 64], head_channels=[32, 16], num_neighbors=8, dropout=0.5
    )
    assert clf.head.dropout == [0.5, 0.5, 0.0]
    seg = DGCNNSegmentation(
        in_channels=3, num_classes=10, channels=[32, 64], head_channels=[32, 16], num_neighbors=8, dropout=0.5
    )
    assert seg.head.dropout == [0.0, 0.5, 0.0]
    part = DGCNNPartSegmentation(
        in_channels=3,
        num_classes=10,
        num_categories=4,
        channels=[32, 64],
        head_channels=[32, 16, 8],
        num_neighbors=8,
        dropout=0.5,
    )
    assert part.head.dropout == [0.5, 0.5, 0.0, 0.0]


def test_dgcnn_classification_forward(model_clf: DGCNNClassification, data: Dict[str, Tensor]) -> None:
    logits = model_clf(data["x"], data["pos"], data["batch"])
    assert logits.shape == (int(data["batch"].max()) + 1, model_clf.num_classes)
    assert logits.dtype == data["x"].dtype


def test_dgcnn_classification_reset_classifier(model_clf: DGCNNClassification, data: Dict[str, Tensor]) -> None:
    model_clf.reset_classifier(num_classes=42)
    logits = model_clf(data["x"], data["pos"], data["batch"])
    assert logits.shape == (int(data["batch"].max()) + 1, 42)


def test_dgcnn_segmentation_forward(model_seg: DGCNNSegmentation, data: Dict[str, Tensor]) -> None:
    logits = model_seg(data["x"], data["pos"], data["batch"])
    assert logits.shape == (data["pos"].shape[0], model_seg.num_classes)
    assert logits.dtype == data["x"].dtype


def test_dgcnn_segmentation_reset_classifier(model_seg: DGCNNSegmentation, data: Dict[str, Tensor]) -> None:
    model_seg.reset_classifier(num_classes=42)
    logits = model_seg(data["x"], data["pos"], data["batch"])
    assert logits.shape == (data["pos"].shape[0], 42)


def test_dgcnn_encoder_block_act_first_forwarded() -> None:
    block = DGCNNEncoderBlock(in_channels=3, out_channels=[16], num_neighbors=8, act_first=True)
    assert block.conv.nn.act_first is True


def test_dgcnn_classification_num_classes_zero_returns_features(
    model_clf: DGCNNClassification,
    data: Dict[str, Tensor],
) -> None:
    model_clf.reset_classifier(num_classes=0)
    assert isinstance(model_clf.head, torch.nn.Identity)
    out = model_clf(data["x"], data["pos"], data["batch"])
    assert out.shape == (int(data["batch"].max()) + 1, model_clf.num_features)


def test_dgcnn_segmentation_forward_head_pre_logits(model_seg: DGCNNSegmentation, data: Dict[str, Tensor]) -> None:
    x, _, batch = model_seg.forward_features(data["x"], data["pos"], data["batch"])
    feats = model_seg.forward_head(x, pre_logits=True)
    assert torch.equal(feats, x)
    assert feats.shape[1] == model_seg.num_features


def test_dgcnn_classification_forward_features_and_head(
    model_clf: DGCNNClassification,
    data: Dict[str, Tensor],
) -> None:
    x, _, batch = model_clf.forward_features(data["x"], data["pos"], data["batch"])
    assert x.shape[0] == batch.shape[0]
    logits = model_clf.forward_head(x, batch)
    assert logits.shape == (int(data["batch"].max()) + 1, model_clf.num_classes)


def test_dgcnn_segmentation_forward_features_and_head(model_seg: DGCNNSegmentation, data: Dict[str, Tensor]) -> None:
    x, _, batch = model_seg.forward_features(data["x"], data["pos"], data["batch"])
    assert x.shape[0] == batch.shape[0] == data["pos"].shape[0]
    logits = model_seg.forward_head(x)
    assert logits.shape == (data["pos"].shape[0], model_seg.num_classes)


@pytest.fixture
def model_partseg() -> DGCNNPartSegmentation:
    return DGCNNPartSegmentation(
        in_channels=3,
        num_classes=10,
        num_categories=4,
        cat_embed_channels=16,
        spatial_dim=3,
        stnet_edge_channels=None,
        stnet_local_channels=None,
        stnet_global_channels=None,
        stnet_num_neighbors=20,
        proj_channels=64,
        channels=[32, 64],
        head_channels=[32],
        num_neighbors=8,
        act="relu",
        act_kwargs=None,
        act_first=False,
        norm="batch_norm",
        norm_kwargs=None,
        bias=True,
        dropout=0.0,
    )


@pytest.fixture
def partseg_category(data: Dict[str, Tensor]) -> Tensor:
    num_batches = int(data["batch"].max()) + 1
    return torch.nn.functional.one_hot(torch.arange(num_batches) % 4, num_classes=4).float()


def test_dgcnn_part_segmentation_forward(
    model_partseg: DGCNNPartSegmentation,
    data: Dict[str, Tensor],
    partseg_category: Tensor,
) -> None:
    logits = model_partseg(data["x"], data["pos"], data["batch"], partseg_category)
    assert logits.shape == (data["pos"].shape[0], model_partseg.num_classes)
    assert logits.dtype == data["x"].dtype


def test_dgcnn_part_segmentation_reset_classifier(
    model_partseg: DGCNNPartSegmentation,
    data: Dict[str, Tensor],
    partseg_category: Tensor,
) -> None:
    model_partseg.reset_classifier(num_classes=42)
    logits = model_partseg(data["x"], data["pos"], data["batch"], partseg_category)
    assert logits.shape == (data["pos"].shape[0], 42)


def test_dgcnn_part_segmentation_forward_features_and_head(
    model_partseg: DGCNNPartSegmentation,
    data: Dict[str, Tensor],
    partseg_category: Tensor,
) -> None:
    x, _, batch = model_partseg.forward_features(data["x"], data["pos"], data["batch"], partseg_category)
    assert x.shape[0] == data["pos"].shape[0]
    logits = model_partseg.forward_head(x)
    assert logits.shape == (data["pos"].shape[0], model_partseg.num_classes)


def test_dgcnn_part_segmentation_forward_head_pre_logits(
    model_partseg: DGCNNPartSegmentation,
    data: Dict[str, Tensor],
    partseg_category: Tensor,
) -> None:
    x, _, batch = model_partseg.forward_features(data["x"], data["pos"], data["batch"], partseg_category)
    feats = model_partseg.forward_head(x, pre_logits=True)
    assert torch.equal(feats, x)
    assert feats.shape[1] == model_partseg.num_features


def test_dgcnn_reset_classifier_keeps_current_pooling(model_clf: DGCNNClassification) -> None:
    model_clf.reset_classifier(10, global_pool=("max", "mean"))
    pool = model_clf.global_pool
    model_clf.reset_classifier(5)
    assert model_clf.global_pool is pool
    assert type(pool).__name__ == "CatPool"


def test_dgcnn_segmentation_stnet_has_no_head_dropout() -> None:
    model = DGCNNSegmentation(
        in_channels=3,
        num_classes=10,
        stnet_local_channels=[16],
        stnet_global_channels=[16],
        proj_channels=64,
        channels=[32, 64],
        head_channels=[32],
        num_neighbors=8,
        dropout=0.5,
    )
    assert model.stnet is not None
    assert model.stnet.local_nn.dropout == [0.0]
    assert model.stnet.global_nn.dropout == [0.0]


@pytest.mark.parametrize(
    "name",
    [
        pytest.param("dgcnn.s3dis-area1.an-tao", id="area1"),
        pytest.param("dgcnn.s3dis-area2.an-tao", id="area2"),
        pytest.param("dgcnn.s3dis-area3.an-tao", id="area3"),
        pytest.param("dgcnn.s3dis-area4.an-tao", id="area4"),
        pytest.param("dgcnn.s3dis-area5.an-tao", id="area5"),
        pytest.param("dgcnn.s3dis-area6.an-tao", id="area6"),
    ],
)
def test_dgcnn_s3dis_entries_feed_room_normalized_positions(name: str) -> None:
    torch.manual_seed(0)
    info = model_info(name)
    sample = {
        DataKeys.POS: torch.rand(64, 3),
        DataKeys.COLOR: torch.rand(64, 3),
        DataKeys.NORM_POS: torch.rand(64, 3),
    }
    transform = info["transform"]
    assert transform is not None
    batch = collate([transform(sample)])
    x, norm_pos, _ = select_inputs(batch, info["input_keys"])
    assert torch.equal(x, torch.cat([sample[DataKeys.POS], sample[DataKeys.COLOR]], dim=1))
    assert torch.equal(norm_pos, sample[DataKeys.NORM_POS])


def test_dgcnn_scannet_entry_feeds_block_centered_positions_and_scene_normalized_positions() -> None:
    torch.manual_seed(0)
    info = model_info("dgcnn.scannet20.an-tao")
    pos = torch.rand(64, 3) * 4.0
    color = torch.randint(0, 256, (64, 3)).float()
    scene_max = torch.tensor([8.0, 6.0, 3.0])
    block_center = torch.tensor([2.0, 2.0, 1.5])
    sample = {
        DataKeys.POS: pos.clone(),
        DataKeys.COLOR: color.clone(),
        DataKeys.SEGMENT: torch.randint(0, 21, (64,)),
        DataKeys.SCENE_MAX: scene_max,
        DataKeys.BLOCK_CENTER: block_center,
    }
    transform = info["transform"]
    assert transform is not None
    batch = collate([transform(sample)])
    x, norm_pos, _ = select_inputs(batch, info["input_keys"])
    centered = pos - torch.tensor([2.0, 2.0, 0.0])
    assert torch.allclose(x, torch.cat([centered, color / 255], dim=1))
    assert torch.allclose(norm_pos, pos / scene_max)
