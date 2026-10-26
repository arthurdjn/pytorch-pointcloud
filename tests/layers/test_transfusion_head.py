from typing import Optional

import pytest
import torch
import torch.nn as nn
from torch import Tensor

from torch_pointcloud.layers.transfusion_head import SeparateHeadTransfusion, TransFusionHead, TransFusionHeadOutput

RANGE = (-54.0, -54.0, -5.0, 54.0, 54.0, 3.0)


def _output(num_classes: int, num_proposals: int, batch_size: int = 2, with_iou: bool = True) -> TransFusionHeadOutput:
    torch.manual_seed(0)
    out: TransFusionHeadOutput = {
        "center": torch.rand(batch_size, 2, num_proposals),
        "height": torch.rand(batch_size, 1, num_proposals),
        "dim": torch.rand(batch_size, 3, num_proposals),
        "rot": torch.randn(batch_size, 2, num_proposals),
        "vel": torch.randn(batch_size, 2, num_proposals),
        "heatmap": torch.randn(batch_size, num_classes, num_proposals),
        "query_heatmap_score": torch.rand(batch_size, num_classes, num_proposals),
        "query_labels": torch.randint(0, num_classes, (batch_size, num_proposals)),
        "dense_heatmap": torch.randn(batch_size, num_classes, 180, 180),
    }
    if with_iou:
        out["iou"] = torch.rand(batch_size, 1, num_proposals)
    return out


class DummyDecoder(nn.Module):
    def forward(
        self,
        query: Tensor,
        key: Tensor,
        query_pos: Tensor,
        key_pos: Tensor,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        self.query_pos = query_pos
        self.key_pos = key_pos
        self.key_padding_mask = key_padding_mask
        return query


def test_separate_head_iou_branch_is_optional() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 8, 5)
    out = SeparateHeadTransfusion(8, 4, num_classes=3)(x)
    assert set(out) == {"center", "height", "dim", "rot", "vel", "heatmap"}
    assert out["center"].shape == (2, 2, 5)
    assert out["heatmap"].shape == (2, 3, 5)
    with_iou = SeparateHeadTransfusion(8, 4, num_classes=3, use_iou=True)(x)
    assert with_iou["iou"].shape == (2, 1, 5)


def test_separate_head_bias_before_norm() -> None:
    head = SeparateHeadTransfusion(8, 4, num_classes=3)
    assert head.center.lins[0].bias is None and head.center.lins[1].bias is not None
    assert SeparateHeadTransfusion(8, 4, num_classes=3, bias=True).center.lins[0].bias is not None


def test_head_decode_packs_detections() -> None:
    """`TransFusionHead.decode` turns raw predictions into PyG-packed detections without needing CUDA."""
    head = TransFusionHead(
        384, 10, (360, 360, 32), RANGE, (0.3, 0.3, 0.25), feature_map_stride=8, use_iou=True, iou_rectifier=0.5
    ).eval()
    det = head.decode(_output(10, head.num_proposals))
    assert det["boxes"].shape[1] == 7
    assert det["scores"].shape[0] == det["boxes"].shape[0] == det["labels"].shape[0] == det["batch"].shape[0]
    assert torch.isfinite(det["boxes"]).all()
    assert set(det["batch"].tolist()) <= {0, 1}


def test_head_decode_rectifies_scores_only_with_the_iou_branch() -> None:
    out = _output(3, 20)
    plain = TransFusionHead(16, 3, (40, 40, 32), RANGE, (0.3, 0.3, 0.25), feature_map_stride=8, num_proposals=20).eval()
    rectified = TransFusionHead(
        16,
        3,
        (40, 40, 32),
        RANGE,
        (0.3, 0.3, 0.25),
        feature_map_stride=8,
        num_proposals=20,
        use_iou=True,
        iou_rectifier=0.5,
    ).eval()
    scores = plain.decode(out)["scores"]
    one_hot = torch.nn.functional.one_hot(out["query_labels"], 3).permute(0, 2, 1)
    expected = (out["heatmap"].sigmoid() * out["query_heatmap_score"] * one_hot).max(1).values.reshape(-1)
    assert torch.allclose(scores, expected)
    iou = ((out["iou"][:, 0] + 1) * 0.5).clamp(0, 1).reshape(-1)
    assert torch.allclose(rectified.decode(out)["scores"], scores.sqrt() * iou.sqrt())


def test_head_rejects_a_rectifier_without_the_iou_branch() -> None:
    with pytest.raises(ValueError, match="use_iou"):
        TransFusionHead(16, 3, (40, 40, 32), RANGE, (0.3, 0.3, 0.25), feature_map_stride=8, iou_rectifier=0.5)


def test_head_decoder_starts_xavier_uniform() -> None:
    """A square decoder matrix reaches beyond the default Kaiming bound, which Xavier's wider one allows."""
    torch.manual_seed(0)
    head = TransFusionHead(16, 3, (40, 40, 32), RANGE, (0.3, 0.3, 0.25), feature_map_stride=8, num_proposals=20)
    square = [p for p in head.decoder.parameters() if p.dim() == 2 and p.shape[0] == p.shape[1]]
    assert square
    for weight in square:
        fan_in = weight.shape[1]
        assert weight.abs().max() <= (6.0 / (2 * fan_in)) ** 0.5
    assert any(weight.abs().max() > 1.0 / weight.shape[1] ** 0.5 for weight in square)


def test_head_decode_returns_velocity() -> None:
    """Decoded candidates carry the head's predicted BEV velocity unchanged under `velocity`."""
    head = TransFusionHead(384, 10, (360, 360, 32), RANGE, (0.3, 0.3, 0.25), feature_map_stride=8).eval()
    out = _output(10, head.num_proposals, with_iou=False)
    out["vel"][:, 0] = 1.5
    out["vel"][:, 1] = -0.5
    det = head.decode(out)
    assert det["boxes"].shape[0] > 0
    assert det["velocity"].shape == (det["boxes"].shape[0], 2)
    assert torch.all(det["velocity"][:, 0] == 1.5)
    assert torch.all(det["velocity"][:, 1] == -0.5)


def test_head_forward_supports_non_nuscenes_num_classes() -> None:
    """The head must not assume the nuScenes 10-class layout (regression: hardcoded class indices 8/9)."""
    torch.manual_seed(0)
    head = TransFusionHead(
        16, 3, (40, 40, 32), RANGE, (0.3, 0.3, 0.25), feature_map_stride=8, num_proposals=20, query_radius=2
    ).eval()
    out = head(torch.randn(2, 16, 5, 5))
    assert int(out["query_labels"].max()) < 3
    assert "iou" not in out


def test_head_bev_positions_follow_the_row_major_map() -> None:
    """Flat index $i$ of a $(H, W)$ map is the cell $(x, y) = (i \\bmod W, i // W)$, so the positions are $(x, y)$."""
    head = TransFusionHead(16, 3, (32, 64, 1), RANGE, (0.3, 0.3, 0.25), feature_map_stride=8, num_proposals=10)
    width, height = 32 // 8, 64 // 8
    assert head.bev_pos.shape == (1, height * width, 2)
    index = torch.arange(height * width)
    assert torch.equal(head.bev_pos[0, :, 0], (index % width).float() + 0.5)
    assert torch.equal(head.bev_pos[0, :, 1], (index // width).float() + 0.5)


def test_head_local_window_keys_on_non_square_grid(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each query's (0, 0)-offset key is its own cell and the window cells outside the map are masked."""
    torch.manual_seed(0)
    radius = 2
    head = TransFusionHead(
        16, 3, (32, 64, 1), RANGE, (0.3, 0.3, 0.25), feature_map_stride=8, num_proposals=10, query_radius=radius
    ).eval()
    capture = DummyDecoder()
    monkeypatch.setattr(head, "decoder", capture)
    width, height = 32 // 8, 64 // 8

    with torch.no_grad():
        out = head(torch.randn(2, 16, height, width))

    window = 2 * radius + 1
    center = window * radius + radius
    assert capture.key_padding_mask is not None
    assert capture.key_padding_mask.shape == (2 * 10, window * window)
    assert not capture.key_padding_mask[:, center].any()
    assert torch.allclose(capture.key_pos[:, center, :], capture.query_pos[:, 0, :])
    # A window never reaches a cell outside the map: the masked keys are exactly the out-of-range offsets.
    offsets = torch.arange(-radius, radius + 1)
    offset_y, offset_x = torch.meshgrid(offsets, offsets, indexing="ij")
    key_x = capture.query_pos[:, 0, 0:1] + offset_x.reshape(1, -1)
    key_y = capture.query_pos[:, 0, 1:2] + offset_y.reshape(1, -1)
    expected = (key_x < 0) | (key_x > width) | (key_y < 0) | (key_y > height)
    assert torch.equal(capture.key_padding_mask, expected)
    assert torch.isfinite(out["heatmap"]).all()


def test_head_local_max_classes_skip_the_peak_suppression() -> None:
    """A `local_max_classes` entry keeps every cell of its heatmap as a candidate, the others keep local maxima only."""
    torch.manual_seed(0)
    head = TransFusionHead(16, 2, (32, 32, 1), RANGE, (0.3, 0.3, 0.25), feature_map_stride=8, num_proposals=16).eval()
    crowded = TransFusionHead(
        16, 2, (32, 32, 1), RANGE, (0.3, 0.3, 0.25), feature_map_stride=8, num_proposals=16, local_max_classes=(1,)
    ).eval()
    crowded.load_state_dict(head.state_dict())
    x = torch.randn(1, 16, 4, 4)
    with torch.no_grad():
        suppressed = (head(x)["query_heatmap_score"].max(1).values > 0).sum()
        kept = (crowded(x)["query_heatmap_score"].max(1).values > 0).sum()
    assert kept > suppressed


def test_head_local_max_classes_decode() -> None:
    """`decode` is class-count agnostic: a 3-class head with `local_max_classes` set still packs raw detections."""
    head = TransFusionHead(
        384, 3, (360, 360, 32), RANGE, (0.3, 0.3, 0.25), feature_map_stride=8, local_max_classes=(1, 2)
    ).eval()
    assert head.local_max_classes == (1, 2)
    det = head.decode(_output(3, head.num_proposals, with_iou=False))
    assert det["boxes"].shape[1] == 7
    assert torch.isfinite(det["boxes"]).all()


def test_head_decode_height_code() -> None:
    """The `top` code is the box top face: the decoded gravity center sits half a box height below it."""
    out = _output(3, 20, with_iou=False)
    center = TransFusionHead(
        16, 3, (40, 40, 32), RANGE, (0.3, 0.3, 0.25), feature_map_stride=8, num_proposals=20
    ).eval()
    top = TransFusionHead(
        16, 3, (40, 40, 32), RANGE, (0.3, 0.3, 0.25), feature_map_stride=8, num_proposals=20, height_code="top"
    ).eval()
    boxes_center, boxes_top = center.decode(out)["boxes"], top.decode(out)["boxes"]
    assert torch.allclose(boxes_center[:, 2], out["height"][:, 0].reshape(-1))
    assert torch.allclose(boxes_top[:, 2], boxes_center[:, 2] - boxes_center[:, 5] / 2)


def test_head_decode_post_center_range() -> None:
    out = _output(3, 20, with_iou=False)
    out["center"][0, 0, :5] = 1000.0
    kept_all = TransFusionHead(
        16, 3, (40, 40, 32), RANGE, (0.3, 0.3, 0.25), feature_map_stride=8, num_proposals=20
    ).eval()
    filtered = TransFusionHead(
        16,
        3,
        (40, 40, 32),
        RANGE,
        (0.3, 0.3, 0.25),
        feature_map_stride=8,
        num_proposals=20,
        post_center_range=(-60, -60, -10, 60, 60, 10),
    ).eval()
    assert kept_all.decode(out)["boxes"].shape[0] == 2 * 20
    det = filtered.decode(out)
    assert det["boxes"].shape[0] == 2 * 20 - 5
    assert (det["boxes"][:, 0] <= 60).all()
