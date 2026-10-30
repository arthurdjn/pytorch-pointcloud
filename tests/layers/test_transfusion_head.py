from typing import Any, Dict, Optional, Tuple

import pytest
import torch
import torch.nn as nn
from torch import Tensor

from torch_pointcloud.layers.transfusion_head import (
    TransFusionHead,
    TransFusionHeadOutput,
    TransFusionPredictionHead,
)

GEOMETRY: Dict[str, Any] = dict(
    voxel_size=(0.3, 0.3, 0.25),
    point_cloud_range=(-54.0, -54.0, -5.0, 54.0, 54.0, 3.0),
    feature_map_stride=8,
)


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
        "dense_heatmap": torch.randn(batch_size, num_classes, 8, 8),
    }
    if with_iou:
        out["iou"] = torch.rand(batch_size, 1, num_proposals) * 2 - 1
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


def test_prediction_head_iou_branch_is_optional() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 8, 5)
    out = TransFusionPredictionHead(8, 4, 3)(x)
    assert set(out) == {"center", "height", "dim", "rot", "vel", "heatmap"}
    assert out["center"].shape == (2, 2, 5)
    assert out["heatmap"].shape == (2, 3, 5)
    with_iou = TransFusionPredictionHead(8, 4, 3, use_iou=True)(x)
    assert with_iou["iou"].shape == (2, 1, 5)


def test_prediction_head_bias_before_norm() -> None:
    head = TransFusionPredictionHead(8, 4, 3)
    assert head.center.lins[0].bias is None and head.center.lins[1].bias is not None
    assert TransFusionPredictionHead(8, 4, 3, bias=True).center.lins[0].bias is not None


def test_head_decode_packs_detections() -> None:
    head = TransFusionHead(384, 10, height_code="top", use_iou=True, **GEOMETRY).eval()
    det = head.decode(_output(10, head.num_proposals), iou_rectifier=(0.5,) * 10)
    assert det["boxes"].shape == (2 * head.num_proposals, 7)
    assert det["scores"].shape[0] == det["labels"].shape[0] == det["batch"].shape[0] == det["boxes"].shape[0]
    assert torch.isfinite(det["boxes"]).all()
    assert torch.equal(det["batch"], torch.arange(2).repeat_interleave(head.num_proposals))


def test_head_decode_rectifies_scores_per_class() -> None:
    """Each candidate is rescored as $s^{1 - r_c} \\cdot \\text{iou}^{r_c}$ with the exponent of its class."""
    out = _output(3, 20)
    head = TransFusionHead(16, 3, height_code="center", num_proposals=20, use_iou=True, **GEOMETRY).eval()
    scores = head.decode(out)["scores"]
    one_hot = torch.nn.functional.one_hot(out["query_labels"], 3).permute(0, 2, 1)
    expected = (out["heatmap"].sigmoid() * out["query_heatmap_score"] * one_hot).max(1).values.reshape(-1)
    assert torch.allclose(scores, expected)

    det = head.decode(out, iou_rectifier=(0.0, 1.0, 0.5))
    iou = ((out["iou"][:, 0] + 1) * 0.5).clamp(0, 1).reshape(-1)
    labels = det["labels"]
    assert torch.allclose(det["scores"][labels == 0], scores[labels == 0])
    assert torch.allclose(det["scores"][labels == 1], iou[labels == 1])
    assert torch.allclose(det["scores"][labels == 2], (scores * iou).sqrt()[labels == 2])


@pytest.mark.parametrize(
    "use_iou, iou_rectifier, match",
    [
        pytest.param(True, (0.5, 0.5), "one entry per class", id="length"),
        pytest.param(False, (0.5, 0.5, 0.5), "use_iou", id="no-iou-branch"),
    ],
)
def test_head_decode_rejects_a_bad_rectifier(use_iou: bool, iou_rectifier: Tuple[float, ...], match: str) -> None:
    head = TransFusionHead(16, 3, height_code="center", num_proposals=20, use_iou=use_iou, **GEOMETRY).eval()
    with pytest.raises(ValueError, match=match):
        head.decode(_output(3, 20, with_iou=use_iou), iou_rectifier=iou_rectifier)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        pytest.param(dict(nms_kernel_size=2), "nms_kernel_size", id="even-kernel"),
        pytest.param(dict(local_max_classes=(3,)), "local_max_classes", id="class-above-range"),
        pytest.param(dict(local_max_classes=(-1,)), "local_max_classes", id="negative-class"),
        pytest.param(dict(height_code="bottom"), "height_code", id="height-code"),
    ],
)
def test_head_rejects_invalid_arguments(kwargs: Dict[str, Any], match: str) -> None:
    arguments: Dict[str, Any] = dict(height_code="center", **GEOMETRY)
    arguments.update(kwargs)
    with pytest.raises(ValueError, match=match):
        TransFusionHead(16, 3, **arguments)


def test_head_rejects_a_map_smaller_than_the_kernel() -> None:
    head = TransFusionHead(16, 3, height_code="center", num_proposals=4, **GEOMETRY).eval()
    with pytest.raises(ValueError, match="nms_kernel_size"):
        head(torch.randn(1, 16, 2, 8))


def test_head_decoder_starts_xavier_uniform() -> None:
    """A square decoder matrix reaches beyond the default Kaiming bound, which Xavier's wider one allows."""
    torch.manual_seed(0)
    head = TransFusionHead(16, 3, height_code="center", num_proposals=20, **GEOMETRY)
    square = [p for p in head.decoder.parameters() if p.dim() == 2 and p.shape[0] == p.shape[1]]
    assert square
    for weight in square:
        fan_in = weight.shape[1]
        assert weight.abs().max() <= (6.0 / (2 * fan_in)) ** 0.5
    assert any(weight.abs().max() > 1.0 / weight.shape[1] ** 0.5 for weight in square)


def test_head_decode_returns_velocity() -> None:
    """Decoded candidates carry the head's predicted BEV velocity unchanged under `velocity`."""
    head = TransFusionHead(384, 10, height_code="center", **GEOMETRY).eval()
    out = _output(10, head.num_proposals, with_iou=False)
    out["vel"][:, 0] = 1.5
    out["vel"][:, 1] = -0.5
    det = head.decode(out)
    assert det["velocity"].shape == (det["boxes"].shape[0], 2)
    assert torch.all(det["velocity"][:, 0] == 1.5)
    assert torch.all(det["velocity"][:, 1] == -0.5)


def test_head_forward_with_three_classes() -> None:
    torch.manual_seed(0)
    head = TransFusionHead(16, 3, height_code="center", num_proposals=20, query_radius=2, **GEOMETRY).eval()
    out = head(torch.randn(2, 16, 5, 5))
    assert int(out["query_labels"].max()) < 3
    assert "iou" not in out


def test_head_bev_positions_follow_each_input_map(monkeypatch: pytest.MonkeyPatch) -> None:
    """One head fed two map sizes in turn: flat index $i$ of a $(H, W)$ map is the cell $(i \\bmod W, i // W)$."""
    head = TransFusionHead(16, 3, height_code="center", num_proposals=10, **GEOMETRY).eval()
    capture = DummyDecoder()
    monkeypatch.setattr(head, "decoder", capture)

    for height, width in ((7, 11), (12, 5)):
        with torch.no_grad():
            out = head(torch.randn(2, 16, height, width))

        index = torch.arange(height * width)
        assert capture.key_pos.shape == (2, height * width, 2)
        assert torch.equal(capture.key_pos[0, :, 0], (index % width).float() + 0.5)
        assert torch.equal(capture.key_pos[0, :, 1], (index // width).float() + 0.5)
        assert out["dense_heatmap"].shape == (2, 3, height, width)


def test_head_local_window_keys_on_non_square_map(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each query's (0, 0)-offset key is its own cell and the window cells outside the map are masked."""
    torch.manual_seed(0)
    radius = 2
    head = TransFusionHead(16, 3, height_code="center", num_proposals=10, query_radius=radius, **GEOMETRY).eval()
    capture = DummyDecoder()
    monkeypatch.setattr(head, "decoder", capture)
    height, width = 8, 4

    with torch.no_grad():
        out = head(torch.randn(2, 16, height, width))

    window = 2 * radius + 1
    center = window * radius + radius
    assert capture.key_padding_mask is not None
    assert capture.key_padding_mask.shape == (2 * 10, window * window)
    assert not capture.key_padding_mask[:, center].any()
    assert torch.allclose(capture.key_pos[:, center, :], capture.query_pos[:, 0, :])
    offsets = torch.arange(-radius, radius + 1)
    offset_y, offset_x = torch.meshgrid(offsets, offsets, indexing="ij")
    key_x = capture.query_pos[:, 0, 0:1] + offset_x.reshape(1, -1)
    key_y = capture.query_pos[:, 0, 1:2] + offset_y.reshape(1, -1)
    expected = (key_x < 0) | (key_x > width) | (key_y < 0) | (key_y > height)
    assert torch.equal(capture.key_padding_mask, expected)
    assert torch.isfinite(out["heatmap"]).all()


def test_head_window_over_the_whole_map_matches_whole_map_attention_for_one_query() -> None:
    """A window wider than the map holds every cell once, so a lone query attends exactly as in whole-map mode."""
    torch.manual_seed(0)
    whole = TransFusionHead(16, 3, height_code="center", num_proposals=1, dropout=0.0, **GEOMETRY).eval()
    window = TransFusionHead(
        16, 3, height_code="center", num_proposals=1, dropout=0.0, query_radius=12, **GEOMETRY
    ).eval()
    window.load_state_dict(whole.state_dict())
    x = torch.randn(2, 16, 7, 11)
    with torch.no_grad():
        expected, actual = whole(x), window(x)
    for key in ("center", "height", "dim", "rot", "heatmap"):
        assert torch.allclose(actual[key], expected[key], atol=1e-5)


def test_head_local_max_classes_skip_the_peak_suppression() -> None:
    """A `local_max_classes` entry keeps every cell of its heatmap as a candidate, the others keep local maxima only."""
    torch.manual_seed(0)
    head = TransFusionHead(16, 2, height_code="center", num_proposals=16, **GEOMETRY).eval()
    crowded = TransFusionHead(16, 2, height_code="center", num_proposals=16, local_max_classes=(1,), **GEOMETRY)
    crowded.eval().load_state_dict(head.state_dict())
    x = torch.randn(1, 16, 4, 4)
    with torch.no_grad():
        suppressed = (head(x)["query_heatmap_score"].max(1).values > 0).sum()
        kept = (crowded(x)["query_heatmap_score"].max(1).values > 0).sum()
    assert kept > suppressed


def test_head_decode_height_code() -> None:
    """The `top` code is the box top face: the decoded gravity center sits half a box height below it."""
    out = _output(3, 20, with_iou=False)
    center = TransFusionHead(16, 3, height_code="center", num_proposals=20, **GEOMETRY).eval()
    top = TransFusionHead(16, 3, height_code="top", num_proposals=20, **GEOMETRY).eval()
    boxes_center, boxes_top = center.decode(out)["boxes"], top.decode(out)["boxes"]
    assert torch.allclose(boxes_center[:, 2], out["height"][:, 0].reshape(-1))
    assert torch.allclose(boxes_top[:, 2], boxes_center[:, 2] - boxes_center[:, 5] / 2)


def test_head_decode_keeps_every_query() -> None:
    """`decode` returns every query, far boxes included: range filtering belongs to the evaluation."""
    out = _output(3, 20, with_iou=False)
    out["center"][0, 0, :5] = 1000.0
    det = TransFusionHead(16, 3, height_code="center", num_proposals=20, **GEOMETRY).eval().decode(out)
    assert det["boxes"].shape[0] == 2 * 20
    assert (det["boxes"][:5, 0] > 1000).all()
