import pytest
import torch

from torch_pointcloud.layers.anchors import (
    AnchorHead,
    AnchorHeadMultiOutput,
    MultiGroupAnchorHead,
    MultiLevelAnchorHead,
)
from torch_pointcloud.ops.box3d import decode_box_residuals, encode_box_residuals

RANGE = (0.0, -4.0, -1.0, 8.0, 4.0, 1.0)


def test_encode_decode_round_trip() -> None:
    torch.manual_seed(0)
    anchors = torch.rand(32, 7)
    anchors[:, 3:6] += 0.5
    boxes = anchors.clone()
    boxes[:, :3] += torch.randn(32, 3) * 0.2
    boxes[:, 3:6] *= 0.8 + 0.4 * torch.rand(32, 3)
    boxes[:, 6] += torch.randn(32) * 0.1

    encoded = encode_box_residuals(boxes, anchors)
    decoded = decode_box_residuals(encoded, anchors)
    assert torch.allclose(decoded, boxes, atol=1e-5)


def _make_multi_head() -> MultiGroupAnchorHead:
    return MultiGroupAnchorHead(
        8,
        2,
        (4, 4),
        RANGE,
        anchor_sizes=[[4.0, 2.0, 1.5], [2.0, 1.0, 1.0]],
        anchor_bottom_heights=[0.0, 0.0],
        head_class_groups=[[0], [1]],
        feature_map_stride=1,
        shared_conv_num_filter=8,
        num_middle_filter=8,
    )


def test_anchor_head_multi_decode_returns_velocity() -> None:
    """`decode` keeps the decoded $(v_x, v_y)$ columns of the multihead box code under `velocity`."""
    torch.manual_seed(0)
    head = _make_multi_head()
    batch_box = torch.randn(2, 64, 9)
    batch_box[..., 7] = 1.5
    batch_box[..., 8] = -0.5
    out: AnchorHeadMultiOutput = {
        "cls": [torch.randn(2, 32, 1), torch.randn(2, 32, 1)],
        "box": [torch.zeros(2, 32, 10), torch.zeros(2, 32, 10)],
        "batch_box": batch_box,
        "multihead_label_mapping": [torch.tensor([1]), torch.tensor([2])],
    }
    det = head.decode(out)
    assert det["boxes"].shape == (128, 7)
    assert det["velocity"].shape == (128, 2)
    assert torch.all(det["velocity"][:, 0] == 1.5)
    assert torch.all(det["velocity"][:, 1] == -0.5)


def test_anchor_head_multi_forward_decode_velocity_matches_batch_box() -> None:
    torch.manual_seed(0)
    head = _make_multi_head().eval()
    with torch.no_grad():
        out = head(torch.randn(2, 8, 4, 4))
    det = head.decode(out)
    assert out["batch_box"].shape == (2, 64, 9)
    assert torch.equal(det["velocity"], out["batch_box"][..., 7:9].reshape(-1, 2))


def test_anchor_head_multi_starts_at_the_foreground_prior() -> None:
    """Every group head's class logits start at the focal-loss prior, so a fresh model scores 1% everywhere."""
    head = _make_multi_head().eval()
    with torch.no_grad():
        out = head(torch.zeros(1, 8, 4, 4))
    for cls_preds in out["cls"]:
        assert torch.allclose(cls_preds.sigmoid(), torch.full_like(cls_preds, 0.01), atol=1e-6)


def test_anchor_head_single_decode_has_no_velocity() -> None:
    """The 7-DoF single head (no velocity in the box code) decodes without a `velocity` key."""
    torch.manual_seed(0)
    head = AnchorHead(
        8,
        1,
        (4, 4),
        RANGE,
        anchor_sizes=[[4.0, 2.0, 1.5]],
        anchor_bottom_heights=[0.0],
        feature_map_stride=1,
        dir_offset=0.78539,
    ).eval()
    with torch.no_grad():
        out = head(torch.randn(2, 8, 4, 4))
    det = head.decode(out)
    assert "velocity" not in det
    assert det["boxes"].shape == (2 * head.anchors.shape[0], 7)


def test_anchor_head_shares_templates_across_classes_and_decodes_velocity() -> None:
    torch.manual_seed(0)
    head = AnchorHead(
        8,
        3,
        (4, 4),
        RANGE,
        anchor_sizes=[[4.0, 2.0, 1.5], [1.0, 1.0, 1.0]],
        anchor_bottom_heights=[0.0, -0.5],
        feature_map_stride=1,
        dir_offset=0.78539,
        anchor_range=(1.0, -2.0, 0.0, 7.0, 2.0, 0.0),
        align_center=True,
        code_size=9,
    ).eval()
    assert head.anchors.shape == (4 * 4 * 2 * 2, 9)
    assert head.conv_cls.out_channels == 4 * 3 and head.conv_box.out_channels == 4 * 9
    # The anchor grid spans `anchor_range`, not the point-cloud range; the velocity slots start at zero.
    assert torch.allclose(head.anchors[:, 0].unique(), torch.tensor([1.75, 3.25, 4.75, 6.25]))
    assert torch.all(head.anchors[:, 7:] == 0)
    with torch.no_grad():
        out = head(torch.randn(2, 8, 4, 4))
    det = head.decode(out)
    assert out["batch_cls"].shape == (2, 64, 3) and out["batch_box"].shape == (2, 64, 9)
    assert torch.equal(det["velocity"], out["batch_box"][..., 7:9].reshape(-1, 2))
    assert torch.equal(det["class_probs"], out["batch_cls"].sigmoid().reshape(-1, 3))
    assert torch.equal(det["scores"], det["class_probs"].max(dim=1).values)


def test_anchor_head_without_dir_offset_warns_and_keeps_the_former_default() -> None:
    with pytest.warns(FutureWarning, match="dir_offset"):
        head = AnchorHead(
            8, 1, (4, 4), RANGE, anchor_sizes=[[4.0, 2.0, 1.5]], anchor_bottom_heights=[0.0], feature_map_stride=1
        )
    assert head.dir_offset == 0.78539


def test_anchor_head_needs_one_bottom_height_per_size() -> None:
    with pytest.raises(ValueError, match="one anchor bottom height per anchor size"):
        AnchorHead(
            8,
            1,
            (4, 4),
            RANGE,
            anchor_sizes=[[4.0, 2.0, 1.5]],
            anchor_bottom_heights=[0.0, 1.0],
            feature_map_stride=1,
            dir_offset=0.78539,
        )


def _make_multi_level_head(strides: list, scales: list) -> MultiLevelAnchorHead:
    return MultiLevelAnchorHead(
        8,
        3,
        (8, 8),
        RANGE,
        anchor_sizes=[[4.0, 2.0, 1.5], [1.0, 1.0, 1.0]],
        anchor_bottom_heights=[0.0, 0.0],
        feature_map_strides=strides,
        anchor_scales=scales,
        dir_offset=0.78539,
        code_size=9,
    )


def test_multi_level_anchor_head_scales_the_templates_of_every_level() -> None:
    head = _make_multi_level_head([1, 2], [1.0, 2.0])
    fine, coarse = 8 * 8 * 4, 4 * 4 * 4
    assert head.anchors.shape == (fine + coarse, 9)
    assert head.anchors[:fine, 3:6].unique(dim=0).tolist() == [[1.0, 1.0, 1.0], [4.0, 2.0, 1.5]]
    assert head.anchors[fine:, 3:6].unique(dim=0).tolist() == [[2.0, 2.0, 2.0], [8.0, 4.0, 3.0]]

    torch.manual_seed(0)
    with torch.no_grad():
        out = head([torch.randn(2, 8, 8, 8), torch.randn(2, 8, 4, 4)])
    assert [tuple(c.shape) for c in out["cls"]] == [(2, 8, 8, 12), (2, 4, 4, 12)]
    assert out["batch_cls"].shape == (2, fine + coarse, 3)
    assert out["batch_box"].shape == (2, fine + coarse, 9)
    det = head.decode(out)
    assert det["boxes"].shape == (2 * (fine + coarse), 7)
    assert torch.equal(det["class_probs"], out["batch_cls"].sigmoid().reshape(-1, 3))
    # Scene by scene, finest level first.
    level = torch.cat([torch.zeros(fine, dtype=torch.long), torch.ones(coarse, dtype=torch.long)])
    assert torch.equal(det["level"], level.repeat(2))
    assert torch.equal(det["batch"], torch.arange(2).repeat_interleave(fine + coarse))


@pytest.mark.parametrize(
    "shapes,match",
    [
        pytest.param([(8, 8)], "one feature map per level", id="missing-level"),
        pytest.param([(8, 8), (5, 5)], "level 1 to be 4 x 4", id="wrong-size"),
    ],
)
def test_multi_level_anchor_head_rejects_maps_that_do_not_match_its_levels(shapes: list, match: str) -> None:
    head = _make_multi_level_head([1, 2], [1.0, 2.0])
    with pytest.raises(ValueError, match=match):
        head([torch.randn(2, 8, *shape) for shape in shapes])


def test_multi_level_anchor_head_with_one_level_matches_the_anchor_head() -> None:
    torch.manual_seed(0)
    multi = _make_multi_level_head([2], [1.0]).eval()
    single = AnchorHead(
        8,
        3,
        (8, 8),
        RANGE,
        anchor_sizes=[[4.0, 2.0, 1.5], [1.0, 1.0, 1.0]],
        anchor_bottom_heights=[0.0, 0.0],
        feature_map_stride=2,
        dir_offset=0.78539,
        code_size=9,
    ).eval()
    single.load_state_dict(multi.state_dict())
    features = torch.randn(2, 8, 4, 4)
    with torch.no_grad():
        out_multi, out_single = multi([features]), single(features)
    assert torch.equal(multi.anchors, single.anchors)
    assert torch.equal(out_multi["batch_cls"], out_single["batch_cls"])
    assert torch.allclose(out_multi["batch_box"], out_single["batch_box"])


def test_multi_level_anchor_head_needs_one_scale_per_level() -> None:
    with pytest.raises(ValueError, match="one anchor scale per level"):
        _make_multi_level_head([1, 2], [1.0])
