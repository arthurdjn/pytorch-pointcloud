import torch

from torch_pointcloud.layers.anchors import (
    AnchorHeadMulti,
    AnchorHeadMultiOutput,
    AnchorHeadSingle,
    assign_anchor_targets,
)
from torch_pointcloud.utils.box3d import decode_box_residuals, encode_box_residuals

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


def test_assign_single_overlapping_gt() -> None:
    anchors = torch.tensor(
        [
            [0.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0],
            [50.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0],
        ]
    )
    gt_boxes = torch.tensor([[0.3, 0.1, 0.0, 4.0, 2.0, 1.5, 0.0]])
    gt_labels = torch.tensor([1])

    out = assign_anchor_targets(anchors, gt_boxes, gt_labels, matched_threshold=0.6, unmatched_threshold=0.45)

    assert out["cls_labels"].tolist() == [1, 0]

    decoded = decode_box_residuals(out["box_reg_targets"][:1], anchors[:1])
    assert torch.allclose(decoded, gt_boxes, atol=1e-5)
    assert torch.count_nonzero(out["box_reg_targets"][1]) == 0


def test_assign_no_gt_all_background() -> None:
    anchors = torch.rand(16, 7)
    anchors[:, 3:6] += 0.5
    out = assign_anchor_targets(
        anchors,
        torch.zeros(0, 7),
        torch.zeros(0, dtype=torch.long),
        matched_threshold=0.6,
        unmatched_threshold=0.45,
    )

    assert (out["cls_labels"] == 0).all()
    assert (out["box_reg_targets"] == 0.0).all()


def test_assign_force_match_below_threshold() -> None:
    anchors = torch.tensor(
        [
            [0.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0],
            [8.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0],
        ]
    )
    gt_boxes = torch.tensor([[2.6, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0]])
    gt_labels = torch.tensor([1])

    out = assign_anchor_targets(anchors, gt_boxes, gt_labels, matched_threshold=0.6, unmatched_threshold=0.45)

    assert out["cls_labels"][0].item() == 1
    assert (out["cls_labels"] > 0).sum().item() == 1


def _make_multi_head() -> AnchorHeadMulti:
    return AnchorHeadMulti(
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


def test_anchor_head_single_decode_has_no_velocity() -> None:
    """The 7-DoF single head (no velocity in the box code) decodes without a `velocity` key."""
    torch.manual_seed(0)
    head = AnchorHeadSingle(
        8, 1, (4, 4), RANGE, anchor_sizes=[[4.0, 2.0, 1.5]], anchor_bottom_heights=[0.0], feature_map_stride=1
    ).eval()
    with torch.no_grad():
        out = head(torch.randn(2, 8, 4, 4))
    det = head.decode(out)
    assert "velocity" not in det
    assert det["boxes"].shape == (2 * head.anchors.shape[0], 7)
