import torch

from torch_pointcloud.layers.anchors import assign_anchor_targets
from torch_pointcloud.utils.box3d import decode_box_residuals, encode_box_residuals


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
    assert out["reg_weights"].tolist() == [1.0, 0.0]
    assert out["cls_weights"].tolist() == [1.0, 1.0]

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
    assert (out["reg_weights"] == 0.0).all()
    assert (out["cls_weights"] == 1.0).all()
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
