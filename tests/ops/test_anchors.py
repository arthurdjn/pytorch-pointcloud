import pytest
import torch

from torch_pointcloud.ops.anchors import assign_anchor_targets, generate_anchors
from torch_pointcloud.ops.box3d import decode_box_residuals

RANGE = (0.0, -4.0, -1.0, 8.0, 4.0, 1.0)


def test_generate_anchors_layout_and_center_lift() -> None:
    anchors = generate_anchors(RANGE, (4, 4), [[4.0, 2.0, 1.5]], [0.0, 1.57], [0.0])
    assert anchors.shape == (1, 4, 4, 1, 2, 7)
    # The grid spans the range with inclusive endpoints, and bottom heights are lifted by half the box height.
    assert torch.allclose(anchors[0, 0, :, 0, 0, 0], torch.tensor([0.0, 8.0 / 3, 16.0 / 3, 8.0]))
    assert torch.allclose(anchors[0, :, 0, 0, 0, 1], torch.tensor([-4.0, -4.0 / 3, 4.0 / 3, 4.0]))
    assert torch.all(anchors[..., 2] == 0.75)
    assert torch.all(anchors[..., 0, 6] == 0.0) and torch.all(anchors[..., 1, 6] == 1.57)


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


def test_assign_batched_matches_per_scene_assignment() -> None:
    """`gt_batch` / `batch_size` match every scene at once, exactly as the per-scene calls do."""
    torch.manual_seed(0)
    anchors = generate_anchors((0.0, -8.0, -2.0, 16.0, 8.0, 2.0), (8, 8), [[3.0, 1.5, 1.5]], [0.0, 1.57], [-1.0]).view(
        -1, 7
    )
    gt_boxes = torch.cat(
        [
            torch.rand(9, 2) * torch.tensor([16.0, 16.0]) + torch.tensor([0.0, -8.0]),
            torch.rand(9, 1) - 1.0,
            torch.rand(9, 3) * 2 + 0.8,
            (torch.rand(9, 1) - 0.5) * 6,
        ],
        dim=1,
    )
    gt_labels = torch.randint(1, 3, (9,))
    gt_batch = torch.tensor([0, 2, 2, 0, 1, 2, 0, 2, 1])  # scenes of different sizes, out of order

    batched = assign_anchor_targets(
        anchors, gt_boxes, gt_labels, matched_threshold=0.5, unmatched_threshold=0.3, gt_batch=gt_batch, batch_size=4
    )
    assert batched["cls_labels"].shape == (4, anchors.shape[0]) and batched["box_reg_targets"].shape == (
        4,
        anchors.shape[0],
        7,
    )
    for scene in range(4):
        keep = gt_batch == scene
        single = assign_anchor_targets(
            anchors, gt_boxes[keep], gt_labels[keep], matched_threshold=0.5, unmatched_threshold=0.3
        )
        assert torch.equal(batched["cls_labels"][scene], single["cls_labels"])
        assert torch.equal(batched["box_reg_targets"][scene], single["box_reg_targets"])
    assert torch.all(batched["cls_labels"][3] == 0)  # the scene without boxes is all background
    with pytest.raises(ValueError, match="go together"):
        assign_anchor_targets(
            anchors, gt_boxes, gt_labels, matched_threshold=0.5, unmatched_threshold=0.3, gt_batch=gt_batch
        )
