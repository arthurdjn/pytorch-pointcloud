import math
from typing import Any, Dict, Tuple

import pytest
import torch
from torch import Tensor

from torch_pointcloud.losses import VoteNetLoss
from torch_pointcloud.ops.box3d import angle_to_class
from torch_pointcloud.utils.data import DataKeys


def _create_data(
    batch_size: int = 2,
    num_proposals: int = 8,
    num_seed: int = 16,
    num_heading_bins: int = 12,
    num_size_clusters: int = 10,
    num_classes: int = 10,
) -> Tuple[Dict[str, Tensor], Dict[str, Any]]:
    torch.manual_seed(0)
    batch_seed = torch.arange(batch_size).repeat_interleave(num_seed)
    output: Dict[str, Tensor] = {
        "objectness_scores": torch.randn(batch_size, num_proposals, 2),
        "center": torch.randn(batch_size, num_proposals, 3),
        "heading_scores": torch.randn(batch_size, num_proposals, num_heading_bins),
        "heading_residuals_normalized": torch.randn(batch_size, num_proposals, num_heading_bins),
        "size_scores": torch.randn(batch_size, num_proposals, num_size_clusters),
        "size_residuals_normalized": torch.randn(batch_size, num_proposals, num_size_clusters, 3),
        "sem_cls_scores": torch.randn(batch_size, num_proposals, num_classes),
        "pos_vote_aggr": torch.randn(batch_size, num_proposals, 3),
        "pos_seed": torch.rand(batch_size * num_seed, 3) * 4,
        "pos_vote": torch.rand(batch_size * num_seed, 3) * 4,
        "batch_seed": batch_seed,
        "batch_vote": batch_seed,
    }
    boxes = torch.tensor(
        [
            [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 0.3],
            [3.0, 2.0, 1.0, 1.5, 1.0, 1.0, -0.5],
            [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0],
        ]
    )
    data: Dict[str, Any] = {
        DataKeys.BOX: boxes,
        DataKeys.LABEL: torch.tensor([0, 3, 7]),
        DataKeys.BATCH_BOX: torch.tensor([0, 0, 1]),
    }
    return output, data


def _mean_size(num_size_clusters: int = 10) -> Tensor:
    torch.manual_seed(1)
    return torch.rand(num_size_clusters, 3) + 0.5


def test_votenet_loss_returns_dict_of_scalars() -> None:
    loss_fn = VoteNetLoss(num_heading_bins=12, num_size_clusters=10, num_classes=10, mean_sizes=_mean_size())
    output, data = _create_data()
    out = loss_fn(output, data)
    assert isinstance(out, dict)
    for key in ("loss", "vote_loss", "objectness_loss", "box_loss", "sem_cls_loss", "obj_acc"):
        assert key in out, key
        assert out[key].ndim == 0
        assert torch.isfinite(out[key])


def test_votenet_loss_backward() -> None:
    loss_fn = VoteNetLoss(num_heading_bins=12, num_size_clusters=10, num_classes=10, mean_sizes=_mean_size())
    output, data = _create_data()
    for value in output.values():
        if value.is_floating_point():
            value.requires_grad_(True)
    loss_fn(output, data)["loss"].backward()
    assert output["center"].grad is not None
    assert output["objectness_scores"].grad is not None
    assert output["pos_vote"].grad is not None


def test_votenet_loss_scannet_single_heading_bin() -> None:
    loss_fn = VoteNetLoss(num_heading_bins=1, num_size_clusters=18, num_classes=18, mean_sizes=_mean_size(18))
    output, data = _create_data(num_heading_bins=1, num_size_clusters=18, num_classes=18)
    assert torch.isfinite(loss_fn(output, data)["loss"])


def test_votenet_loss_no_boxes_is_finite() -> None:
    loss_fn = VoteNetLoss(num_heading_bins=12, num_size_clusters=10, num_classes=10, mean_sizes=_mean_size())
    output, data = _create_data()
    data[DataKeys.BOX] = data[DataKeys.BOX][:0]
    data[DataKeys.LABEL] = data[DataKeys.LABEL][:0]
    data[DataKeys.BATCH_BOX] = data[DataKeys.BATCH_BOX][:0]
    out = loss_fn(output, data)
    assert torch.isfinite(out["loss"])
    assert out["vote_loss"] == 0.0


def test_votenet_loss_ragged_seed_counts_are_supported() -> None:
    """Targets are assigned on the packed seeds, so scenes may carry different seed counts."""
    loss_fn = VoteNetLoss(num_heading_bins=12, num_size_clusters=10, num_classes=10, mean_sizes=_mean_size())
    output, data = _create_data()
    keep = torch.cat([torch.ones(16, dtype=torch.bool), torch.rand(16) > 0.5])
    for key in ("pos_seed", "pos_vote", "batch_seed", "batch_vote"):
        output[key] = output[key][keep]
    assert torch.isfinite(loss_fn(output, data)["loss"])


def test_votenet_loss_bad_mean_size_shape() -> None:
    with pytest.raises(ValueError, match="mean_sizes"):
        VoteNetLoss(num_heading_bins=12, num_size_clusters=10, num_classes=10, mean_sizes=torch.rand(5, 3))


def test_votenet_loss_empty_seeds_raises() -> None:
    loss_fn = VoteNetLoss(num_heading_bins=12, num_size_clusters=10, num_classes=10, mean_sizes=_mean_size())
    output, data = _create_data()
    for key in ("pos_seed", "pos_vote", "batch_seed", "batch_vote"):
        output[key] = output[key][:0]
    with pytest.raises(ValueError, match="non-empty"):
        loss_fn(output, data)


def test_votenet_vote_targets_overlapping_boxes_get_distinct_votes() -> None:
    """A seed inside two boxes votes for both centers; a seed inside one repeats it; outside seeds are masked."""
    loss_fn = VoteNetLoss(num_heading_bins=12, num_size_clusters=10, num_classes=10, mean_sizes=_mean_size())
    boxes = torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0], [0.25, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]])
    seeds = torch.tensor([[0.1, 0.0, 0.0], [-0.4, 0.0, 0.0], [5.0, 5.0, 5.0]])
    data: Dict[str, Any] = {DataKeys.BOX: boxes, DataKeys.BATCH_BOX: torch.zeros(2, dtype=torch.long)}
    votes, mask = loss_fn._vote_targets(seeds, torch.zeros(3, dtype=torch.long), data)
    assert votes.shape == (3, 3, 3)
    assert torch.allclose(votes[0, 0], boxes[0, 0:3])
    assert torch.allclose(votes[0, 1], boxes[1, 0:3])
    assert torch.allclose(votes[0, 2], votes[0, 0])
    assert torch.allclose(votes[1], boxes[0, 0:3].expand(3, 3))
    assert mask.tolist() == [1.0, 1.0, 0.0]


def test_votenet_vote_targets_use_oriented_counterclockwise_containment() -> None:
    loss_fn = VoteNetLoss(num_heading_bins=12, num_size_clusters=10, num_classes=10, mean_sizes=_mean_size())
    heading = math.pi / 4
    box = torch.tensor([[0.0, 0.0, 0.0, 2.0, 0.5, 1.0, heading]])
    cos, sin = math.cos(heading), math.sin(heading)
    local = torch.tensor([[0.9, 0.2, 0.0]])
    seed = torch.stack([local[:, 0] * cos - local[:, 1] * sin, local[:, 0] * sin + local[:, 1] * cos, local[:, 2]], 1)
    scene = torch.zeros(1, dtype=torch.long)
    _, mask = loss_fn._vote_targets(seed, scene, {DataKeys.BOX: box, DataKeys.BATCH_BOX: scene})
    assert mask.tolist() == [1.0]
    axis_aligned: Dict[str, Any] = {DataKeys.BOX: box * torch.tensor([1.0] * 6 + [0.0]), DataKeys.BATCH_BOX: scene}
    _, mask_axis = loss_fn._vote_targets(seed, scene, axis_aligned)
    assert mask_axis.tolist() == [0.0]


def _perfect_data(native_headings: Tensor) -> Tuple[Dict[str, Tensor], Dict[str, Any], Tensor]:
    """One scene, two GT objects, two proposals predicting them exactly (headings in native space)."""
    nh, ns, nc = 12, 3, 3
    mean_sizes = torch.rand(ns, 3) + 1.0
    centers = torch.tensor([[1.0, 1.0, 0.5], [3.0, 2.0, 0.8]])
    classes = torch.tensor([0, 2])
    native_class, native_residual = angle_to_class(native_headings % (2 * math.pi), nh)

    heading_scores = torch.full((1, 2, nh), -10.0)
    heading_res_norm = torch.zeros(1, 2, nh)
    size_scores = torch.full((1, 2, ns), -10.0)
    sem_cls_scores = torch.full((1, 2, nc), -10.0)
    objectness = torch.zeros(1, 2, 2)
    objectness[..., 0] = -10.0
    objectness[..., 1] = 10.0
    for i in range(2):
        heading_scores[0, i, native_class[i]] = 10.0
        heading_res_norm[0, i, native_class[i]] = native_residual[i] / (math.pi / nh)
        size_scores[0, i, classes[i]] = 10.0
        sem_cls_scores[0, i, classes[i]] = 10.0

    # Two seeds inside each object vote exactly for its center; a far seed is masked out.
    pos_seed = torch.cat([centers, centers + 0.1, torch.tensor([[9.0, 9.0, 9.0]])])
    pos_vote = torch.cat([centers, centers, torch.tensor([[0.0, 0.0, 0.0]])])
    output: Dict[str, Tensor] = {
        "objectness_scores": objectness,
        "center": centers.unsqueeze(0).clone(),
        "heading_scores": heading_scores,
        "heading_residuals_normalized": heading_res_norm,
        "size_scores": size_scores,
        "size_residuals_normalized": torch.zeros(1, 2, ns, 3),
        "sem_cls_scores": sem_cls_scores,
        "pos_vote_aggr": centers.unsqueeze(0).clone(),
        "pos_seed": pos_seed,
        "pos_vote": pos_vote,
        "batch_seed": torch.zeros(5, dtype=torch.long),
        "batch_vote": torch.zeros(5, dtype=torch.long),
    }
    # The library heading is counter-clockwise, the model's native heading is its negation.
    boxes = torch.cat([centers, mean_sizes[classes], (-native_headings).unsqueeze(-1)], dim=-1)
    data: Dict[str, Any] = {
        DataKeys.BOX: boxes,
        DataKeys.LABEL: classes,
        DataKeys.BATCH_BOX: torch.zeros(2, dtype=torch.long),
    }
    return output, data, mean_sizes


def test_votenet_loss_perfect_predictions_near_zero() -> None:
    torch.manual_seed(0)
    native = torch.tensor([0.4, -1.2])
    output, data, mean_sizes = _perfect_data(native)
    loss_fn = VoteNetLoss(num_heading_bins=12, num_size_clusters=3, num_classes=3, mean_sizes=mean_sizes)
    out = loss_fn(output, data)
    for key in ("vote_loss", "center_loss", "heading_cls_loss", "heading_res_loss", "size_cls_loss", "size_res_loss"):
        assert out[key] < 1e-4, key
    assert out["loss"] < 1e-2
    assert out["obj_acc"] > 0.99


def test_votenet_loss_heading_labels_expect_ccw_convention() -> None:
    """Native-space heading predictions score ~0 against CCW GT boxes, and worse against unnegated headings."""
    torch.manual_seed(0)
    native = torch.tensor([0.4, -1.2])
    output, data, mean_sizes = _perfect_data(native)
    loss_fn = VoteNetLoss(num_heading_bins=12, num_size_clusters=3, num_classes=3, mean_sizes=mean_sizes)
    ccw = loss_fn(output, data)

    data[DataKeys.BOX] = data[DataKeys.BOX].clone()
    data[DataKeys.BOX][:, 6] = native
    wrong = loss_fn(output, data)
    assert ccw["heading_cls_loss"] < 1e-4
    assert wrong["heading_cls_loss"] > 0.1
    assert wrong["loss"] > ccw["loss"]


def test_votenet_loss_perturbed_center_is_larger() -> None:
    torch.manual_seed(0)
    output, data, mean_sizes = _perfect_data(torch.tensor([0.4, -1.2]))
    loss_fn = VoteNetLoss(num_heading_bins=12, num_size_clusters=3, num_classes=3, mean_sizes=mean_sizes)
    perfect = loss_fn(output, data)
    output["center"] = output["center"] + 0.2
    out = loss_fn(output, data)
    assert out["center_loss"] > perfect["center_loss"] + 0.01
    assert out["loss"] > perfect["loss"]


def test_votenet_loss_perturbed_vote_is_larger() -> None:
    torch.manual_seed(0)
    output, data, mean_sizes = _perfect_data(torch.tensor([0.4, -1.2]))
    loss_fn = VoteNetLoss(num_heading_bins=12, num_size_clusters=3, num_classes=3, mean_sizes=mean_sizes)
    perfect = loss_fn(output, data)
    output["pos_vote"] = output["pos_vote"].clone()
    output["pos_vote"][0] += 0.3
    out = loss_fn(output, data)
    assert out["vote_loss"] > perfect["vote_loss"] + 0.01
    output["pos_vote"][4] += 5.0  # the masked far seed changes nothing
    assert torch.isclose(loss_fn(output, data)["vote_loss"], out["vote_loss"])
