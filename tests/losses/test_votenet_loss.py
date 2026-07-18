import math
from typing import Any, Dict, Tuple

import pytest
import torch
from torch import Tensor

from torch_pointcloud.losses import VoteNetLoss
from torch_pointcloud.transforms.functional import angle_to_class


def _create_data(
    batch_size: int = 2,
    num_proposal: int = 8,
    max_obj: int = 4,
    num_point: int = 64,
    num_seed: int = 16,
    num_heading_bin: int = 12,
    num_size_cluster: int = 10,
    num_classes: int = 10,
) -> Tuple[Dict[str, Tensor], Dict[str, Any]]:
    torch.manual_seed(0)
    batch_seed = torch.arange(batch_size).repeat_interleave(num_seed)
    output: Dict[str, Tensor] = {
        "objectness_scores": torch.randn(batch_size, num_proposal, 2),
        "center": torch.randn(batch_size, num_proposal, 3),
        "heading_scores": torch.randn(batch_size, num_proposal, num_heading_bin),
        "heading_residuals_normalized": torch.randn(batch_size, num_proposal, num_heading_bin),
        "size_scores": torch.randn(batch_size, num_proposal, num_size_cluster),
        "size_residuals_normalized": torch.randn(batch_size, num_proposal, num_size_cluster, 3),
        "sem_cls_scores": torch.randn(batch_size, num_proposal, num_classes),
        "pos_vote_aggr": torch.randn(batch_size, num_proposal, 3),
        "pos_seed": torch.randn(batch_size * num_seed, 3),
        "pos_vote": torch.randn(batch_size * num_seed, 3),
        "seed_indices": torch.randint(0, num_point, (batch_size * num_seed,)) + batch_seed * num_point,
        "batch_seed": batch_seed,
        "batch_vote": batch_seed,
    }
    box_label_mask = torch.zeros(batch_size, max_obj)
    box_label_mask[:, :2] = 1.0
    batch: Dict[str, Any] = {
        "center_label": torch.randn(batch_size, max_obj, 3),
        "heading_class_label": torch.randint(0, num_heading_bin, (batch_size, max_obj)),
        "heading_residual_label": torch.randn(batch_size, max_obj),
        "size_class_label": torch.randint(0, num_size_cluster, (batch_size, max_obj)),
        "size_residual_label": torch.randn(batch_size, max_obj, 3),
        "sem_cls_label": torch.randint(0, num_classes, (batch_size, max_obj)),
        "box_label_mask": box_label_mask,
        "vote_label": torch.randn(batch_size, num_point, 9),
        "vote_label_mask": (torch.rand(batch_size, num_point) > 0.5).long(),
        "batch": torch.arange(batch_size).repeat_interleave(num_point),
    }
    return output, batch


def _mean_size(num_size_cluster: int = 10) -> Tensor:
    torch.manual_seed(1)
    return torch.rand(num_size_cluster, 3) + 0.5


def test_votenet_loss_returns_scalar_dict() -> None:
    loss_fn = VoteNetLoss(num_heading_bin=12, num_size_cluster=10, num_classes=10, mean_sizes=_mean_size())
    output, batch = _create_data()
    out = loss_fn(output, batch)
    for key in ("loss", "vote_loss", "objectness_loss", "box_loss", "sem_cls_loss", "obj_acc"):
        assert key in out, key
        assert out[key].ndim == 0
        assert torch.isfinite(out[key])


def test_votenet_loss_backward() -> None:
    loss_fn = VoteNetLoss(num_heading_bin=12, num_size_cluster=10, num_classes=10, mean_sizes=_mean_size())
    output, batch = _create_data()
    for value in output.values():
        if value.is_floating_point():
            value.requires_grad_(True)
    loss_fn(output, batch)["loss"].backward()
    assert output["center"].grad is not None
    assert output["objectness_scores"].grad is not None


def test_votenet_loss_scannet_single_heading_bin() -> None:
    loss_fn = VoteNetLoss(num_heading_bin=1, num_size_cluster=18, num_classes=18, mean_sizes=_mean_size(18))
    output, batch = _create_data(num_heading_bin=1, num_size_cluster=18, num_classes=18)
    assert torch.isfinite(loss_fn(output, batch)["loss"])


def test_votenet_loss_no_positive_targets_is_finite() -> None:
    loss_fn = VoteNetLoss(num_heading_bin=12, num_size_cluster=10, num_classes=10, mean_sizes=_mean_size())
    output, batch = _create_data()
    batch["box_label_mask"] = torch.zeros_like(batch["box_label_mask"])
    batch["vote_label_mask"] = torch.zeros_like(batch["vote_label_mask"])
    assert torch.isfinite(loss_fn(output, batch)["loss"])


def test_votenet_loss_bad_mean_size_shape() -> None:
    with pytest.raises(ValueError, match="mean_sizes"):
        VoteNetLoss(num_heading_bin=12, num_size_cluster=10, num_classes=10, mean_sizes=torch.rand(5, 3))


def test_votenet_loss_heading_to_native_inverts_ccw_binning() -> None:
    """Re-binning the bins of the negated angle recovers the bins of the angle itself, exactly."""
    loss_fn = VoteNetLoss(num_heading_bin=12, num_size_cluster=10, num_classes=10, mean_sizes=_mean_size())
    theta = torch.arange(0.01, 2 * math.pi, 0.13)
    ccw_class, ccw_residual = angle_to_class((-theta) % (2 * math.pi), 12)
    native_class, native_residual = loss_fn._heading_to_native(ccw_class, ccw_residual)
    expected_class, expected_residual = angle_to_class(theta, 12)
    assert torch.equal(native_class, expected_class)
    assert torch.allclose(native_residual, expected_residual, atol=1e-6)


def test_votenet_loss_single_bin_heading_conversion_is_identity() -> None:
    loss_fn = VoteNetLoss(num_heading_bin=1, num_size_cluster=10, num_classes=10, mean_sizes=_mean_size())
    cls = torch.zeros(2, 4, dtype=torch.long)
    residual = torch.zeros(2, 4)
    native_class, native_residual = loss_fn._heading_to_native(cls, residual)
    assert torch.equal(native_class, cls)
    assert torch.allclose(native_residual, residual, atol=1e-6)


def _perfect_data(native_headings: Tensor) -> Tuple[Dict[str, Tensor], Dict[str, Any], Tensor]:
    """One scene, two GT objects, two proposals predicting them exactly (headings in native space)."""
    nh, ns, nc, num_point, num_seed = 12, 3, 3, 8, 4
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

    pos_seed = torch.rand(num_seed, 3)
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
        "pos_vote": pos_seed.clone(),
        "seed_indices": torch.arange(num_seed),
        "batch_seed": torch.zeros(num_seed, dtype=torch.long),
        "batch_vote": torch.zeros(num_seed, dtype=torch.long),
    }
    ccw_class, ccw_residual = angle_to_class((-native_headings) % (2 * math.pi), nh)
    batch: Dict[str, Any] = {
        "center_label": centers.unsqueeze(0).clone(),
        "heading_class_label": ccw_class.unsqueeze(0),
        "heading_residual_label": ccw_residual.unsqueeze(0),
        "size_class_label": classes.unsqueeze(0),
        "size_residual_label": torch.zeros(1, 2, 3),
        "sem_cls_label": classes.unsqueeze(0),
        "box_label_mask": torch.ones(1, 2),
        "vote_label": torch.zeros(1, num_point, 9),
        "vote_label_mask": torch.ones(1, num_point, dtype=torch.long),
        "batch": torch.zeros(num_point, dtype=torch.long),
    }
    return output, batch, mean_sizes


def test_votenet_loss_perfect_predictions_near_zero() -> None:
    torch.manual_seed(0)
    native = torch.tensor([0.4, -1.2])
    output, batch, mean_sizes = _perfect_data(native)
    loss_fn = VoteNetLoss(num_heading_bin=12, num_size_cluster=3, num_classes=3, mean_sizes=mean_sizes)
    out = loss_fn(output, batch)
    for key in ("vote_loss", "center_loss", "heading_cls_loss", "heading_res_loss", "size_cls_loss", "size_res_loss"):
        assert out[key] < 1e-4, key
    assert out["loss"] < 1e-2
    assert out["obj_acc"] > 0.99


def test_votenet_loss_heading_labels_expect_ccw_convention() -> None:
    """Native-space heading predictions score ~0 against CCW GT bins, and worse against unnegated bins."""
    torch.manual_seed(0)
    native = torch.tensor([0.4, -1.2])
    output, batch, mean_sizes = _perfect_data(native)
    loss_fn = VoteNetLoss(num_heading_bin=12, num_size_cluster=3, num_classes=3, mean_sizes=mean_sizes)
    ccw = loss_fn(output, batch)

    wrong_class, wrong_residual = angle_to_class(native % (2 * math.pi), 12)
    batch["heading_class_label"] = wrong_class.unsqueeze(0)
    batch["heading_residual_label"] = wrong_residual.unsqueeze(0)
    wrong = loss_fn(output, batch)
    assert ccw["heading_cls_loss"] < 1e-4
    assert wrong["heading_cls_loss"] > 0.1
    assert wrong["loss"] > ccw["loss"]


def test_votenet_loss_perturbed_center_is_larger() -> None:
    torch.manual_seed(0)
    output, batch, mean_sizes = _perfect_data(torch.tensor([0.4, -1.2]))
    loss_fn = VoteNetLoss(num_heading_bin=12, num_size_cluster=3, num_classes=3, mean_sizes=mean_sizes)
    perfect = loss_fn(output, batch)
    output["center"] = output["center"] + 0.2
    out = loss_fn(output, batch)
    assert out["center_loss"] > perfect["center_loss"] + 0.01
    assert out["loss"] > perfect["loss"]
