from typing import Any, Dict, Tuple

import pytest
import torch
from torch import Tensor

from torch_pointcloud.losses import VoteNetLoss


def _fake_inputs(
    batch_size: int = 2,
    num_proposal: int = 8,
    max_obj: int = 4,
    num_point: int = 64,
    num_seed: int = 16,
    num_heading_bin: int = 12,
    num_size_cluster: int = 10,
    num_class: int = 10,
) -> Tuple[Dict[str, Tensor], Dict[str, Any]]:
    torch.manual_seed(0)
    output: Dict[str, Tensor] = {
        "objectness_scores": torch.randn(batch_size, num_proposal, 2),
        "center": torch.randn(batch_size, num_proposal, 3),
        "heading_scores": torch.randn(batch_size, num_proposal, num_heading_bin),
        "heading_residuals_normalized": torch.randn(batch_size, num_proposal, num_heading_bin),
        "size_scores": torch.randn(batch_size, num_proposal, num_size_cluster),
        "size_residuals_normalized": torch.randn(batch_size, num_proposal, num_size_cluster, 3),
        "sem_cls_scores": torch.randn(batch_size, num_proposal, num_class),
        "aggregated_vote_pos": torch.randn(batch_size, num_proposal, 3),
        "seed_pos": torch.randn(batch_size, num_seed, 3),
        "vote_pos": torch.randn(batch_size, num_seed, 3),
        "seed_inds": torch.randint(0, num_point, (batch_size, num_seed)),
    }
    box_label_mask = torch.zeros(batch_size, max_obj)
    box_label_mask[:, :2] = 1.0
    batch: Dict[str, Any] = {
        "center_label": torch.randn(batch_size, max_obj, 3),
        "heading_class_label": torch.randint(0, num_heading_bin, (batch_size, max_obj)),
        "heading_residual_label": torch.randn(batch_size, max_obj),
        "size_class_label": torch.randint(0, num_size_cluster, (batch_size, max_obj)),
        "size_residual_label": torch.randn(batch_size, max_obj, 3),
        "sem_cls_label": torch.randint(0, num_class, (batch_size, max_obj)),
        "box_label_mask": box_label_mask,
        "vote_label": torch.randn(batch_size, num_point, 9),
        "vote_label_mask": (torch.rand(batch_size, num_point) > 0.5).long(),
    }
    return output, batch


def _mean_size(num_size_cluster: int = 10) -> Tensor:
    torch.manual_seed(1)
    return torch.rand(num_size_cluster, 3) + 0.5


def test_votenet_loss_returns_scalar_dict() -> None:
    loss_fn = VoteNetLoss(num_heading_bin=12, num_size_cluster=10, num_class=10, mean_size_arr=_mean_size())
    output, batch = _fake_inputs()
    out = loss_fn(output, batch)
    for key in ("loss", "vote_loss", "objectness_loss", "box_loss", "sem_cls_loss", "obj_acc"):
        assert key in out, key
        assert out[key].ndim == 0
        assert torch.isfinite(out[key])


def test_votenet_loss_backward() -> None:
    loss_fn = VoteNetLoss(num_heading_bin=12, num_size_cluster=10, num_class=10, mean_size_arr=_mean_size())
    output, batch = _fake_inputs()
    for value in output.values():
        if value.is_floating_point():
            value.requires_grad_(True)
    loss_fn(output, batch)["loss"].backward()
    assert output["center"].grad is not None
    assert output["objectness_scores"].grad is not None


def test_votenet_loss_scannet_single_heading_bin() -> None:
    loss_fn = VoteNetLoss(num_heading_bin=1, num_size_cluster=18, num_class=18, mean_size_arr=_mean_size(18))
    output, batch = _fake_inputs(num_heading_bin=1, num_size_cluster=18, num_class=18)
    assert torch.isfinite(loss_fn(output, batch)["loss"])


def test_votenet_loss_no_positive_targets_is_finite() -> None:
    loss_fn = VoteNetLoss(num_heading_bin=12, num_size_cluster=10, num_class=10, mean_size_arr=_mean_size())
    output, batch = _fake_inputs()
    batch["box_label_mask"] = torch.zeros_like(batch["box_label_mask"])
    batch["vote_label_mask"] = torch.zeros_like(batch["vote_label_mask"])
    assert torch.isfinite(loss_fn(output, batch)["loss"])


def test_votenet_loss_bad_mean_size_shape() -> None:
    with pytest.raises(ValueError, match="mean_size_arr"):
        VoteNetLoss(num_heading_bin=12, num_size_cluster=10, num_class=10, mean_size_arr=torch.rand(5, 3))
