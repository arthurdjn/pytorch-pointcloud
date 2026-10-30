import math

import pytest
import torch

from torch_pointcloud.metrics import (
    alignment_error,
    feature_matching_recall,
    inlier_ratio,
    registration_recall,
    root_mean_square_alignment_error,
    rotation_error,
    translation_error,
)


def _transform(degrees: float, translation: list[float]) -> torch.Tensor:
    c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    transform = torch.eye(4)
    transform[:3, :3] = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    transform[:3, 3] = torch.tensor(translation)
    return transform


def test_rotation_and_translation_errors() -> None:
    target = _transform(10.0, [0.0, 0.0, 0.0])
    estimate = _transform(25.0, [0.3, 0.4, 0.0])
    assert rotation_error(estimate, target).item() == pytest.approx(15.0, abs=1e-3)
    assert translation_error(estimate, target).item() == pytest.approx(0.5, abs=1e-6)


def test_errors_are_batched() -> None:
    target = torch.stack([_transform(0.0, [0.0, 0.0, 0.0]), _transform(0.0, [0.0, 0.0, 0.0])])
    estimate = torch.stack([_transform(5.0, [0.0, 0.0, 0.0]), _transform(0.0, [1.0, 0.0, 0.0])])
    assert torch.allclose(rotation_error(estimate, target), torch.tensor([5.0, 0.0]), atol=1e-3)
    assert torch.allclose(translation_error(estimate, target), torch.tensor([0.0, 1.0]), atol=1e-6)


def test_root_mean_square_alignment_error_translation_only_with_identity_information() -> None:
    """With an identity information matrix the RMSE of a pure translation is its norm."""
    target = _transform(0.0, [0.0, 0.0, 0.0])
    estimate = _transform(0.0, [0.3, 0.4, 0.0])
    assert root_mean_square_alignment_error(estimate, target, torch.eye(6)).item() == pytest.approx(0.5, abs=1e-5)


@pytest.mark.parametrize(
    "degrees",
    [
        pytest.param(90.0, id="quarter-turn"),
        pytest.param(179.99, id="near-half-turn"),
        pytest.param(180.0, id="half-turn"),
    ],
)
def test_root_mean_square_alignment_error_of_a_rotation_reads_the_quaternion(degrees: float) -> None:
    """With an identity information matrix the RMSE of a pure rotation is the norm of its quaternion vector part,
    $\sin(\theta / 2)$, up to and including a half turn."""
    estimate = _transform(degrees, [0.0, 0.0, 0.0])
    rmse = root_mean_square_alignment_error(estimate, torch.eye(4), torch.eye(6)).item()
    assert rmse == pytest.approx(math.sin(math.radians(degrees) / 2), abs=1e-5)


def test_root_mean_square_alignment_error_matches_point_displacement() -> None:
    """The information matrix of a point set reproduces the mean squared displacement of its points under a
    translation (the rotational part of the protocol reads the quaternion, a half-angle approximation)."""
    torch.manual_seed(0)
    points = torch.randn(500, 3)
    g = torch.zeros(500, 3, 6)
    g[:, :3, :3] = torch.eye(3)
    g[:, 0, 4], g[:, 0, 5] = points[:, 2], -points[:, 1]
    g[:, 1, 3], g[:, 1, 5] = -points[:, 2], points[:, 0]
    g[:, 2, 3], g[:, 2, 4] = points[:, 1], -points[:, 0]
    information = (g.transpose(1, 2) @ g).sum(0)

    target = torch.eye(4)
    estimate = _transform(0.0, [0.01, -0.02, 0.005])
    rmse = root_mean_square_alignment_error(estimate, target, information).item()
    moved = points @ estimate[:3, :3].T + estimate[:3, 3]
    expected = (moved - points).norm(dim=1).pow(2).mean().sqrt().item()
    assert rmse == pytest.approx(expected, rel=1e-2)


def test_inlier_ratio_per_pair() -> None:
    src = torch.tensor([[0.0, 0, 0], [1.0, 0, 0], [0.0, 0, 0], [0.0, 0, 0]])
    ref = torch.tensor([[0.0, 0, 0], [1.5, 0, 0], [0.0, 0, 1.0], [0.0, 0, 1.3]])
    target = torch.stack([torch.eye(4), _transform(0.0, [0.0, 0.0, 1.0])])
    batch = torch.tensor([0, 0, 1, 1])
    assert inlier_ratio(src, ref, target, radius=0.1, batch=batch).tolist() == [0.5, 0.5]


def test_inlier_ratio_empty_pair_is_nan() -> None:
    src = torch.zeros(2, 3)
    target = torch.eye(4).repeat(3, 1, 1)
    ratios = inlier_ratio(src, src, target, radius=0.1, batch=torch.tensor([0, 0]))
    assert ratios.shape == (3,)
    assert ratios[0].item() == 1.0
    assert torch.isnan(ratios[1]) and torch.isnan(ratios[2])


def test_alignment_error_per_pair() -> None:
    points = torch.rand(20, 3)
    batch = torch.tensor([0] * 12 + [1] * 8)
    transform = torch.stack([_transform(0.0, [0.3, 0.0, 0.0]), _transform(0.0, [0.0, 0.0, 0.0])])
    target = torch.stack([torch.eye(4), _transform(0.0, [0.0, 0.4, 0.0])])
    errors = alignment_error(points, transform, target, batch=batch)
    assert torch.allclose(errors, torch.tensor([0.3, 0.4]), atol=1e-6)


def test_feature_matching_recall_counts_ratios_at_threshold() -> None:
    ratios = torch.tensor([0.0, 0.04, 0.05, 0.7])
    assert feature_matching_recall(ratios, threshold=0.05) == pytest.approx(0.5)
    assert feature_matching_recall(ratios, threshold=0.5) == pytest.approx(0.25)


def test_registration_recall_counts_errors_under_threshold() -> None:
    errors = torch.tensor([0.01, 0.1, 0.2, 0.5])
    assert registration_recall(errors, threshold=0.2) == pytest.approx(0.5)
    assert registration_recall(errors, threshold=1.0) == pytest.approx(1.0)
