import math

import pytest
import torch
import torch.nn.functional as F

from torch_pointcloud.losses import (
    ChamferDistance,
    CornerLoss,
    GaussianFocalLoss,
    Poly1FocalLoss,
    SigmoidFocalLoss,
    chamfer_distance,
    corner_loss,
    gaussian_focal_loss,
    lovasz_softmax,
    one_hot_foreground,
    poly1_focal_loss,
    sigmoid_focal_loss,
)
from torch_pointcloud.losses import functional as LF


def test_functional_reexports_every_tensor_function() -> None:
    expected = {
        "chamfer_distance",
        "corner_loss",
        "gaussian_focal_loss",
        "hungarian_match",
        "hungarian_match_batched",
        "kpconv_deform_regularizer",
        "lovasz_softmax",
        "one_hot_foreground",
        "poly1_focal_loss",
        "sigmoid_focal_loss",
        "tnet_orthogonality_regularizer",
    }
    assert set(LF.__all__) == expected
    assert all(callable(getattr(LF, name)) for name in expected)
    assert LF.sigmoid_focal_loss is sigmoid_focal_loss


def test_sigmoid_focal_loss_matches_reference_formula() -> None:
    torch.manual_seed(0)
    logits = torch.randn(4, 3)
    targets = (torch.rand(4, 3) > 0.5).float()
    p = logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    alpha_t = 0.25 * targets + 0.75 * (1 - targets)
    expected = alpha_t * (1 - p_t) ** 2 * ce
    assert torch.allclose(sigmoid_focal_loss(logits, targets), expected, atol=1e-6)
    weights = torch.rand(4)
    assert torch.allclose(sigmoid_focal_loss(logits, targets, weights), expected * weights[:, None], atol=1e-6)


def test_sigmoid_focal_loss_module_reductions() -> None:
    torch.manual_seed(0)
    logits, targets = torch.randn(4, 3), (torch.rand(4, 3) > 0.5).float()
    per_element = sigmoid_focal_loss(logits, targets)
    assert SigmoidFocalLoss(reduction="none")(logits, targets).shape == (4, 3)
    assert torch.allclose(SigmoidFocalLoss(reduction="sum")(logits, targets), per_element.sum())
    assert torch.allclose(SigmoidFocalLoss()(logits, targets), per_element.mean())
    with pytest.raises(ValueError, match="reduction"):
        SigmoidFocalLoss(reduction="avg")  # type: ignore[arg-type]


def test_gaussian_focal_loss_matches_centernet_formula() -> None:
    torch.manual_seed(0)
    pred = torch.rand(2, 3, 4, 4)
    target = torch.rand(2, 3, 4, 4)
    target[0, 0, 1, 1] = 1.0
    target[1, 2, 0, 3] = 1.0
    p = pred.clamp(1e-4, 1 - 1e-4)
    pos = target.eq(1).float()
    expected = -(p.log() * (1 - p) ** 2 * pos + (1 - p).log() * p**2 * (1 - target) ** 4 * (1 - pos)).sum() / 2
    assert torch.allclose(gaussian_focal_loss(pred, target), expected, atol=1e-5)


def test_gaussian_focal_loss_without_positives_is_unnormalized_negative_term() -> None:
    pred = torch.full((2, 2), 0.3)
    target = torch.full((2, 2), 0.5)
    expected = -((1 - pred).log() * pred**2 * (1 - target) ** 4).sum()
    assert torch.allclose(gaussian_focal_loss(pred, target), expected)


def test_gaussian_focal_loss_module_takes_logits() -> None:
    torch.manual_seed(0)
    logits = torch.randn(1, 2, 4, 4)
    target = torch.zeros(1, 2, 4, 4)
    target[0, 1, 2, 2] = 1.0
    assert torch.allclose(GaussianFocalLoss()(logits, target), gaussian_focal_loss(logits.sigmoid(), target))


def test_corner_loss_zero_for_equal_and_flipped_boxes_and_hand_value_for_shift() -> None:
    box = torch.tensor([[1.0, -2.0, 0.5, 3.9, 1.6, 1.56, 0.3]])
    assert corner_loss(box, box).abs().max() < 1e-6
    flipped = box.clone()
    flipped[:, 6] += math.pi
    assert corner_loss(box, flipped).abs().max() < 1e-5
    shifted = box.clone()
    shifted[:, 0] += 0.5  # every corner is 0.5 off, inside the quadratic zone: 0.5 * 0.5^2
    assert torch.isclose(corner_loss(box, shifted)[0], torch.tensor(0.125), atol=1e-6)


def test_corner_loss_module_reductions() -> None:
    torch.manual_seed(0)
    pred = torch.rand(4, 7) + torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0])
    gt = pred + torch.randn(4, 7) * 0.2
    per_box = corner_loss(pred, gt, beta=0.5)
    assert torch.allclose(CornerLoss(beta=0.5, reduction="none")(pred, gt), per_box)
    assert torch.allclose(CornerLoss(beta=0.5, reduction="sum")(pred, gt), per_box.sum())
    assert torch.allclose(CornerLoss(beta=0.5)(pred, gt), per_box.mean())
    assert "reduction='mean'" in repr(CornerLoss())
    with pytest.raises(ValueError, match="reduction"):
        CornerLoss(reduction="avg")  # type: ignore[arg-type]


def test_poly1_focal_loss_matches_openpoints_formula() -> None:
    """The openpoints `Poly1FocalLoss` (epsilon 1, alpha 0.25, gamma 2) written out: focal term + eps * (1 - p_t)^3."""
    torch.manual_seed(0)
    logits, labels = torch.randn(5, 4), torch.tensor([0, 1, 3, 2, 0])
    targets = torch.nn.functional.one_hot(labels, 4).float()
    p = logits.sigmoid()
    p_t = targets * p + (1 - targets) * (1 - p)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    focal = (0.25 * targets + 0.75 * (1 - targets)) * bce * (1 - p_t) ** 2
    expected = focal + (1 - p_t) ** 3
    assert torch.allclose(poly1_focal_loss(logits, targets), expected)
    assert torch.allclose(poly1_focal_loss(logits, targets, alpha=None), bce * (1 - p_t) ** 2 + (1 - p_t) ** 3)
    # the module one-hot encodes class indices and averages over every element, as openpoints does
    assert torch.allclose(Poly1FocalLoss()(logits, labels), expected.mean())
    assert torch.allclose(Poly1FocalLoss(reduction="sum")(logits, targets), expected.sum())
    assert Poly1FocalLoss(reduction="none")(logits, labels).shape == (5, 4)
    with pytest.raises(ValueError, match="reduction"):
        Poly1FocalLoss(reduction="avg")  # type: ignore[arg-type]


def test_one_hot_foreground_drops_background_and_ignored_rows() -> None:
    out = one_hot_foreground(torch.tensor([[2, 0, -1, 1]]), 3)
    assert out.tolist() == [[[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]


def test_chamfer_distance_module_matches_function() -> None:
    torch.manual_seed(0)
    pred, target = torch.randn(2, 8, 3), torch.randn(2, 6, 3)
    criterion = ChamferDistance(norm="l1")
    assert torch.allclose(criterion(pred, target), chamfer_distance(pred, target, norm="l1"))
    assert "norm='l1'" in repr(criterion)
    with pytest.raises(ValueError, match="norm"):
        ChamferDistance(norm="linf")  # type: ignore[arg-type]


def test_chamfer_distance_matches_the_dense_formula_and_its_gradients() -> None:
    torch.manual_seed(0)
    pred = torch.randn(2, 50, 3, requires_grad=True)
    target = torch.randn(2, 40, 3, requires_grad=True)
    sq_dist = (pred[:, :, None] - target[:, None]).pow(2).sum(-1)
    dense = sq_dist.min(2).values.mean() + sq_dist.min(1).values.mean()
    dense_grads = torch.autograd.grad(dense, (pred, target))

    loss = chamfer_distance(pred, target, norm="l2")
    grads = torch.autograd.grad(loss, (pred, target))
    assert torch.allclose(loss, dense)
    assert all(torch.allclose(g, d) for g, d in zip(grads, dense_grads))


def test_lovasz_softmax_perfect_prediction_is_zero_and_validates_classes() -> None:
    labels = torch.tensor([0, 1, 2, 0])
    probas = F.one_hot(labels, 3).float()
    assert lovasz_softmax(probas, labels).item() == 0.0
    with pytest.raises(ValueError, match="'present' or 'all'"):
        lovasz_softmax(probas, labels, classes="some")  # type: ignore[arg-type]
