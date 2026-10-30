import pytest
import torch

from torch_pointcloud.losses import WeightedCircleLoss, weighted_circle_loss

MARGINS = (0.1, 1.4, 0.1, 1.4, 24.0)


def test_weighted_circle_loss_prefers_close_positives() -> None:
    positive = torch.eye(3, dtype=torch.bool)
    good = torch.full((3, 3), 1.5)
    good.fill_diagonal_(0.05)
    bad = torch.full((3, 3), 0.3)
    bad.fill_diagonal_(1.0)
    loss_good = weighted_circle_loss(good, positive, ~positive, positive.float(), *MARGINS)
    loss_bad = weighted_circle_loss(bad, positive, ~positive, positive.float(), *MARGINS)
    assert loss_good.ndim == 0
    assert loss_good < loss_bad


def test_weighted_circle_loss_without_positives_is_zero() -> None:
    distances = torch.rand(3, 3, requires_grad=True)
    no_positive = torch.zeros(3, 3, dtype=torch.bool)
    loss = weighted_circle_loss(distances, no_positive, ~no_positive, no_positive.float(), *MARGINS)
    assert loss.item() == 0.0 and loss.requires_grad


def test_weighted_circle_loss_module_matches_function() -> None:
    torch.manual_seed(0)
    distances = torch.rand(4, 5) * 2
    positive = torch.rand(4, 5) > 0.7
    weights = torch.rand(4, 5)
    criterion = WeightedCircleLoss(*MARGINS)
    expected = weighted_circle_loss(distances, positive, ~positive, weights, *MARGINS)
    assert torch.equal(criterion(distances, positive, ~positive, weights), expected)
    assert "log_scale=24.0" in repr(criterion)


def test_weighted_circle_loss_batches_padded_matrices() -> None:
    """Every matrix of a padded batch gets the loss it has alone, the padding left out of every term."""
    torch.manual_seed(0)
    sizes = [(4, 5), (3, 2)]
    distances = torch.rand(2, 4, 5) * 2
    positive = torch.rand(2, 4, 5) > 0.7
    weights = torch.rand(2, 4, 5)
    mask = torch.zeros(2, 4, 5, dtype=torch.bool)
    for i, (n, m) in enumerate(sizes):
        mask[i, :n, :m] = True
    distances.requires_grad_(True)
    losses = weighted_circle_loss(distances, positive, ~positive, weights, *MARGINS, mask=mask)
    assert losses.shape == (2,)
    for i, (n, m) in enumerate(sizes):
        alone = weighted_circle_loss(
            distances[i, :n, :m], positive[i, :n, :m], ~positive[i, :n, :m], weights[i, :n, :m], *MARGINS
        )
        assert losses[i].item() == pytest.approx(alone.item(), rel=1e-5)
    losses.sum().backward()
    assert distances.grad is not None and torch.isfinite(distances.grad).all()
    assert torch.all(distances.grad[~mask] == 0)
