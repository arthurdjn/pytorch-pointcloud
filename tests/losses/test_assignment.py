import pytest
import torch

from torch_pointcloud.losses import assignment_negative_log_likelihood


def test_assignment_negative_log_likelihood_labels_matches_and_dustbins() -> None:
    rows = torch.tensor([[[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]])
    cols = torch.tensor([[[0.0, 0.0, 0.01], [9.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
    log_assignment = torch.full((1, 3, 4), -10.0)
    # Row 0 matches columns 0 and 2, row 1 and column 1 go to the dustbins; column 2 is masked out.
    log_assignment[0, 0, 0] = -1.0
    log_assignment[0, 1, 3] = -2.0
    log_assignment[0, 2, 1] = -3.0
    col_mask = torch.tensor([[True, True, False]])
    loss = assignment_negative_log_likelihood(log_assignment, rows, cols, radius=0.05, col_mask=col_mask)
    assert loss.item() == pytest.approx(2.0)


def test_assignment_negative_log_likelihood_row_mask_drops_the_row() -> None:
    points = torch.tensor([[[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]])
    log_assignment = torch.full((1, 3, 3), -10.0)
    # Row 0 is masked out, so column 0 loses its match and goes to the dustbin; row 1 matches column 1.
    log_assignment[0, 1, 1] = -1.0
    log_assignment[0, 2, 0] = -4.0
    row_mask = torch.tensor([[False, True]])
    loss = assignment_negative_log_likelihood(log_assignment, points, points, radius=0.05, row_mask=row_mask)
    assert loss.item() == pytest.approx(2.5)


def test_assignment_negative_log_likelihood_without_labels_is_zero() -> None:
    points = torch.rand(1, 2, 3)
    log_assignment = torch.full((1, 3, 3), -10.0, requires_grad=True)
    mask = torch.zeros(1, 2, dtype=torch.bool)
    loss = assignment_negative_log_likelihood(log_assignment, points, points, 0.05, row_mask=mask, col_mask=mask)
    loss.backward()
    assert loss.item() == 0.0
    assert log_assignment.grad is not None and torch.count_nonzero(log_assignment.grad) == 0
