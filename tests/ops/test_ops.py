import pytest
import torch

from torch_pointcloud.ops import k_interpolate, knn, three_interpolate


@pytest.mark.parametrize(
    "p1, p2, k, lengths1, lengths2, expected_dists, expected_idxs",
    [
        (
            torch.tensor([[[0.0, 0.0, 0.0]]]),
            torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]]),
            3,
            None,
            None,
            torch.tensor([[[0.0, 1.0, 4.0]]]),
            torch.tensor([[[0, 1, 2]]]),
        ),
        (
            torch.tensor([[[0.0, 0.0, 0.0]]]),
            torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]]),
            3,
            1,
            3,
            torch.tensor([[[0.0, 1.0, 4.0]]]),
            torch.tensor([[[0, 1, 2]]]),
        ),
        (
            torch.tensor([[[0.0, 0.0, 0.0]]], dtype=torch.float64),
            torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0, 0.0]]], dtype=torch.float64),
            3,
            torch.tensor([1], dtype=torch.int64),
            torch.tensor([3], dtype=torch.int64),
            torch.tensor([[[0.0, 1.0, 4.0]]], dtype=torch.float64),
            torch.tensor([[[0, 1, 2]]], dtype=torch.int64),
        ),
    ],
)
def test_knn(
    p1: torch.Tensor,
    p2: torch.Tensor,
    k: torch.Tensor,
    lengths1: torch.Tensor,
    lengths2: torch.Tensor,
    expected_dists: torch.Tensor,
    expected_idxs: torch.Tensor,
) -> None:
    dists, idxs = knn(p1, p2, k=k, lengths1=lengths1, lengths2=lengths2)
    torch.testing.assert_close(dists, expected_dists)
    torch.testing.assert_close(idxs, expected_idxs)


def test_three_nn() -> None:
    pass


def test_k_interpolate() -> None:
    pass


@pytest.mark.parametrize(
    "points, idxs, weights, k, lengths, out_lengths",
    [
        (
            torch.randn(16, 50, 3).double(),
            torch.randint(0, 50, (16, 100, 3)),
            torch.rand(16, 100, 3).double(),
            3,
            torch.tensor([50] * 16),
            torch.tensor([100] * 16),
        ),
    ],
)
def test_k_interpolate_backward(
    points: torch.Tensor,
    idxs: torch.Tensor,
    weights: torch.Tensor,
    k: torch.Tensor,
    lengths: torch.Tensor,
    out_lengths: torch.Tensor,
) -> None:
    def grad_fn(
        points: torch.Tensor,
        idxs: torch.Tensor,
        weights: torch.Tensor,
        k: torch.Tensor,
        lengths: torch.Tensor,
        out_lengths: torch.Tensor,
    ) -> torch.Tensor:
        return k_interpolate(points, idxs, weights, k=k, lengths=lengths, out_lengths=out_lengths)

    points.requires_grad = True
    args = (points, idxs, weights, k, lengths, out_lengths)
    torch.autograd.gradcheck(grad_fn, args)


def test_three_interpolate() -> None:
    pass


@pytest.mark.parametrize(
    "points, idxs, weights, lengths, out_lengths",
    [
        (
            torch.randn(16, 50, 3).double(),
            torch.randint(0, 50, (16, 100, 3)),
            torch.rand(16, 100, 3).double(),
            torch.tensor([50] * 16),
            torch.tensor([100] * 16),
        ),
    ],
)
def test_three_interpolate_backward(
    points: torch.Tensor,
    idxs: torch.Tensor,
    weights: torch.Tensor,
    lengths: torch.Tensor,
    out_lengths: torch.Tensor,
) -> None:
    def grad_fn(
        points: torch.Tensor,
        idxs: torch.Tensor,
        weights: torch.Tensor,
        lengths: torch.Tensor,
        out_lengths: torch.Tensor,
    ) -> torch.Tensor:
        return three_interpolate(points, idxs, weights, lengths=lengths, out_lengths=out_lengths)

    points.requires_grad = True
    args = (points, idxs, weights, lengths, out_lengths)
    torch.autograd.gradcheck(grad_fn, args)


def test_knn_interpolate() -> None:
    pass


def test_three_nn_interpolate() -> None:
    pass


def test_fps() -> None:
    pass


def test_ball_query() -> None:
    pass


def test_grouping() -> None:
    pass
