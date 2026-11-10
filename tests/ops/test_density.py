import pytest
import torch

from torch_pointcloud.ops.density import gaussian_kernel_density


def test_gaussian_kernel_density_matches_each_sample_alone() -> None:
    torch.manual_seed(0)
    pos = torch.rand(10, 3)
    batch = torch.tensor([0] * 4 + [1] * 6)
    bandwidth = 0.5

    density = gaussian_kernel_density(pos, batch, bandwidth)

    assert density.shape == (10,)
    for sample in range(2):
        points = pos[batch == sample]
        squared_distance = torch.cdist(points, points).pow(2)
        expected = (torch.exp(-squared_distance / (2 * bandwidth**2)) / (2.5 * bandwidth)).mean(dim=1)
        assert torch.allclose(density[batch == sample], expected, atol=1e-6)


def test_gaussian_kernel_density_single_point_keeps_its_dimension() -> None:
    density = gaussian_kernel_density(torch.rand(1, 3), torch.zeros(1, dtype=torch.long), 0.5)
    assert density.shape == (1,)


def test_gaussian_kernel_density_unsorted_batch_raises() -> None:
    pos = torch.rand(6, 3)
    batch = torch.tensor([0, 1, 0, 1, 0, 1])
    with pytest.raises(ValueError, match="`batch` must be sorted"):
        gaussian_kernel_density(pos, batch, 0.5)
