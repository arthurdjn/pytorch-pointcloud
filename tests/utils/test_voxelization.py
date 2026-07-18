import pytest
import torch

from torch_pointcloud.utils.imports import _SPCONV_AVAILABLE, _TORCH_CLUSTER_AVAILABLE
from torch_pointcloud.utils.voxelization import (
    _point_to_voxel_generator,
    dense_voxelize,
    hard_voxelize,
    sparse_voxelize,
    trilinear_dense_devoxelize,
)


def test_dense_voxelize_mean_known_values() -> None:
    x = torch.tensor([[1.0], [3.0], [5.0]])
    pos = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    batch = torch.zeros(3, dtype=torch.long)
    out = dense_voxelize(x, pos, batch, resolution=2, reduce="mean")
    assert out.shape == (1, 1, 2, 2, 2)  # (B, C, x, y, z)
    assert out[0, 0, 0, 0, 0].item() == pytest.approx(2.0)
    assert out[0, 0, 1, 0, 0].item() == pytest.approx(5.0)
    assert out.sum().item() == pytest.approx(7.0)


def test_dense_voxelize_separates_batches() -> None:
    x = torch.tensor([[1.0], [5.0]])
    pos = torch.zeros(2, 3)
    batch = torch.tensor([0, 1])
    out = dense_voxelize(x, pos, batch, resolution=2, reduce="sum")
    assert out.shape == (2, 1, 2, 2, 2)
    assert out[0, 0, 0, 0, 0].item() == pytest.approx(1.0)
    assert out[1, 0, 0, 0, 0].item() == pytest.approx(5.0)


def test_dense_voxelize_rejects_non_3d_positions() -> None:
    with pytest.raises(ValueError, match="must be 3D"):
        dense_voxelize(torch.zeros(2, 1), torch.zeros(2, 2), torch.zeros(2, dtype=torch.long), resolution=2)


def test_trilinear_dense_devoxelize_at_grid_points() -> None:
    x_voxel = torch.arange(8.0).view(1, 1, 2, 2, 2)  # value at (x, y, z) = 4x + 2y + z
    pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    batch = torch.zeros(3, dtype=torch.long)
    out = trilinear_dense_devoxelize(x_voxel, pos, batch, resolution=2)
    assert out.shape == (3, 1)
    assert out[:, 0].tolist() == [0.0, 4.0, 7.0]


def test_trilinear_dense_devoxelize_midpoint_averages_neighbors() -> None:
    x_voxel = torch.arange(8.0).view(1, 1, 2, 2, 2)
    pos = torch.tensor([[0.5, 0.0, 0.0]])
    batch = torch.zeros(1, dtype=torch.long)
    out = trilinear_dense_devoxelize(x_voxel, pos, batch, resolution=2)
    assert out.item() == pytest.approx(2.0)  # midpoint of voxels at x=0 (value 0) and x=1 (value 4)


def test_trilinear_devoxelize_inverts_dense_voxelize_at_grid_points() -> None:
    x = torch.tensor([[1.0], [3.0], [5.0]])
    pos = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    batch = torch.zeros(3, dtype=torch.long)
    x_voxel = dense_voxelize(x, pos, batch, resolution=2)
    out = trilinear_dense_devoxelize(x_voxel, pos, batch, resolution=2)
    assert out[:, 0].tolist() == [2.0, 2.0, 5.0]


@pytest.mark.skipif(not _TORCH_CLUSTER_AVAILABLE, reason="torch_cluster is not available")
def test_sparse_voxelize_mean_known_values() -> None:
    # Exact binary fractions: the first two points share the voxel at grid (0, 0, 0), the third is at (2, 0, 0).
    x = torch.tensor([[2.0], [4.0], [10.0]])
    pos = torch.tensor([[0.0625, 0.0625, 0.0625], [0.125, 0.0625, 0.0625], [0.5625, 0.0625, 0.0625]])
    batch = torch.zeros(3, dtype=torch.long)
    x_voxel, pos_voxel, batch_voxel, inverse = sparse_voxelize(
        x, pos, batch, voxel_size=0.25, reduce="mean", return_inverse=True
    )
    assert x_voxel.shape == (2, 1)
    assert pos_voxel.dtype == torch.int32
    assert batch_voxel.tolist() == [0, 0]
    assert torch.equal(x_voxel[inverse], torch.tensor([[3.0], [3.0], [10.0]]))
    order = pos_voxel[:, 0].argsort()
    assert pos_voxel[order].tolist() == [[0, 0, 0], [2, 0, 0]]


@pytest.mark.skipif(not _TORCH_CLUSTER_AVAILABLE, reason="torch_cluster is not available")
def test_sparse_voxelize_separates_batches() -> None:
    x = torch.tensor([[1.0], [5.0]])
    pos = torch.zeros(2, 3)
    batch = torch.tensor([0, 1])
    x_voxel, pos_voxel, batch_voxel = sparse_voxelize(x, pos, batch, voxel_size=0.25)
    assert x_voxel.shape == (2, 1)
    order = batch_voxel.argsort()
    assert batch_voxel[order].tolist() == [0, 1]
    assert x_voxel[order].tolist() == [[1.0], [5.0]]


def test_point_to_voxel_generator_cache_is_bounded() -> None:
    assert _point_to_voxel_generator.cache_info().maxsize == 8


@pytest.mark.skipif(not _SPCONV_AVAILABLE, reason="spconv is not installed")
def test_hard_voxelize_known_coords() -> None:
    points = torch.tensor([[0.25, 0.25, 0.25], [0.75, 0.25, 0.25]])
    batch = torch.zeros(2, dtype=torch.long)
    voxels, coords, num_points = hard_voxelize(
        points, batch, (0.5, 0.5, 0.5), (0, 0, 0, 1, 1, 1), max_num_points=5, max_num_voxels=10
    )
    assert voxels.shape == (2, 5, 3)
    assert coords.tolist() == [[0, 0, 0, 0], [0, 0, 0, 1]]  # (batch, z, y, x)
    assert num_points.tolist() == [1, 1]
    assert voxels[0, 0].tolist() == [0.25, 0.25, 0.25]
    assert voxels[1, 0].tolist() == [0.75, 0.25, 0.25]


@pytest.mark.skipif(not _SPCONV_AVAILABLE, reason="spconv is not installed")
def test_hard_voxelize_batch_column() -> None:
    points = torch.tensor([[0.25, 0.25, 0.25], [0.25, 0.25, 0.25]])
    batch = torch.tensor([0, 1])
    voxels, coords, num_points = hard_voxelize(
        points, batch, (0.5, 0.5, 0.5), (0, 0, 0, 1, 1, 1), max_num_points=5, max_num_voxels=10
    )
    assert coords.shape == (2, 4)
    assert coords[:, 0].tolist() == [0, 1]
    assert num_points.tolist() == [1, 1]


@pytest.mark.skipif(not _SPCONV_AVAILABLE, reason="spconv is not installed")
def test_hard_voxelize_empty_input_returns_empty_outputs() -> None:
    voxels, coords, num_points = hard_voxelize(
        torch.zeros(0, 4),
        torch.zeros(0, dtype=torch.long),
        (0.1, 0.1, 0.1),
        (0, 0, 0, 1, 1, 1),
        max_num_points=5,
        max_num_voxels=10,
    )
    assert voxels.shape == (0, 5, 4)
    assert coords.shape == (0, 4)
    assert num_points.shape == (0,)
    assert coords.dtype == torch.int32
    assert num_points.dtype == torch.int32
