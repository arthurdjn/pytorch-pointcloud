from typing import Literal

import pytest
import torch

from torch_pointcloud.layers.voxel_grid_pool import VoxelGridPool
from torch_pointcloud.utils.imports import _PYG_LIB_AVAILABLE


@pytest.mark.skipif(not _PYG_LIB_AVAILABLE, reason="pyg-lib is not installed")
@pytest.mark.parametrize("origin", ["grid", "min"])
def test_voxel_grid_pool_forward(origin: Literal["grid", "min"]) -> None:
    torch.manual_seed(0)
    x = torch.randn(200, 8)
    pos = torch.rand(200, 3) * 4.0
    batch = torch.cat([torch.zeros(80), torch.ones(120)]).long()

    pool = VoxelGridPool(grid_size=1.0, reduce="max", origin=origin)
    x_pooled, pos_pooled, batch_pooled, inverse = pool(x, pos, batch, return_inverse=True)

    assert x_pooled.shape[1] == 8 and pos_pooled.shape[1] == 3
    assert x_pooled.shape[0] == pos_pooled.shape[0] == batch_pooled.shape[0] == int(inverse.max()) + 1
    assert torch.equal(batch_pooled, batch_pooled.sort().values)
    assert torch.equal(batch_pooled[inverse], batch)
    expected = torch.zeros_like(x_pooled).scatter_reduce(
        0, inverse[:, None].expand_as(x), x, "amax", include_self=False
    )
    assert torch.allclose(x_pooled, expected)
    assert (pos_pooled.max(dim=0).values <= 4.0).all() and (pos_pooled.min(dim=0).values >= 0.0).all()


@pytest.mark.skipif(not _PYG_LIB_AVAILABLE, reason="pyg-lib is not installed")
def test_voxel_grid_pool_grid_origin_is_independent_of_other_clouds() -> None:
    torch.manual_seed(0)
    pos = torch.randint(0, 300, (2000, 3)).float() / 100.0
    x = torch.randn(2000, 4)
    batch = torch.zeros(2000, dtype=torch.long)
    pool = VoxelGridPool(grid_size=0.06)
    _, pos_alone, _, inverse_alone = pool(x, pos, batch, return_inverse=True)

    other = torch.rand(100, 3) - 3.37
    _, pos_batched, _, inverse_batched = pool(
        torch.cat([x, torch.randn(100, 4)]),
        torch.cat([pos, other]),
        torch.cat([batch, torch.ones(100, dtype=torch.long)]),
        return_inverse=True,
    )

    assert torch.allclose(pos_batched[inverse_batched[:2000]], pos_alone[inverse_alone], atol=1e-6)


def test_voxel_grid_pool_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError, match="reduce"):
        VoxelGridPool(grid_size=1.0, reduce="median")
    with pytest.raises(ValueError, match="origin"):
        VoxelGridPool(grid_size=1.0, origin="center")  # type: ignore[arg-type]


@pytest.mark.skipif(not _PYG_LIB_AVAILABLE, reason="pyg-lib is not installed")
@pytest.mark.parametrize(
    "origin",
    [
        pytest.param("grid", id="grid"),
        pytest.param("min", id="min"),
    ],
)
def test_voxel_grid_pool_unsorted_batch_matches_sorted(origin: Literal["grid", "min"]) -> None:
    torch.manual_seed(0)
    x = torch.randn(40, 5)
    pos = torch.rand(40, 3) * 4.0
    pos[20:] += 100.0
    batch = torch.cat([torch.zeros(20), torch.ones(20)]).long()
    perm = torch.randperm(40)

    pool = VoxelGridPool(grid_size=1.0, reduce="mean", origin=origin)
    x_pooled, pos_pooled, _ = pool(x, pos, batch)
    x_shuffled, pos_shuffled, _ = pool(x[perm], pos[perm], batch[perm])

    cells = torch.cat([pos_pooled, x_pooled], dim=1)
    cells_shuffled = torch.cat([pos_shuffled, x_shuffled], dim=1)
    assert cells.shape == cells_shuffled.shape
    assert torch.allclose(cells[cells[:, 0].argsort()], cells_shuffled[cells_shuffled[:, 0].argsort()], atol=1e-5)
