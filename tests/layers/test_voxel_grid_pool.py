import pytest
import torch

from torch_pointcloud.layers.voxel_grid_pool import VoxelGridPool


@pytest.mark.parametrize("origin", ["grid", "min"])
def test_voxel_grid_pool_forward(origin: str) -> None:
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


def test_voxel_grid_pool_grid_origin_is_extent_independent() -> None:
    torch.manual_seed(0)
    pos = torch.rand(100, 3)
    batch = torch.zeros(100, dtype=torch.long)
    pool = VoxelGridPool(grid_size=0.25)
    _, _, _, inverse = pool(pos, pos, batch, return_inverse=True)
    _, _, _, inverse_shifted = pool(
        pos, torch.cat([pos, pos.new_tensor([[0.9, 0.9, 0.9]])]).roll(1, 0)[1:], batch, return_inverse=True
    )
    assert torch.equal(inverse, inverse_shifted)


def test_voxel_grid_pool_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError, match="reduce"):
        VoxelGridPool(grid_size=1.0, reduce="median")
    with pytest.raises(ValueError, match="origin"):
        VoxelGridPool(grid_size=1.0, origin="center")  # type: ignore[arg-type]
