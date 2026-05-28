import torch

from torch_pointcloud.utils.ops import voxel_grid_fnv


def _two_voxel_cloud() -> torch.Tensor:
    """Six points: 4 in voxel A=[0,1)^3 and 2 in voxel B=[1,2)x[0,1)^2."""
    return torch.tensor(
        [
            [0.1, 0.1, 0.1],
            [0.2, 0.2, 0.2],
            [0.3, 0.3, 0.3],
            [0.4, 0.4, 0.4],
            [1.1, 0.1, 0.1],
            [1.2, 0.2, 0.2],
        ]
    )


def test_voxel_grid_fnv_default_returns_only_hashed() -> None:
    """Without flags the return is a single $(N,)$ hash tensor (back-compat path)."""
    pos = _two_voxel_cloud()
    hashed = voxel_grid_fnv(pos, size=1.0)
    assert isinstance(hashed, torch.Tensor)
    assert hashed.shape == (6,)


def test_voxel_grid_fnv_inverse_marks_each_point_with_its_voxel_id() -> None:
    """`return_inverse=True` returns `(hashed, inverse)` where inverse is $(N,)$ consecutive voxel IDs."""
    pos = _two_voxel_cloud()
    hashed, inverse = voxel_grid_fnv(pos, size=1.0, return_inverse=True)
    assert hashed.shape == (6,)
    assert inverse.shape == (6,)
    # Points 0-3 share a voxel, points 4-5 share another; the two groups must differ in inverse.
    assert torch.equal(inverse[:4], inverse[:4][0].repeat(4))
    assert torch.equal(inverse[4:], inverse[4:][0].repeat(2))
    assert inverse[0].item() != inverse[4].item()


def test_voxel_grid_fnv_counts_returns_per_voxel_population() -> None:
    """`return_counts=True` returns `(hashed, count)` where `count` is $(V,)$ per-voxel sizes."""
    pos = _two_voxel_cloud()
    hashed, count = voxel_grid_fnv(pos, size=1.0, return_counts=True)
    assert hashed.shape == (6,)
    assert count.shape == (2,)
    assert sorted(count.tolist()) == [2, 4]


def test_voxel_grid_fnv_inverse_and_counts_return_triple() -> None:
    """Both flags enabled: returns `(hashed, inverse, count)` consistent with each other."""
    pos = _two_voxel_cloud()
    hashed, inverse, count = voxel_grid_fnv(pos, size=1.0, return_inverse=True, return_counts=True)
    assert hashed.shape == (6,) and inverse.shape == (6,) and count.shape == (2,)
    # `count[v]` must match how often `inverse` references voxel `v`.
    assert torch.equal(torch.bincount(inverse), count)
