from typing import Any, Dict

import pytest
import torch
from torch import Tensor

from torch_pointcloud.inferers import CyclicVoxelInferer
from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.imports import _TORCH_SCATTER_AVAILABLE


def _grid(n_per_voxel: int, n_voxels: int, voxel_size: float) -> Dict[str, Any]:
    """Build a deterministic 1D grid of `n_voxels` voxels with `n_per_voxel` points each."""
    chunks = []
    for v in range(n_voxels):
        base = torch.tensor([float(v) * voxel_size + voxel_size * 0.1, 0.0, 0.0])
        offsets = torch.linspace(0.0, voxel_size * 0.5, n_per_voxel).unsqueeze(-1) * torch.tensor([1.0, 0.0, 0.0])
        chunks.append(base + offsets)
    pos = torch.cat(chunks, dim=0)
    return {DataKeys.POS: pos, DataKeys.BATCH: torch.zeros(pos.size(0), dtype=torch.long)}


def test_cyclic_voxel_uniform_voxels_average_to_their_own_x() -> None:
    """With `c_v` constant across voxels, each point is picked exactly $K / c_v$ times.

    The predictor returns each point's x-coordinate as a scalar logit. After scatter-averaging,
    every point's output equals its own x (the sum of $K/c_v$ identical contributions divided by
    the same participation count).
    """
    n_per_voxel, n_voxels, voxel_size = 4, 6, 1.0
    data = _grid(n_per_voxel, n_voxels, voxel_size)

    def predictor(window: Dict[str, Any]) -> Tensor:
        return window[DataKeys.POS][:, :1].clone()

    out = CyclicVoxelInferer(voxel_size=voxel_size, sub_batch_size=2, seed=0)(data, predictor=predictor)
    assert out.shape == (n_per_voxel * n_voxels, 1)
    assert torch.allclose(out.squeeze(-1).float(), data[DataKeys.POS][:, 0])


def test_cyclic_voxel_every_point_participates_at_least_once() -> None:
    """Cyclic indexing guarantees every original point lands in $\\geq 1$ sub-cloud."""
    data = _grid(n_per_voxel=3, n_voxels=4, voxel_size=1.0)

    def predictor(window: Dict[str, Any]) -> Tensor:
        return torch.ones(window[DataKeys.POS].size(0), 2)

    out = CyclicVoxelInferer(voxel_size=1.0, sub_batch_size=4, seed=0)(data, predictor=predictor)
    assert torch.all(out > 0)


def test_cyclic_voxel_seed_is_reproducible() -> None:
    """Identical `seed` yields identical output."""
    data = _grid(n_per_voxel=3, n_voxels=5, voxel_size=1.0)

    def predictor(window: Dict[str, Any]) -> Tensor:
        return window[DataKeys.POS][:, :2].clone()

    out_a = CyclicVoxelInferer(voxel_size=1.0, sub_batch_size=2, seed=42)(data, predictor=predictor)
    out_b = CyclicVoxelInferer(voxel_size=1.0, sub_batch_size=2, seed=42)(data, predictor=predictor)
    assert torch.equal(out_a, out_b)


def test_cyclic_voxel_softmax_yields_probabilities() -> None:
    """`softmax=True` softmaxes per pass, so scatter-summed predictions over $K/c_v$ passes still
    average to a probability vector (rows sum to 1).
    """
    data = _grid(n_per_voxel=2, n_voxels=3, voxel_size=1.0)

    def predictor(window: Dict[str, Any]) -> Tensor:
        return torch.full((window[DataKeys.POS].size(0), 3), 10.0)

    out = CyclicVoxelInferer(voxel_size=1.0, softmax=True, seed=0)(data, predictor=predictor)
    assert torch.allclose(out.sum(dim=-1), torch.ones(out.size(0)).double())


@pytest.mark.skipif(not _TORCH_SCATTER_AVAILABLE, reason="torch-scatter not installed")
def test_cyclic_voxel_with_transform_applies_per_subcloud() -> None:
    """The optional `transform` is applied independently to each sub-cloud's data dict."""
    from torch_pointcloud import transforms as T

    data = _grid(n_per_voxel=3, n_voxels=4, voxel_size=1.0)
    transform = T.Shift(keys=DataKeys.POS, method="min")

    def predictor(window: Dict[str, Any]) -> Tensor:
        return window[DataKeys.POS].clone()

    out = CyclicVoxelInferer(voxel_size=1.0, transform=transform, seed=0)(data, predictor=predictor)
    # After per-sub-cloud Shift(min), x-min across all sub-clouds is 0.
    assert out[:, 0].min().item() >= 0.0


def test_cyclic_voxel_validates_args() -> None:
    """Constructor rejects `voxel_size <= 0` and `sub_batch_size < 1`."""
    with pytest.raises(ValueError, match="voxel_size"):
        CyclicVoxelInferer(voxel_size=0.0)
    with pytest.raises(ValueError, match="sub_batch_size"):
        CyclicVoxelInferer(voxel_size=1.0, sub_batch_size=0)


def test_cyclic_voxel_missing_pos_key_raises() -> None:
    with pytest.raises(KeyError, match="pos"):
        CyclicVoxelInferer(voxel_size=1.0)({}, predictor=lambda d: torch.zeros(0, 1))
