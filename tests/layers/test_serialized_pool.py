from typing import List

import pytest
import torch

from torch_pointcloud.layers.serialized_pool import SerializedPool, SerializedUpsample
from torch_pointcloud.ops.serialization import serialization_depth, serialize


def test_serialized_pool_forward() -> None:
    pool = SerializedPool(
        in_channels=16,
        out_channels=32,
        stride=2,
        bias=True,
        act="relu",
        norm="batch_norm",
        act_kwargs=None,
        norm_kwargs=None,
        reduce="max",
    )
    torch.manual_seed(0)
    n = 64
    x = torch.randn(n, 16)
    pos_grid = torch.randint(0, 16, (n, 3), dtype=torch.long)
    batch = torch.zeros(n, dtype=torch.long)
    # Single serialization order, monotonically increasing codes.
    serialized_code = torch.arange(n, dtype=torch.long).unsqueeze(0)

    x_pooled, pos_grid_pooled, batch_pooled, pooled_code, cluster, pos_pooled = pool(
        x, pos_grid, batch, serialized_code, return_inverse=True
    )
    assert x_pooled.shape[1] == 32
    assert x_pooled.shape[0] == pos_grid_pooled.shape[0] == batch_pooled.shape[0]
    assert cluster.shape == (n,)
    assert pooled_code.shape[0] == 1
    assert pos_pooled is None


def test_serialized_pool_forward_pools_pos() -> None:
    pool = SerializedPool(in_channels=4, out_channels=8, stride=2)
    n = 16
    x = torch.randn(n, 4)
    pos_grid = torch.randint(0, 8, (n, 3), dtype=torch.long)
    pos = pos_grid.float()
    batch = torch.zeros(n, dtype=torch.long)
    # Stride 2 shifts codes by 3 bits, so codes 0..7 and 8..15 form two clusters of 8 points.
    serialized_code = torch.arange(n, dtype=torch.long).unsqueeze(0)

    x_pooled, _, _, _, pos_pooled = pool(x, pos_grid, batch, serialized_code, pos=pos)
    assert pos_pooled is not None
    assert pos_pooled.shape == (x_pooled.shape[0], 3)
    expected = torch.stack([pos[:8].mean(dim=0), pos[8:].mean(dim=0)])
    assert torch.allclose(pos_pooled, expected)


def test_serialized_pool_invalid_reduce_raises() -> None:
    with pytest.raises(ValueError, match="Invalid reduce"):
        SerializedPool(in_channels=4, out_channels=8, reduce="foo")  # type: ignore[arg-type]


def test_serialized_upsample_forward() -> None:
    up = SerializedUpsample(
        in_channels=32,
        skip_channels=16,
        out_channels=24,
        norm="batch_norm",
        act="relu",
        act_kwargs=None,
        norm_kwargs=None,
        bias=True,
    )
    x = torch.randn(8, 32)
    x_skip = torch.randn(20, 16)
    inverse = torch.randint(0, 8, (20,), dtype=torch.long)
    out = up(x, x_skip, inverse)
    assert out.shape == (20, 24)


@pytest.mark.parametrize(
    "extent, stride, batch_ids, num_points",
    [
        pytest.param(1, 2, [0, 1, 2], [1, 1, 1], id="depth-exhausted"),
        pytest.param(4, 2, [0, 1, 2], [1, 1, 1], id="one-cell-per-sample"),
        pytest.param(8, 2, [0, 1, 2], [8, 8, 8], id="eight-cells-per-sample"),
        pytest.param(1, 2, [0, 2], [1, 0, 1], id="no-sample-with-index-one"),
        pytest.param(6, 4, [0, 1], [1, 1], id="stride-larger-than-remaining-depth"),
        pytest.param(6, 4, [0], [1], id="single-sample"),
    ],
)
def test_serialized_pool_with_depth_never_merges_samples(
    extent: int,
    stride: int,
    batch_ids: List[int],
    num_points: List[int],
) -> None:
    torch.manual_seed(0)
    cells = torch.cartesian_prod(torch.arange(extent), torch.arange(extent), torch.arange(extent))
    pos_grid = cells.repeat(len(batch_ids), 1)
    batch = torch.tensor(batch_ids).repeat_interleave(cells.size(0))
    x = torch.randn(pos_grid.size(0), 4)
    depth = serialization_depth(pos_grid)
    serialized_code, _, _ = serialize(pos_grid, batch, orders=["z", "hilbert"], depth=depth)
    pool = SerializedPool(in_channels=4, out_channels=4, stride=stride)
    for _ in range(2):
        x, pos_grid, batch, serialized_code, _ = pool(x, pos_grid, batch, serialized_code, serialized_depth=depth)
        depth = max(depth - pool.pooling_depth, 0)

    assert batch.bincount().tolist() == num_points


def test_serialized_pool_depth_leaves_a_deep_enough_code_unchanged() -> None:
    torch.manual_seed(0)
    pos_grid = torch.randint(0, 16, (64, 3))
    batch = torch.arange(2).repeat_interleave(32)
    x = torch.randn(64, 4)
    depth = serialization_depth(pos_grid)
    serialized_code, _, _ = serialize(pos_grid, batch, orders=["z", "hilbert"], depth=depth)
    pool = SerializedPool(in_channels=4, out_channels=4, stride=2)

    expected = pool(x, pos_grid, batch, serialized_code)
    actual = pool(x, pos_grid, batch, serialized_code, serialized_depth=depth)
    for expected_tensor, actual_tensor in zip(expected[:4], actual[:4]):
        assert torch.equal(expected_tensor, actual_tensor)
