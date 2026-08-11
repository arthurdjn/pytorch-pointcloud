import pytest
import torch

from torch_pointcloud.layers.serialized_pool import SerializedPool, SerializedUpsample
from torch_pointcloud.utils.imports import _TORCH_SCATTER_AVAILABLE

# See: https://docs.pytest.org/en/stable/how-to/skipping.html#summary
pytestmark = pytest.mark.skipif(
    not _TORCH_SCATTER_AVAILABLE,
    reason="torch-scatter is not installed",
)


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
