import pytest
import torch

from torch_pointcloud.layers.vfe import DynamicMeanVFE
from torch_pointcloud.utils.imports import _TORCH_SCATTER_AVAILABLE

# See: https://docs.pytest.org/en/stable/how-to/skipping.html#summary
pytestmark = pytest.mark.skipif(
    not _TORCH_SCATTER_AVAILABLE,
    reason="torch-scatter is not installed",
)


def test_dynamic_mean_vfe_forward() -> None:
    vfe = DynamicMeanVFE(
        in_channels=4,
        num_filters=[16, 16],
        voxel_size=[0.5, 0.5, 0.5],
        point_cloud_range=[0.0, 0.0, 0.0, 4.0, 4.0, 4.0],
        grid_size=[8, 8, 8],
    )
    vfe.eval()
    torch.manual_seed(0)
    pos = torch.rand(200, 3) * 4.0
    x = torch.randn(200, 1)
    batch = torch.cat([torch.zeros(80), torch.ones(120)]).long()
    features, voxel_indices = vfe(pos, x, batch)

    coords = torch.floor(pos / 0.5).int()
    rows = torch.unique(torch.cat([batch.view(-1, 1).int(), coords], dim=1), sorted=True, dim=0)
    assert features.shape == (rows.shape[0], 16)
    assert torch.equal(voxel_indices, rows[:, [0, 3, 2, 1]])
