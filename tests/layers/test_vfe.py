import pytest
import torch
from torch_geometric.utils import scatter

from torch_pointcloud.layers.vfe import DynamicMaxVFE, DynamicMeanVFE, PillarFeatureLayer


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
    features, voxel_indices = vfe(x, pos, batch)

    pos_grid = torch.floor(pos / 0.5).int()
    rows = torch.unique(torch.cat([batch.view(-1, 1).int(), pos_grid], dim=1), sorted=True, dim=0)
    assert features.shape == (rows.shape[0], 16)
    assert torch.equal(voxel_indices, rows[:, [0, 3, 2, 1]])


def test_pillar_feature_layer_dense_and_packed_agree() -> None:
    torch.manual_seed(0)
    num_pillars, points_per_pillar = 6, 5
    dense = torch.randn(num_pillars, points_per_pillar, 4)
    packed = dense.reshape(-1, 4)
    pillar_index = torch.arange(num_pillars).repeat_interleave(points_per_pillar)

    layer = PillarFeatureLayer(4, 8, last=False).eval()
    out_dense = layer(dense)
    out_packed = layer(packed, pillar_index)
    assert out_dense.shape == (num_pillars, points_per_pillar, 8)
    assert torch.allclose(out_dense.reshape(-1, 8), out_packed)

    last = PillarFeatureLayer(4, 8, last=True).eval()
    assert last(dense).shape == (num_pillars, 1, 8)
    assert torch.allclose(last(dense).squeeze(1), last(packed, pillar_index))


def test_dynamic_max_vfe_voxels_and_inverse() -> None:
    vfe = DynamicMaxVFE(
        1,
        (8, 16),
        4,
        point_cloud_range=(0.0, 0.0, 0.0, 4.0, 4.0, 4.0),
        grid_shape=(5, 5, 5),
    ).eval()
    torch.manual_seed(0)
    pos = torch.rand(300, 3) * 6.0 - 1.0
    x = torch.randn(300, 1)
    batch = torch.cat([torch.zeros(100), torch.ones(200)]).long()
    voxel_x, voxel_indices, inverse = vfe(x, pos, batch)

    pos_grid = torch.floor(pos.clamp(0.0, 4.0)).long()
    rows = torch.unique(torch.cat([batch.view(-1, 1), pos_grid], dim=1), sorted=True, dim=0)
    assert voxel_x.shape == (rows.shape[0], 4)
    assert voxel_indices.dtype == torch.int32
    assert torch.equal(voxel_indices.long(), rows)
    assert torch.equal(voxel_indices[inverse].long(), torch.cat([batch.view(-1, 1), pos_grid], dim=1))


def test_dynamic_max_vfe_max_pools_the_point_features() -> None:
    vfe = DynamicMaxVFE(
        1,
        (8,),
        4,
        point_cloud_range=(0.0, 0.0, 0.0, 2.0, 2.0, 2.0),
        grid_shape=(3, 3, 3),
    ).eval()
    torch.manual_seed(0)
    pos = torch.rand(50, 3) * 2.0
    x = torch.randn(50, 1)
    batch = torch.zeros(50, dtype=torch.long)
    voxel_x, _, inverse = vfe(x, pos, batch)

    pos_grid = torch.floor(pos).long()
    center = pos_grid.float() + 0.5
    point_x = vfe.mlp(vfe.norm(torch.cat([pos - center, pos, x], dim=1)))
    expected = vfe.compression(scatter(point_x, inverse, dim=0, reduce="max"))
    assert torch.allclose(voxel_x, expected, atol=1e-6)


@pytest.mark.parametrize(
    "grid_shape, voxel_size",
    [
        pytest.param((5, 5, 5), (1.0, 1.0, 1.0), id="unit-voxels"),
        pytest.param((3, 9, 2), (2.0, 0.5, 4.0), id="anisotropic"),
    ],
)
def test_dynamic_max_vfe_voxel_size_spans_the_range_with_the_last_cell_at_the_bound(
    grid_shape: tuple, voxel_size: tuple
) -> None:
    vfe = DynamicMaxVFE(0, (8,), 4, point_cloud_range=(0.0, 0.0, 0.0, 4.0, 4.0, 4.0), grid_shape=grid_shape)
    assert torch.allclose(vfe.voxel_size, torch.tensor(voxel_size))
