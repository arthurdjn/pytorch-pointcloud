import torch

from torch_pointcloud.layers.vfe import DynamicMeanVFE, PillarFeatureLayer


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
