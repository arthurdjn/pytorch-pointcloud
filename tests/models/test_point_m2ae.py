import pytest
import torch

from torch_pointcloud.layers import PointPatchEmbed
from torch_pointcloud.models.point_m2ae import (
    HierarchicalEncoder,
    PointM2AEClassification,
    PointM2AEMaskedAutoEncoder,
    PointM2AESegmentation,
    multi_scale_group,
)
from torch_pointcloud.utils.imports import (
    _CUDA_AVAILABLE,
    _TORCH_CLUSTER_AVAILABLE,
    _TORCH_SCATTER_AVAILABLE,
)

pytestmark = [
    pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA is not available"),
    pytest.mark.skipif(not _TORCH_CLUSTER_AVAILABLE, reason="torch_cluster is not available"),
    pytest.mark.skipif(not _TORCH_SCATTER_AVAILABLE, reason="torch_scatter is not available"),
]

NUM_GROUPS = (512, 256, 64)
GROUP_SIZES = (16, 8, 8)


def _packed(batch_size: int, num_points: int) -> tuple[torch.Tensor, torch.Tensor]:
    pos = torch.randn(batch_size * num_points, 3).cuda()
    batch = torch.arange(batch_size).repeat_interleave(num_points).cuda()
    return pos, batch


def test_multi_scale_group_shapes() -> None:
    pos, batch = _packed(2, 1024)
    neighborhoods, centers, idxs = multi_scale_group(pos, batch, NUM_GROUPS, GROUP_SIZES)
    assert [tuple(c.shape) for c in centers] == [(2, 512, 3), (2, 256, 3), (2, 64, 3)]
    assert [tuple(n.shape) for n in neighborhoods] == [(2, 512, 16, 3), (2, 256, 8, 3), (2, 64, 8, 3)]
    assert int(idxs[1].max()) < 2 * 512
    assert int(idxs[2].max()) < 2 * 256


def test_hierarchical_encoder_basic() -> None:
    model = HierarchicalEncoder(
        encoder_depths=(5, 5, 5),
        encoder_dims=(96, 192, 384),
        local_radius=(0.32, 0.64, 1.28),
        num_heads=6,
        drop_path_rate=0.1,
        with_norms=True,
    ).cuda()
    pos, batch = _packed(2, 1024)
    neighborhoods, centers, idxs = multi_scale_group(pos, batch, NUM_GROUPS, GROUP_SIZES)

    out = model(neighborhoods, centers, idxs)
    assert out.shape == (2, 64, 384)

    stages = model(neighborhoods, centers, idxs, return_stages=True)
    assert [tuple(s.shape) for s in stages] == [(2, 512, 96), (2, 256, 192), (2, 64, 384)]


def test_hierarchical_encoder_custom_token_channels() -> None:
    model = HierarchicalEncoder(
        encoder_depths=(1, 1),
        encoder_dims=(64, 128),
        local_radius=(0.32, 0.64),
        num_heads=4,
        token_local_channels=(32, 48),
        token_global_channels=(96,),
    ).cuda()
    embed_0, embed_1 = model.token_embed[0], model.token_embed[1]
    assert isinstance(embed_0, PointPatchEmbed) and isinstance(embed_1, PointPatchEmbed)
    assert embed_0.local_mlp.channel_list == [3, 32, 48]
    assert embed_0.global_mlp.channel_list == [96, 96, 64]
    assert embed_1.local_mlp.channel_list == [64, 64, 64]
    assert embed_1.global_mlp.channel_list == [128, 128, 128]

    pos, batch = _packed(2, 1024)
    neighborhoods, centers, idxs = multi_scale_group(pos, batch, (512, 128), (16, 8))
    out = model(neighborhoods, centers, idxs)
    assert out.shape == (2, 128, 128)


def test_point_m2ae_classification_modelnet() -> None:
    model = PointM2AEClassification(
        in_channels=0,
        num_classes=40,
        group_sizes=(16, 8, 8),
        num_groups=(512, 256, 64),
        encoder_depths=(5, 5, 5),
        encoder_dims=(96, 192, 384),
        local_radius=(0.32, 0.64, 1.28),
        num_heads=6,
        drop_path_rate=0.1,
        concat_pooling=False,
        dropout=0.5,
        head_channels=(256, 256),
    ).cuda()
    pos, batch = _packed(2, 1024)
    out = model(None, pos, batch)
    assert out.shape == (2, 40)


def test_point_m2ae_classification_scanobjectnn_concat_pooling() -> None:
    model = PointM2AEClassification(
        in_channels=0,
        num_classes=15,
        group_sizes=(32, 16, 16),
        num_groups=(512, 256, 64),
        encoder_depths=(5, 5, 5),
        encoder_dims=(96, 192, 384),
        local_radius=(0.32, 0.64, 1.28),
        num_heads=6,
        drop_path_rate=0.1,
        concat_pooling=True,
        dropout=0.5,
        head_channels=(256, 256),
    ).cuda()
    pos, batch = _packed(2, 2048)
    out = model(None, pos, batch)
    assert out.shape == (2, 15)


def test_point_m2ae_segmentation_basic() -> None:
    model = PointM2AESegmentation(
        in_channels=0,
        num_classes=50,
        num_categories=16,
        group_sizes=(16, 8, 8),
        num_groups=(512, 256, 64),
        encoder_depths=(5, 5, 5),
        encoder_dims=(96, 192, 384),
        local_radius=(0.32, 0.64, 1.28),
        num_heads=6,
    ).cuda()
    pos, batch = _packed(2, 2048)
    category = torch.nn.functional.one_hot(torch.tensor([0, 3]), 16).float().cuda()
    out = model(None, pos, batch, category)
    assert out.shape == (2 * 2048, 50)


def test_point_m2ae_mae_basic() -> None:
    model = PointM2AEMaskedAutoEncoder(
        in_channels=0,
        group_sizes=(16, 8, 8),
        num_groups=(512, 256, 64),
        mask_ratio=0.8,
        encoder_depths=(5, 5, 5),
        encoder_dims=(96, 192, 384),
        local_radius=(0.32, 0.64, 1.28),
        decoder_depths=(1, 1),
        decoder_dims=(384, 192),
        decoder_up_blocks=(1, 1),
        num_heads=6,
        drop_path_rate=0.1,
    ).cuda()
    pos, batch = _packed(2, 2048)
    pred, target = model(None, pos, batch)
    assert pred.ndim == target.ndim == 3
    assert pred.shape[0] == target.shape[0]
    assert pred.shape[-1] == target.shape[-1] == 3


def test_point_m2ae_classification_accepts_features() -> None:
    in_channels = 3
    model = PointM2AEClassification(
        in_channels=in_channels,
        num_classes=40,
        group_sizes=(16, 8, 8),
        num_groups=(512, 256, 64),
        encoder_depths=(1, 1, 1),
        encoder_dims=(96, 192, 384),
        local_radius=(0.32, 0.64, 1.28),
        num_heads=6,
    ).cuda()
    model.eval()
    embed = model.h_encoder.token_embed[0]
    assert isinstance(embed, PointPatchEmbed)
    assert embed.local_mlp.channel_list[0] == 3 + in_channels
    pos, batch = _packed(2, 1024)
    x_a = torch.randn(pos.size(0), in_channels).cuda()
    x_b = torch.randn(pos.size(0), in_channels).cuda()

    out_a = model(x_a, pos, batch)
    out_b = model(x_b, pos, batch)
    assert out_a.shape == (2, 40)
    assert not torch.allclose(out_a, out_b)


def test_point_m2ae_segmentation_accepts_features() -> None:
    in_channels = 3
    model = PointM2AESegmentation(
        in_channels=in_channels,
        num_classes=50,
        num_categories=16,
        group_sizes=(16, 8, 8),
        num_groups=(512, 256, 64),
        encoder_depths=(1, 1, 1),
        encoder_dims=(96, 192, 384),
        local_radius=(0.32, 0.64, 1.28),
        num_heads=6,
    ).cuda()
    model.eval()
    embed = model.h_encoder.token_embed[0]
    assert isinstance(embed, PointPatchEmbed)
    assert embed.local_mlp.channel_list[0] == 3 + in_channels
    pos, batch = _packed(2, 2048)
    category = torch.nn.functional.one_hot(torch.tensor([0, 3]), 16).float().cuda()
    x_a = torch.randn(pos.size(0), in_channels).cuda()
    x_b = torch.randn(pos.size(0), in_channels).cuda()

    out_a = model(x_a, pos, batch, category)
    out_b = model(x_b, pos, batch, category)
    assert out_a.shape == (2 * 2048, 50)
    assert not torch.allclose(out_a, out_b)
