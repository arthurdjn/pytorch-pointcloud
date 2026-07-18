import pytest
import torch

from torch_pointcloud.models.point_mae import (
    PointMAEClassification,
    PointMAEMaskedAutoEncoder,
    PointMAESegmentation,
)
from torch_pointcloud.utils.imports import (
    _CUDA_AVAILABLE,
    _TORCH_CLUSTER_AVAILABLE,
    _TORCH_SCATTER_AVAILABLE,
)

# See: https://docs.pytest.org/en/stable/how-to/skipping.html#summary
pytestmark = [
    pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA is not available"),
    pytest.mark.skipif(not _TORCH_CLUSTER_AVAILABLE, reason="torch_cluster is not available"),
    pytest.mark.skipif(not _TORCH_SCATTER_AVAILABLE, reason="torch_scatter is not available"),
]


def test_point_mae_classification_basic() -> None:
    model = PointMAEClassification(
        in_channels=0,
        num_classes=10,
        embed_dim=384,
        depth=12,
        num_heads=6,
        num_group=64,
        group_size=32,
        drop_path_rate=0.1,
        dropout=0.5,
        act="gelu",
        spatial_dim=3,
    )
    model.cuda()
    pos = torch.randn(2048, 3).cuda()
    batch = torch.cat([torch.zeros(1024), torch.ones(1024)]).long().cuda()

    out = model(None, pos, batch)
    assert out.shape == (2, 10)


def test_point_mae_segmentation_basic() -> None:
    model = PointMAESegmentation(
        in_channels=0,
        num_classes=50,
        num_categories=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        num_group=128,
        group_size=32,
        drop_path_rate=0.1,
        dropout=0.5,
        act="gelu",
        spatial_dim=3,
    )
    model.cuda()
    pos = torch.randn(4096, 3).cuda()
    batch = torch.cat([torch.zeros(2048), torch.ones(2048)]).long().cuda()
    category = torch.nn.functional.one_hot(torch.tensor([3, 7]), 16).float().cuda()

    out = model(None, pos, batch, category)
    assert out.shape == (4096, 50)


def test_point_mae_segmentation_returns_raw_logits() -> None:
    model = PointMAESegmentation(
        in_channels=0,
        num_classes=50,
        num_categories=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        num_group=128,
        group_size=32,
    )
    model.cuda()
    model.eval()
    pos = torch.randn(4096, 3).cuda()
    batch = torch.cat([torch.zeros(2048), torch.ones(2048)]).long().cuda()
    category = torch.nn.functional.one_hot(torch.tensor([3, 7]), 16).float().cuda()

    out = model(None, pos, batch, category)
    prob_mass = out.exp().sum(dim=-1)
    assert not torch.allclose(prob_mass, torch.ones_like(prob_mass), atol=1e-3)


def test_point_mae_classification_num_classes_zero_returns_features() -> None:
    model = PointMAEClassification(
        in_channels=0,
        num_classes=0,
        embed_dim=96,
        depth=2,
        num_heads=2,
        num_group=16,
        group_size=8,
    ).cuda()
    assert isinstance(model.head, torch.nn.Identity)
    pos = torch.randn(2048, 3).cuda()
    batch = torch.cat([torch.zeros(1024), torch.ones(1024)]).long().cuda()
    out = model(None, pos, batch)
    assert out.shape == (2, 2 * model.embed_dim)


def test_point_mae_masked_autoencoder_basic() -> None:
    model = PointMAEMaskedAutoEncoder(
        in_channels=0,
        embed_dim=384,
        encoder_depth=12,
        decoder_depth=4,
        num_heads=6,
        decoder_num_heads=6,
        num_group=64,
        group_size=32,
        mask_ratio=0.6,
        drop_path_rate=0.1,
        act="gelu",
        spatial_dim=3,
    )
    model.cuda()
    pos = torch.randn(2048, 3).cuda()
    batch = torch.cat([torch.zeros(1024), torch.ones(1024)]).long().cuda()
    pred, target = model(None, pos, batch)

    assert pred.ndim == target.ndim == 3
    assert pred.shape == target.shape
    assert pred.shape[1:] == (model.group_size, 3)


def test_point_mae_classification_accepts_features() -> None:
    in_channels = 3
    model = PointMAEClassification(
        in_channels=in_channels,
        num_classes=10,
        embed_dim=384,
        depth=2,
        num_heads=6,
        num_group=64,
        group_size=32,
        act="gelu",
        spatial_dim=3,
    )
    model.cuda()
    model.eval()
    assert model.encoder.local_mlp.channel_list[0] == 3 + in_channels
    pos = torch.randn(2048, 3).cuda()
    batch = torch.cat([torch.zeros(1024), torch.ones(1024)]).long().cuda()
    x_a = torch.randn(2048, in_channels).cuda()
    x_b = torch.randn(2048, in_channels).cuda()

    out_a = model(x_a, pos, batch)
    out_b = model(x_b, pos, batch)
    assert out_a.shape == (2, 10)
    assert not torch.allclose(out_a, out_b)


def test_point_mae_segmentation_accepts_features() -> None:
    in_channels = 3
    model = PointMAESegmentation(
        in_channels=in_channels,
        num_classes=50,
        num_categories=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        num_group=128,
        group_size=32,
        act="gelu",
        spatial_dim=3,
    )
    model.cuda()
    model.eval()
    assert model.encoder.local_mlp.channel_list[0] == 3 + in_channels
    pos = torch.randn(4096, 3).cuda()
    batch = torch.cat([torch.zeros(2048), torch.ones(2048)]).long().cuda()
    category = torch.nn.functional.one_hot(torch.tensor([3, 7]), 16).float().cuda()
    x_a = torch.randn(4096, in_channels).cuda()
    x_b = torch.randn(4096, in_channels).cuda()

    out_a = model(x_a, pos, batch, category)
    out_b = model(x_b, pos, batch, category)
    assert out_a.shape == (4096, 50)
    assert not torch.allclose(out_a, out_b)
