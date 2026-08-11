import pytest
import torch
from torch_geometric.nn import MLP

from torch_pointcloud.models.point_mae import (
    PointMAEClassification,
    PointMAEMaskedAutoEncoder,
    PointMAESegmentation,
)
from torch_pointcloud.utils.imports import (
    _TORCH_CLUSTER_AVAILABLE,
    _TORCH_SCATTER_AVAILABLE,
)

# See: https://docs.pytest.org/en/stable/how-to/skipping.html#summary
pytestmark = [
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
        drop_path=0.1,
        dropout=0.5,
        act="gelu",
        spatial_dim=3,
    )
    pos = torch.randn(2048, 3)
    batch = torch.cat([torch.zeros(1024), torch.ones(1024)]).long()

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
        drop_path=0.1,
        dropout=0.5,
        act="gelu",
        spatial_dim=3,
    )
    pos = torch.randn(4096, 3)
    batch = torch.cat([torch.zeros(2048), torch.ones(2048)]).long()
    category = torch.nn.functional.one_hot(torch.tensor([3, 7]), 16).float()

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
    model.eval()
    pos = torch.randn(4096, 3)
    batch = torch.cat([torch.zeros(2048), torch.ones(2048)]).long()
    category = torch.nn.functional.one_hot(torch.tensor([3, 7]), 16).float()

    out = model(None, pos, batch, category)
    prob_mass = out.exp().sum(dim=-1)
    assert not torch.allclose(prob_mass, torch.ones_like(prob_mass), atol=1e-3)


def test_point_mae_segmentation_head_width_follows_embed_dim() -> None:
    model = PointMAESegmentation(
        in_channels=0,
        num_classes=5,
        num_categories=16,
        embed_dim=48,
        depth=12,
        num_heads=2,
        num_group=8,
        group_size=4,
    )
    model.eval()
    assert isinstance(model.head, MLP)
    assert model.head.channel_list[0] == 1024 + 3 * 48 * 2 + 64
    pos = torch.randn(256, 3)
    batch = torch.cat([torch.zeros(128), torch.ones(128)]).long()
    category = torch.nn.functional.one_hot(torch.tensor([3, 7]), 16).float()

    out = model(None, pos, batch, category)
    assert out.shape == (256, 5)


def test_point_mae_segmentation_ragged_batch_raises() -> None:
    model = PointMAESegmentation(in_channels=0, num_classes=5, num_categories=16, num_group=8, group_size=4)
    model.eval()
    pos = torch.randn(128, 3)
    batch = torch.cat([torch.zeros(96), torch.ones(32)]).long()
    category = torch.nn.functional.one_hot(torch.tensor([3, 7]), 16).float()

    with pytest.raises(ValueError, match="same number of points per sample"):
        model(None, pos, batch, category)


def test_point_mae_classification_num_classes_zero_returns_features() -> None:
    model = PointMAEClassification(
        in_channels=0,
        num_classes=0,
        embed_dim=96,
        depth=2,
        num_heads=2,
        num_group=16,
        group_size=8,
    )
    assert isinstance(model.head, torch.nn.Identity)
    pos = torch.randn(2048, 3)
    batch = torch.cat([torch.zeros(1024), torch.ones(1024)]).long()
    out = model(None, pos, batch)
    assert out.shape == (2, 2 * model.embed_dim)


def test_point_mae_classification_reset_classifier_rejects_global_pool() -> None:
    model = PointMAEClassification(in_channels=0, num_classes=10, embed_dim=96, depth=2, num_heads=2)
    model.reset_classifier(5)
    assert isinstance(model.head, MLP)
    assert model.head.channel_list[-1] == 5
    with pytest.raises(ValueError, match="global_pool"):
        model.reset_classifier(5, global_pool="mean")


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
        drop_path=0.1,
        act="gelu",
        spatial_dim=3,
    )
    pos = torch.randn(2048, 3)
    batch = torch.cat([torch.zeros(1024), torch.ones(1024)]).long()
    pred, target = model(None, pos, batch)

    assert pred.ndim == target.ndim == 3
    assert pred.shape == target.shape
    assert pred.shape[1:] == (model.group_size, 3)


@pytest.mark.parametrize(
    "mask_ratio",
    [
        pytest.param(0.0, id="zero"),
        pytest.param(1.0, id="one"),
        pytest.param(-0.1, id="negative"),
    ],
)
def test_point_mae_masked_autoencoder_invalid_mask_ratio_raises(mask_ratio: float) -> None:
    with pytest.raises(ValueError, match="mask_ratio"):
        PointMAEMaskedAutoEncoder(
            in_channels=0,
            embed_dim=48,
            encoder_depth=1,
            decoder_depth=1,
            num_heads=2,
            decoder_num_heads=2,
            num_group=8,
            group_size=4,
            mask_ratio=mask_ratio,
        )


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
    model.eval()
    assert model.encoder.local_mlp.channel_list[0] == 3 + in_channels
    pos = torch.randn(2048, 3)
    batch = torch.cat([torch.zeros(1024), torch.ones(1024)]).long()
    x_a = torch.randn(2048, in_channels)
    x_b = torch.randn(2048, in_channels)

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
    model.eval()
    assert model.encoder.local_mlp.channel_list[0] == 3 + in_channels
    pos = torch.randn(4096, 3)
    batch = torch.cat([torch.zeros(2048), torch.ones(2048)]).long()
    category = torch.nn.functional.one_hot(torch.tensor([3, 7]), 16).float()
    x_a = torch.randn(4096, in_channels)
    x_b = torch.randn(4096, in_channels)

    out_a = model(x_a, pos, batch, category)
    out_b = model(x_b, pos, batch, category)
    assert out_a.shape == (4096, 50)
    assert not torch.allclose(out_a, out_b)
