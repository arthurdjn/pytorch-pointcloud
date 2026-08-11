import pytest
import torch

from torch_pointcloud.layers.dropouts import DropPath
from torch_pointcloud.models.point_mamba import (
    PointMambaBlock,
    PointMambaClassification,
    PointMambaDecoderMAE,
    PointMambaEncoder,
    PointMambaMAE,
)
from torch_pointcloud.utils.imports import (
    _CUDA_AVAILABLE,
    _MAMBA_SSM_AVAILABLE,
    _TORCH_CLUSTER_AVAILABLE,
    _TORCH_SCATTER_AVAILABLE,
)

# See: https://docs.pytest.org/en/stable/how-to/skipping.html#summary
pytestmark = [
    pytest.mark.skipif(not _MAMBA_SSM_AVAILABLE, reason="mamba_ssm is not available"),
    pytest.mark.skipif(not _TORCH_CLUSTER_AVAILABLE, reason="torch_cluster is not available"),
    pytest.mark.skipif(not _TORCH_SCATTER_AVAILABLE, reason="torch_scatter is not available"),
]

# The mamba_ssm selective-scan kernels only run on CUDA tensors, so every test that calls a
# forward pass is gated; construction-only tests run on CPU.
requires_cuda = pytest.mark.skipif(not _CUDA_AVAILABLE, reason="mamba_ssm kernels require CUDA")


@requires_cuda
def test_point_mamba_block_uses_stochastic_depth() -> None:
    block = PointMambaBlock(16, drop_path=0.5).cuda()
    assert isinstance(block.drop_path, DropPath)

    block.train()
    x = torch.randn(64, 4, 16).cuda()
    torch.manual_seed(0)
    delta = (block(x) - x).flatten(1)
    dropped = (delta == 0).all(dim=1)
    kept = (delta != 0).all(dim=1)
    assert (dropped | kept).all()
    assert dropped.any() and kept.any()

    block.eval()
    out_a = block(x)
    out_b = block(x)
    assert torch.equal(out_a, out_b)
    assert not torch.equal(out_a, x)


def test_point_mamba_decoder_mask_token_initialized() -> None:
    decoder = PointMambaDecoderMAE(embed_dim=64, depth=1, drop_path=0.0)
    assert not torch.all(decoder.mask_token == 0)


def test_point_mamba_classification_num_classes_zero_head_is_identity() -> None:
    model = PointMambaClassification(in_channels=0, num_classes=0, embed_dim=64, depth=1)
    assert isinstance(model.head, torch.nn.Identity)


def test_point_mamba_classification_head_dropout_is_pinned() -> None:
    model = PointMambaClassification(
        in_channels=0, num_classes=10, embed_dim=64, depth=1, dropout=0.1, head_channels=(256, 256)
    )
    assert model.head.dropout == [0.5, 0.5, 0.0]


def test_point_mamba_classification_reset_classifier_keeps_config() -> None:
    model = PointMambaClassification(
        in_channels=0,
        num_classes=15,
        embed_dim=64,
        depth=1,
        dropout=0.1,
        global_pool="mean",
        head_channels=(256, 256),
    )
    pool = model.global_pool
    model.reset_classifier(7)
    assert model.global_pool is pool
    assert model.dropout == 0.1
    assert model.head.channel_list == [64, 256, 256, 7]
    model.reset_classifier(7, global_pool="max", head_channels=(), dropout=0.3)
    assert isinstance(model.global_pool, torch.nn.AdaptiveMaxPool1d)
    assert model.dropout == 0.3
    assert model.head.channel_list == [64, 7]


@requires_cuda
def test_point_mamba_encoder_basic() -> None:
    """Test the basic functionality of the PointMambaEncoder model,
    following similar architecture as the original PointMamba model."""
    model = PointMambaEncoder(
        in_channels=0,
        embed_dim=384,
        depth=12,
        num_group=64,
        group_size=32,
        drop_path=0.1,
        use_cls_token=False,
        spatial_dim=3,
        act="relu",
        act_kwargs=None,
        act_first=False,
        norm="batch_norm",
        norm_kwargs=None,
        bias=True,
    )
    model.cuda()
    pos = torch.randn(100, 3).cuda()
    batch = torch.cat([torch.zeros(40), torch.ones(60)]).long().cuda()

    out = model(None, pos, batch)
    assert out.shape == (2, 128, 384)


@requires_cuda
def test_point_mamba_classification_basic() -> None:
    """Test the basic functionality of the PointMambaClassification model,
    following similar architecture as the original PointMamba model."""
    # Specify all the parameters so that if the model's API changes this test will fail
    model = PointMambaClassification(
        in_channels=0,
        num_classes=10,
        embed_dim=384,
        depth=12,
        num_group=64,
        group_size=32,
        drop_path=0.1,
        use_cls_token=False,
        spatial_dim=3,
        act="relu",
        act_kwargs=None,
        act_first=False,
        norm="batch_norm",
        norm_kwargs=None,
        bias=True,
        dropout=0.5,
        global_pool="mean",
        head_channels=None,
    )
    model.cuda()
    pos = torch.randn(100, 3).cuda()
    batch = torch.cat([torch.zeros(40), torch.ones(60)]).long().cuda()

    out = model(None, pos, batch)
    assert out.shape == (2, 10)


@requires_cuda
def test_point_mamba_classification_accepts_features() -> None:
    in_channels = 3
    model = PointMambaClassification(
        in_channels=in_channels,
        num_classes=10,
        embed_dim=192,
        depth=2,
        num_group=64,
        group_size=32,
    )
    model.cuda()
    model.eval()
    assert model.encoder.patch_embed.local_mlp.channel_list[0] == 2 * in_channels + 3
    pos = torch.randn(2048, 3).cuda()
    batch = torch.cat([torch.zeros(1024), torch.ones(1024)]).long().cuda()
    x_a = torch.randn(2048, in_channels).cuda()
    x_b = torch.randn(2048, in_channels).cuda()

    out_a = model(x_a, pos, batch)
    out_b = model(x_b, pos, batch)
    assert out_a.shape == (2, 10)
    assert not torch.allclose(out_a, out_b)


@requires_cuda
def test_point_mamba_mae_basic() -> None:
    """Test the basic functionality of the PointMambaMAE model,
    following similar architecture as the original PointMamba model."""
    model = PointMambaMAE(
        in_channels=0,
        embed_dim=384,
        encoder_depth=12,
        decoder_depth=4,
        num_group=64,
        group_size=32,
        mask_ratio=0.6,
        drop_path=0.1,
        spatial_dim=3,
        act="relu",
        norm="batch_norm",
    )
    model.cuda()
    pos = torch.randn(100, 3).cuda()
    batch = torch.cat([torch.zeros(40), torch.ones(60)]).long().cuda()
    pred, target = model(None, pos, batch)

    assert pred.ndim == target.ndim == 3
    assert pred.shape == target.shape


@requires_cuda
def test_point_mamba_mae_accepts_features() -> None:
    in_channels = 3
    model = PointMambaMAE(
        in_channels=in_channels,
        embed_dim=192,
        encoder_depth=2,
        decoder_depth=1,
        num_group=64,
        group_size=32,
        mask_ratio=0.6,
    )
    model.cuda()
    model.eval()
    assert model.encoder.patch_embed.local_mlp.channel_list[0] == 2 * in_channels + 3
    pos = torch.randn(2048, 3).cuda()
    batch = torch.cat([torch.zeros(1024), torch.ones(1024)]).long().cuda()
    x = torch.randn(2048, in_channels).cuda()

    pred, target = model(x, pos, batch)
    assert pred.ndim == target.ndim == 3
    assert pred.shape == target.shape
