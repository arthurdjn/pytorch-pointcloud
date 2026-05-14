import pytest
import torch

from torch_pointcloud.models.point_mamba import PointMambaClassification, PointMambaEncoder, PointMambaMAE
from torch_pointcloud.utils.imports import (
    _CUDA_AVAILABLE,
    _MAMBA_SSM_AVAILABLE,
    _TORCH_CLUSTER_AVAILABLE,
    _TORCH_SCATTER_AVAILABLE,
)

pytestmark = [
    pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA is not available"),
    pytest.mark.skipif(not _MAMBA_SSM_AVAILABLE, reason="mamba_ssm is not available"),
    pytest.mark.skipif(not _TORCH_CLUSTER_AVAILABLE, reason="torch_cluster is not available"),
    pytest.mark.skipif(not _TORCH_SCATTER_AVAILABLE, reason="torch_scatter is not available"),
]


def test_point_mamba_encoder_basic() -> None:
    """Test the basic functionality of the PointMambaEncoder model,
    following similar architecture as the original PointMamba model."""
    model = PointMambaEncoder(
        in_channels=0,
        embedding_dim=384,
        depth=12,
        num_patches=64,
        group_size=32,
        drop_path_rate=0.1,
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


def test_point_mamba_classification_basic() -> None:
    """Test the basic functionality of the PointMambaClassification model,
    following similar architecture as the original PointMamba model."""
    # Specify all the parameters so that if the model's API changes this test will fail
    model = PointMambaClassification(
        in_channels=0,
        num_classes=10,
        embedding_dim=384,
        depth=12,
        num_patches=64,
        group_size=32,
        drop_path_rate=0.1,
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


def test_point_mamba_mae_basic() -> None:
    """Test the basic functionality of the PointMambaMAE model,
    following similar architecture as the original PointMamba model."""
    model = PointMambaMAE(
        in_channels=0,
        embedding_dim=384,
        encoder_depth=12,
        decoder_depth=4,
        num_patches=64,
        group_size=32,
        mask_ratio=0.6,
        drop_path_rate=0.1,
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
