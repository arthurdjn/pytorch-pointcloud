import pytest
import torch

from torch_pointcloud.models.pointgpt import (
    PointGPTClassification,
    PointGPTGenerativePretraining,
    morton_sort,
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


def _packed_batch() -> tuple[torch.Tensor, torch.Tensor]:
    pos = torch.randn(2048, 3).cuda()
    batch = torch.cat([torch.zeros(1024), torch.ones(1024)]).long().cuda()
    return pos, batch


@pytest.mark.parametrize(
    ("embed_dim", "num_heads", "depth"),
    [
        pytest.param(384, 6, 12, id="s"),
        pytest.param(768, 12, 12, id="b"),
        pytest.param(1024, 16, 24, id="l"),
    ],
)
def test_pointgpt_classification_basic(embed_dim: int, num_heads: int, depth: int) -> None:
    model = PointGPTClassification(
        in_channels=0,
        num_classes=40,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        num_group=64,
        group_size=32,
        act="gelu",
    ).cuda()
    pos, batch = _packed_batch()
    out = model(None, pos, batch)
    assert out.shape == (2, 40)


def test_pointgpt_generative_pretraining_basic() -> None:
    model = PointGPTGenerativePretraining(
        in_channels=0,
        embed_dim=384,
        depth=12,
        decoder_depth=4,
        num_heads=6,
        num_group=64,
        group_size=32,
        act="gelu",
    ).cuda()
    pos, batch = _packed_batch()
    pred, target = model(None, pos, batch)
    assert pred.ndim == target.ndim == 3
    assert pred.shape == target.shape
    assert pred.shape[1:] == (model.group_size, 3)


def test_pointgpt_morton_sort_is_permutation() -> None:
    center = torch.randn(2, 64, 3).cuda()
    order = morton_sort(center)
    assert order.shape == (2, 64)
    for b in range(2):
        assert torch.equal(order[b].sort().values, torch.arange(64, device=order.device))
    assert (order[:, 0] == 0).all()


def test_pointgpt_classification_accepts_features() -> None:
    in_channels = 3
    model = PointGPTClassification(
        in_channels=in_channels,
        num_classes=40,
        embed_dim=384,
        depth=2,
        num_heads=6,
        num_group=64,
        group_size=32,
        act="gelu",
    ).cuda()
    model.eval()
    assert model.encoder.local_mlp.channel_list[0] == 3 + in_channels
    pos, batch = _packed_batch()
    x_a = torch.randn(pos.size(0), in_channels).cuda()
    x_b = torch.randn(pos.size(0), in_channels).cuda()
    out_a = model(x_a, pos, batch)
    out_b = model(x_b, pos, batch)
    assert out_a.shape == (2, 40)
    assert not torch.allclose(out_a, out_b)
