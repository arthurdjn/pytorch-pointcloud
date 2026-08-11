import pytest
import torch
from torch_geometric.nn import MLP

from torch_pointcloud.models.pointgpt import (
    PointGPTClassification,
    PointGPTExtractor,
    PointGPTGenerativePretraining,
    morton_sort,
)
from torch_pointcloud.utils.imports import (
    _CUDA_AVAILABLE,
    _TORCH_CLUSTER_AVAILABLE,
    _TORCH_SCATTER_AVAILABLE,
)

pytestmark = [
    pytest.mark.skipif(not _TORCH_CLUSTER_AVAILABLE, reason="torch_cluster is not available"),
    pytest.mark.skipif(not _TORCH_SCATTER_AVAILABLE, reason="torch_scatter is not available"),
]


def _packed_batch() -> tuple[torch.Tensor, torch.Tensor]:
    pos = torch.randn(2048, 3)
    batch = torch.cat([torch.zeros(1024), torch.ones(1024)]).long()
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
    )
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
    )
    pos, batch = _packed_batch()
    pred, target = model(None, pos, batch)
    assert pred.ndim == target.ndim == 3
    assert pred.shape == target.shape
    assert pred.shape[1:] == (model.group_size, 3)


def test_pointgpt_classification_num_classes_zero_returns_features() -> None:
    model = PointGPTClassification(
        in_channels=0,
        num_classes=0,
        embed_dim=96,
        depth=2,
        num_heads=2,
        num_group=16,
        group_size=8,
        act="gelu",
    )
    assert isinstance(model.head, torch.nn.Identity)
    pos, batch = _packed_batch()
    out = model(None, pos, batch)
    assert out.shape == (2, 2 * model.embed_dim)


def test_pointgpt_classification_reset_classifier_keeps_global_pool() -> None:
    model = PointGPTClassification(
        in_channels=0,
        num_classes=40,
        embed_dim=96,
        depth=2,
        num_heads=2,
        num_group=16,
        group_size=8,
        global_pool="mean",
    )
    pool = model.global_pool
    model.reset_classifier(10)
    assert model.global_pool is pool
    assert isinstance(model.head, MLP)
    assert model.head.channel_list[-1] == 10
    model.reset_classifier(10, global_pool="max")
    assert isinstance(model.global_pool, torch.nn.AdaptiveMaxPool1d)


def test_pointgpt_generative_pretraining_duplicate_centers_finite() -> None:
    model = PointGPTGenerativePretraining(
        in_channels=0,
        embed_dim=96,
        depth=2,
        decoder_depth=1,
        num_heads=2,
        num_group=8,
        group_size=4,
        mask_ratio=0.5,
        keep_attend=2,
        act="gelu",
    )
    model.eval()
    pos = torch.zeros(512, 3)
    batch = torch.cat([torch.zeros(256), torch.ones(256)]).long()
    pred, target = model(None, pos, batch)
    assert torch.isfinite(pred).all()
    assert torch.isfinite(target).all()


@pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA is not available")
def test_pointgpt_extractor_half_precision() -> None:
    extractor = PointGPTExtractor(embed_dim=32, num_heads=2, depth=1).cuda().half()
    tokens = torch.randn(2, 5, 32).cuda().half()
    pos = torch.randn(2, 6, 32).cuda().half()
    mask = torch.triu(torch.ones(6, 6, dtype=torch.bool, device="cuda"), diagonal=1)
    out = extractor(tokens, pos, mask)
    assert out.dtype == torch.float16
    assert out.shape == (2, 6, 32)


def test_pointgpt_morton_sort_is_permutation() -> None:
    center = torch.randn(2, 64, 3)
    order = morton_sort(center)
    assert order.shape == (2, 64)
    for b in range(2):
        assert torch.equal(order[b].sort().values, torch.arange(64, device=order.device))
    assert (order[:, 0] == 0).all()


def test_pointgpt_generative_pretraining_accepts_features() -> None:
    in_channels = 3
    model = PointGPTGenerativePretraining(
        in_channels=in_channels,
        embed_dim=384,
        depth=2,
        decoder_depth=1,
        num_heads=6,
        num_group=64,
        group_size=32,
        act="gelu",
    )
    model.eval()
    assert model.encoder.local_mlp.channel_list[0] == 3 + in_channels
    pos, batch = _packed_batch()
    x_a = torch.randn(pos.size(0), in_channels)
    x_b = torch.randn(pos.size(0), in_channels)

    pred_a, target_a = model(x_a, pos, batch)
    pred_b, _ = model(x_b, pos, batch)
    assert pred_a.shape == target_a.shape
    assert pred_a.shape == pred_b.shape
    assert not torch.allclose(pred_a, pred_b)


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
    )
    model.eval()
    assert model.encoder.local_mlp.channel_list[0] == 3 + in_channels
    pos, batch = _packed_batch()
    x_a = torch.randn(pos.size(0), in_channels)
    x_b = torch.randn(pos.size(0), in_channels)
    out_a = model(x_a, pos, batch)
    out_b = model(x_b, pos, batch)
    assert out_a.shape == (2, 40)
    assert not torch.allclose(out_a, out_b)
