from typing import Any, Dict, List

import pytest
import torch
from torch_geometric.nn import MLP

from torch_pointcloud.models import model_info
from torch_pointcloud.models.pointgpt import (
    PointGPTClassification,
    PointGPTExtractor,
    PointGPTPartSegmentation,
    PointGPTPretraining,
    morton_sort,
)
from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.imports import (
    _CUDA_AVAILABLE,
    _PYG_LIB_AVAILABLE,
)

pytestmark = [
    pytest.mark.skipif(not _PYG_LIB_AVAILABLE, reason="pyg-lib is not installed"),
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
        num_groups=64,
        group_size=32,
        act="gelu",
    )
    pos, batch = _packed_batch()
    out = model(None, pos, batch)
    assert out.shape == (2, 40)


def test_pointgpt_generative_pretraining_basic() -> None:
    model = PointGPTPretraining(
        in_channels=0,
        embed_dim=384,
        depth=12,
        decoder_depth=4,
        num_heads=6,
        num_groups=64,
        group_size=32,
        act="gelu",
    )
    pos, batch = _packed_batch()
    pred, target, pred_batch, target_batch = model(None, pos, batch)
    assert pred.shape == target.shape == (pred_batch.shape[0], 3)
    assert torch.equal(pred_batch, target_batch)
    assert torch.equal(
        pred_batch, torch.arange(pred_batch.shape[0] // model.group_size).repeat_interleave(model.group_size)
    )


def test_pointgpt_classification_num_classes_zero_returns_features() -> None:
    model = PointGPTClassification(
        in_channels=0,
        num_classes=0,
        embed_dim=96,
        depth=2,
        num_heads=2,
        num_groups=16,
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
        num_groups=16,
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
    model = PointGPTPretraining(
        in_channels=0,
        embed_dim=96,
        depth=2,
        decoder_depth=1,
        num_heads=2,
        num_groups=8,
        group_size=4,
        mask_ratio=0.5,
        keep_attend=2,
        act="gelu",
    )
    model.eval()
    pos = torch.zeros(512, 3)
    batch = torch.cat([torch.zeros(256), torch.ones(256)]).long()
    pred, target, _, _ = model(None, pos, batch)
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
    model = PointGPTPretraining(
        in_channels=in_channels,
        embed_dim=384,
        depth=2,
        decoder_depth=1,
        num_heads=6,
        num_groups=64,
        group_size=32,
        act="gelu",
    )
    model.eval()
    assert model.encoder.local_mlp.channel_list[0] == 3 + in_channels
    pos, batch = _packed_batch()
    x_a = torch.randn(pos.size(0), in_channels)
    x_b = torch.randn(pos.size(0), in_channels)

    pred_a, target_a, _, _ = model(x_a, pos, batch)
    pred_b, _, _, _ = model(x_b, pos, batch)
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
        num_groups=64,
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


def _uniform_batch(num_points: int = 2048) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pos = torch.randn(2 * num_points, 3)
    batch = torch.cat([torch.zeros(num_points), torch.ones(num_points)]).long()
    category = torch.nn.functional.one_hot(torch.tensor([3, 7]), 16).float()
    return pos, batch, category


@pytest.mark.parametrize(
    ("embed_dim", "num_heads", "depth", "propagation_channels", "head_channels"),
    [
        pytest.param(384, 6, 12, (1536, 1024), (512, 256), id="s"),
        pytest.param(768, 12, 12, (3072, 2048), (1024, 512), id="b"),
        pytest.param(1024, 16, 24, (4096, 2048), (1024, 512), id="l"),
    ],
)
def test_pointgpt_part_segmentation_basic(
    embed_dim: int, num_heads: int, depth: int, propagation_channels: tuple, head_channels: tuple
) -> None:
    model = PointGPTPartSegmentation(
        in_channels=0,
        num_classes=50,
        num_categories=16,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        num_groups=128,
        group_size=32,
        propagation_channels=propagation_channels,
        head_channels=head_channels,
    )
    model.eval()
    pos, batch, category = _uniform_batch()
    out = model(None, pos, batch, category)
    assert out.shape == (4096, 50)
    assert model.num_features == propagation_channels[-1] + 6 * embed_dim + 64
    assert model.head.channel_list == [model.num_features, *head_channels, 50]


def test_pointgpt_part_segmentation_every_parameter_gets_a_gradient() -> None:
    model = PointGPTPartSegmentation(
        in_channels=0,
        num_classes=5,
        num_categories=16,
        embed_dim=96,
        depth=12,
        num_heads=2,
        num_groups=8,
        group_size=4,
        propagation_channels=(64, 32),
    )
    model.train()
    pos, batch, category = _uniform_batch(128)
    model(None, pos, batch, category).sum().backward()
    assert [name for name, param in model.named_parameters() if param.grad is None] == []


def test_pointgpt_part_segmentation_num_classes_zero_returns_features() -> None:
    model = PointGPTPartSegmentation(
        in_channels=0,
        num_classes=0,
        num_categories=16,
        embed_dim=96,
        depth=12,
        num_heads=2,
        num_groups=8,
        group_size=4,
        propagation_channels=(64, 32),
    )
    model.eval()
    assert isinstance(model.head, torch.nn.Identity)
    pos, batch, category = _uniform_batch(128)
    out = model(None, pos, batch, category)
    assert out.shape == (256, model.num_features)


def test_pointgpt_part_segmentation_ragged_batch_raises() -> None:
    model = PointGPTPartSegmentation(
        in_channels=0,
        num_classes=5,
        num_categories=16,
        embed_dim=96,
        num_heads=2,
        num_groups=8,
        group_size=4,
        propagation_channels=(64, 32),
    )
    model.eval()
    pos = torch.randn(128, 3)
    batch = torch.cat([torch.zeros(96), torch.ones(32)]).long()
    category = torch.nn.functional.one_hot(torch.tensor([3, 7]), 16).float()
    with pytest.raises(ValueError, match="same number of points per sample"):
        model(None, pos, batch, category)


def test_pointgpt_part_segmentation_validates_fetch_idx_against_depth() -> None:
    with pytest.raises(ValueError, match="fetch_idx"):
        PointGPTPartSegmentation(in_channels=0, num_classes=5, num_categories=16, depth=6, embed_dim=96, num_heads=2)


def test_pointgpt_part_segmentation_accepts_features() -> None:
    in_channels = 3
    model = PointGPTPartSegmentation(
        in_channels=in_channels,
        num_classes=50,
        num_categories=16,
        embed_dim=96,
        depth=12,
        num_heads=2,
        num_groups=8,
        group_size=4,
        propagation_channels=(64, 32),
    )
    model.eval()
    assert model.encoder.local_mlp.channel_list[0] == 3 + in_channels
    pos, batch, category = _uniform_batch(128)
    x_a = torch.randn(pos.size(0), in_channels)
    x_b = torch.randn(pos.size(0), in_channels)
    out_a = model(x_a, pos, batch, category)
    out_b = model(x_b, pos, batch, category)
    assert out_a.shape == (256, 50)
    assert not torch.allclose(out_a, out_b)


def test_pointgpt_extractor_fetch_idx_returns_hidden_states() -> None:
    extractor = PointGPTExtractor(embed_dim=32, num_heads=2, depth=4)
    extractor.eval()
    tokens = torch.randn(2, 5, 32)
    pos = torch.randn(2, 6, 32)
    mask = torch.triu(torch.ones(6, 6, dtype=torch.bool), diagonal=1)
    features = extractor(tokens, pos, mask, fetch_idx=(1, 3))
    assert [tuple(feat.shape) for feat in features] == [(2, 6, 32), (2, 6, 32)]
    assert torch.allclose(features[-1], extractor(tokens, pos, mask))


def test_pointgpt_extractor_fetch_idx_skips_the_later_blocks() -> None:
    extractor = PointGPTExtractor(embed_dim=32, num_heads=2, depth=6)
    calls: List[int] = []
    for i, layer in enumerate(extractor.layers):
        layer.register_forward_hook(lambda *_, i=i: calls.append(i))
    extractor(torch.randn(2, 5, 32), torch.randn(2, 6, 32), None, fetch_idx=(0, 2))
    assert calls == [0, 1, 2]


def test_pointgpt_shapenetpart_transform_runs_without_labels() -> None:
    transform = model_info("pointgpt-s.shapenetpart.guangyan-chen", task="part-segmentation")["transform"]
    assert transform is not None
    sample = {DataKeys.POS: torch.randn(3000, 3), DataKeys.CATEGORY: torch.tensor(4)}
    out = transform(sample)
    assert out[DataKeys.POS].shape == (2048, 3)
    assert out[DataKeys.CATEGORY].shape[-1] == 16
    assert DataKeys.SEGMENT not in out


def test_pointgpt_part_segmentation_causal_mask_changes_the_output() -> None:
    kwargs: Dict[str, Any] = dict(
        in_channels=0,
        num_classes=5,
        num_categories=16,
        embed_dim=96,
        depth=12,
        num_heads=2,
        num_groups=8,
        group_size=4,
        propagation_channels=(64, 32),
    )
    causal = PointGPTPartSegmentation(**kwargs, causal=True).eval()
    bidirectional = PointGPTPartSegmentation(**kwargs, causal=False).eval()
    bidirectional.load_state_dict(causal.state_dict())
    pos, batch, category = _uniform_batch(128)
    with torch.no_grad():
        out_causal = causal(None, pos, batch, category)
        out_bidirectional = bidirectional(None, pos, batch, category)
    assert out_causal.shape == out_bidirectional.shape == (256, 5)
    assert not torch.allclose(out_causal, out_bidirectional)
