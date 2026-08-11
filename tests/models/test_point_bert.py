import pytest
import torch

from torch_pointcloud.layers import TransformerBlock
from torch_pointcloud.models.point_bert import (
    PointBERTClassification,
    PointBERTDiscreteVAE,
    PointBERTEncoder,
    PointBERTMaskedTransformer,
)
from torch_pointcloud.utils.imports import (
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


def test_point_bert_encoder_basic() -> None:
    model = PointBERTEncoder(
        embed_dim=384,
        depth=12,
        num_heads=6,
        num_group=64,
        group_size=32,
        encoder_dims=256,
        act="gelu",
        act_kwargs=None,
    )
    pos, batch = _packed_batch()
    out = model(None, pos, batch)
    assert out.shape == (2, 65, 384)


def test_point_bert_classification_basic() -> None:
    model = PointBERTClassification(
        in_channels=0,
        num_classes=40,
        embed_dim=384,
        depth=12,
        num_heads=6,
        num_group=64,
        group_size=32,
        encoder_dims=256,
        drop_path=0.1,
        act="gelu",
        act_kwargs=None,
        head_act="relu",
        dropout=0.5,
        head_channels=256,
    )
    pos, batch = _packed_batch()
    out = model(None, pos, batch)
    assert out.shape == (2, 40)


def _small_encoder(drop_path: float) -> PointBERTEncoder:
    return PointBERTEncoder(
        embed_dim=96,
        depth=4,
        num_heads=2,
        num_group=16,
        group_size=8,
        encoder_dims=64,
        drop_path=drop_path,
    )


def test_point_bert_encoder_drop_path_schedule() -> None:
    model = _small_encoder(drop_path=0.9)
    baseline = _small_encoder(drop_path=0.0)
    baseline.load_state_dict(model.state_dict())

    rates = []
    for block in model.blocks:
        assert isinstance(block, TransformerBlock)
        rates.append(block.drop_path.drop_prob)
    assert rates[0] == 0.0
    assert rates[-1] == pytest.approx(0.9)
    assert rates == sorted(rates)

    pos, batch = _packed_batch()
    model.train()
    baseline.train()
    torch.manual_seed(0)
    out_dp = model(None, pos, batch)
    torch.manual_seed(0)
    out_base = baseline(None, pos, batch)
    assert not torch.allclose(out_dp, out_base)

    # The train forwards update batch-norm running stats from fps-randomized groupings (fps'
    # random start is not governed by `torch.manual_seed`), so re-sync before the eval check.
    baseline.load_state_dict(model.state_dict())
    model.eval()
    baseline.eval()
    assert torch.allclose(model(None, pos, batch), baseline(None, pos, batch))


def test_point_bert_classification_num_classes_zero_returns_features() -> None:
    model = PointBERTClassification(
        in_channels=0,
        num_classes=0,
        embed_dim=96,
        depth=2,
        num_heads=2,
        num_group=16,
        group_size=8,
        encoder_dims=64,
    )
    assert isinstance(model.head, torch.nn.Identity)
    pos, batch = _packed_batch()
    out = model(None, pos, batch)
    assert out.shape == (2, 2 * model.embed_dim)


def test_point_bert_classification_accepts_features() -> None:
    in_channels = 3
    model = PointBERTClassification(
        in_channels=in_channels,
        num_classes=40,
        embed_dim=384,
        depth=2,
        num_heads=6,
        num_group=64,
        group_size=32,
        encoder_dims=256,
        act="gelu",
    )
    model.eval()
    assert model.encoder.encoder.local_mlp.channel_list[0] == 3 + in_channels
    pos, batch = _packed_batch()
    x_a = torch.randn(pos.size(0), in_channels)
    x_b = torch.randn(pos.size(0), in_channels)

    out_a = model(x_a, pos, batch)
    out_b = model(x_b, pos, batch)
    assert out_a.shape == (2, 40)
    assert not torch.allclose(out_a, out_b)


def test_point_bert_masked_transformer_basic() -> None:
    model = PointBERTMaskedTransformer(
        in_channels=0,
        embed_dim=384,
        depth=12,
        num_heads=6,
        num_group=64,
        group_size=32,
        encoder_dims=256,
        num_tokens=8192,
        cls_dim=512,
        act="gelu",
        act_kwargs=None,
    )
    pos, batch = _packed_batch()
    out = model(None, pos, batch)
    assert out["cls_feature"].shape == (2, 512)
    assert out["logits"].shape == (2, 64, 8192)


def test_point_bert_masked_transformer_accepts_features() -> None:
    in_channels = 3
    model = PointBERTMaskedTransformer(
        in_channels=in_channels,
        embed_dim=384,
        depth=2,
        num_heads=6,
        num_group=64,
        group_size=32,
        encoder_dims=256,
        num_tokens=512,
        cls_dim=128,
    )
    model.eval()
    assert model.encoder.local_mlp.channel_list[0] == 3 + in_channels
    pos, batch = _packed_batch()
    x_a = torch.randn(pos.size(0), in_channels)
    x_b = torch.randn(pos.size(0), in_channels)

    out_a = model(x_a, pos, batch)
    out_b = model(x_b, pos, batch)
    assert out_a["logits"].shape == (2, 64, 512)
    assert not torch.allclose(out_a["logits"], out_b["logits"])


def _small_masked_transformer(mask_ratio: tuple[float, float]) -> PointBERTMaskedTransformer:
    return PointBERTMaskedTransformer(
        in_channels=0,
        embed_dim=96,
        depth=2,
        num_heads=2,
        num_group=16,
        group_size=8,
        encoder_dims=64,
        num_tokens=128,
        cls_dim=32,
        mask_ratio=mask_ratio,
        drop_path=0.0,
    )


def test_point_bert_masked_transformer_masks_only_in_training() -> None:
    model = _small_masked_transformer(mask_ratio=(0.25, 0.45))
    with torch.no_grad():
        model.mask_token.normal_()
    unmasked = _small_masked_transformer(mask_ratio=(0.0, 0.0))
    unmasked.load_state_dict(model.state_dict())

    pos, batch = _packed_batch()
    model.train()
    unmasked.train()
    torch.manual_seed(0)
    out_masked = model(None, pos, batch)
    torch.manual_seed(0)
    out_unmasked = unmasked(None, pos, batch)
    assert not torch.allclose(out_masked["logits"], out_unmasked["logits"])

    # The train forwards update batch-norm running stats from fps-randomized groupings (fps'
    # random start is not governed by `torch.manual_seed`), so re-sync before the eval check.
    unmasked.load_state_dict(model.state_dict())
    model.eval()
    unmasked.eval()
    out_a = model(None, pos, batch)
    out_b = unmasked(None, pos, batch)
    assert torch.equal(out_a["logits"], out_b["logits"])
    assert torch.equal(out_a["cls_feature"], out_b["cls_feature"])


def test_point_bert_dvae_basic() -> None:
    model = PointBERTDiscreteVAE(
        in_channels=0,
        num_group=64,
        group_size=32,
        encoder_dims=256,
        num_tokens=8192,
        tokens_dims=256,
        decoder_dims=256,
    )
    pos, batch = _packed_batch()
    out = model(None, pos, batch)
    assert out["logits"].shape == (2, 64, 8192)
    assert out["fine"].shape == (2, 64, 32, 3)
    assert out["coarse"].shape == (2, 64, 8, 3)
