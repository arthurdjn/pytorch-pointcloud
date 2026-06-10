import pytest
import torch

from torch_pointcloud.models.point_bert import (
    PointBERTClassification,
    PointBERTDiscreteVAE,
    PointBERTEncoder,
    PointBERTMaskedTransformer,
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
    ).cuda()
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
        drop_path_rate=0.1,
        act="gelu",
        act_kwargs=None,
        head_act="relu",
        dropout=0.5,
        head_channels=256,
    ).cuda()
    pos, batch = _packed_batch()
    out = model(None, pos, batch)
    assert out.shape == (2, 40)


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
    ).cuda()
    model.eval()
    assert model.encoder.encoder.local_mlp.channel_list[0] == 3 + in_channels
    pos, batch = _packed_batch()
    x_a = torch.randn(pos.size(0), in_channels).cuda()
    x_b = torch.randn(pos.size(0), in_channels).cuda()

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
    ).cuda()
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
    ).cuda()
    model.eval()
    assert model.encoder.local_mlp.channel_list[0] == 3 + in_channels
    pos, batch = _packed_batch()
    x_a = torch.randn(pos.size(0), in_channels).cuda()
    x_b = torch.randn(pos.size(0), in_channels).cuda()

    out_a = model(x_a, pos, batch)
    out_b = model(x_b, pos, batch)
    assert out_a["logits"].shape == (2, 64, 512)
    assert not torch.allclose(out_a["logits"], out_b["logits"])


def test_point_bert_dvae_basic() -> None:
    model = PointBERTDiscreteVAE(
        in_channels=0,
        num_group=64,
        group_size=32,
        encoder_dims=256,
        num_tokens=8192,
        tokens_dims=256,
        decoder_dims=256,
    ).cuda()
    pos, batch = _packed_batch()
    out = model(None, pos, batch)
    assert out["logits"].shape == (2, 64, 8192)
    assert out["fine"].shape == (2, 64, 32, 3)
    assert out["coarse"].shape == (2, 64, 8, 3)
