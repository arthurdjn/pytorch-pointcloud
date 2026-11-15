import pytest
import torch
from torch_geometric.nn.norm import LayerNorm as GraphLayerNorm

from torch_pointcloud.layers.dropouts import DropPath
from torch_pointcloud.models import model_info, point_mamba
from torch_pointcloud.models.point_mamba import (
    PointMambaBlock,
    PointMambaClassification,
    PointMambaDecoderMAE,
    PointMambaEncoder,
    PointMambaPretraining,
    PointPatchEmbedding,
    order_sort,
)
from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.imports import (
    _CUDA_AVAILABLE,
    _MAMBA_SSM_AVAILABLE,
    _PYG_LIB_AVAILABLE,
)

# See: https://docs.pytest.org/en/stable/how-to/skipping.html#summary
pytestmark = [
    pytest.mark.skipif(not _MAMBA_SSM_AVAILABLE, reason="mamba_ssm is not available"),
    pytest.mark.skipif(not _PYG_LIB_AVAILABLE, reason="pyg-lib is not installed"),
]

# The mamba_ssm selective-scan kernels only run on CUDA tensors, so every test that calls a
# forward pass is gated; construction-only tests run on CPU.
requires_cuda = pytest.mark.skipif(not _CUDA_AVAILABLE, reason="mamba_ssm kernels require CUDA")


def test_order_sort_negative_grid_raises() -> None:
    """Negative grid coordinates would silently wrap to valid codes; order_sort rejects them."""
    grid = torch.tensor([[-1, 0, 0], [1, 2, 3]], dtype=torch.long)
    batch = torch.zeros(2, dtype=torch.long)
    with pytest.raises(ValueError, match="non-negative"):
        order_sort(grid, batch, order="hilbert")


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
        num_groups=64,
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
        num_groups=64,
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
        num_groups=64,
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
    """Test the basic functionality of the PointMambaPretraining model,
    following similar architecture as the original PointMamba model."""
    model = PointMambaPretraining(
        in_channels=0,
        embed_dim=384,
        encoder_depth=12,
        decoder_depth=4,
        num_groups=64,
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
    pred, target, pred_batch, target_batch = model(None, pos, batch)

    assert pred.shape == target.shape == (pred_batch.shape[0], 3)
    assert torch.equal(pred_batch, target_batch)


@requires_cuda
def test_point_mamba_mae_accepts_features() -> None:
    in_channels = 3
    model = PointMambaPretraining(
        in_channels=in_channels,
        embed_dim=192,
        encoder_depth=2,
        decoder_depth=1,
        num_groups=64,
        group_size=32,
        mask_ratio=0.6,
    )
    model.cuda()
    model.eval()
    assert model.encoder.patch_embed.local_mlp.channel_list[0] == 2 * in_channels + 3
    pos = torch.randn(2048, 3).cuda()
    batch = torch.cat([torch.zeros(1024), torch.ones(1024)]).long().cuda()
    x = torch.randn(2048, in_channels).cuda()

    pred, target, pred_batch, target_batch = model(x, pos, batch)
    assert pred.shape == target.shape == (pred_batch.shape[0], 3)
    assert torch.equal(pred_batch, target_batch)


class _CausalScan(torch.nn.Module):
    """CPU stand-in for the selective-scan kernel: a GRU reads the tokens left to right."""

    def __init__(self, d_model: int, **kwargs: object) -> None:
        super().__init__()
        self.gru = torch.nn.GRU(d_model, d_model, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gru(x)[0]


@pytest.mark.parametrize(
    "norm",
    [
        pytest.param("layer_norm", id="string"),
        pytest.param("LayerNorm", id="class-name"),
        pytest.param("layernorm", id="lowercase"),
    ],
)
def test_point_mamba_layer_norm_normalizes_every_point_on_its_own(norm: str) -> None:
    torch.manual_seed(0)
    embed = PointPatchEmbedding(0, 32, num_groups=8, group_size=4, norm=norm).eval()
    pos = torch.rand(128, 3)
    batch = torch.arange(2).repeat_interleave(64)
    with torch.no_grad():
        both, _, _ = embed(None, pos, batch)
        first, _, _ = embed(None, pos[:64], batch[:64])
    assert torch.allclose(both[:8], first, atol=1e-6)


@pytest.mark.parametrize(
    "model_cls, kwargs",
    [
        pytest.param(PointMambaClassification, dict(num_classes=5, depth=1), id="classification"),
        pytest.param(PointMambaPretraining, dict(encoder_depth=1, decoder_depth=1), id="pretraining"),
    ],
)
def test_point_mamba_builds_with_layer_norm(model_cls: type, kwargs: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(point_mamba, "Mamba", _CausalScan)
    model = model_cls(0, embed_dim=32, norm="layer_norm", **kwargs)
    assert isinstance(model, torch.nn.Module)


@pytest.mark.parametrize(
    "norm",
    [
        pytest.param("layer_norm", id="string"),
        pytest.param("LayerNorm", id="class-name"),
        pytest.param("layernorm", id="lowercase"),
    ],
)
def test_point_mamba_classification_layer_norm_names_are_per_token(norm: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(point_mamba, "Mamba", _CausalScan)
    model = PointMambaClassification(0, num_classes=5, depth=1, embed_dim=32, norm=norm, head_channels=[16])
    layer_norms = [module for module in model.modules() if isinstance(module, GraphLayerNorm)]
    assert layer_norms
    assert all(layer_norm.mode == "node" for layer_norm in layer_norms)


def test_point_mamba_classification_drops_tokens_after_every_block(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(point_mamba, "Mamba", _CausalScan)
    functional_dropout = torch.nn.functional.dropout
    calls = []

    def record(input: torch.Tensor, p: float = 0.5, training: bool = True, inplace: bool = False) -> torch.Tensor:
        calls.append((tuple(input.shape), p))
        return functional_dropout(input, p, training, inplace)

    monkeypatch.setattr(torch.nn.functional, "dropout", record)
    model = PointMambaClassification(0, 5, embed_dim=32, depth=3, num_groups=8, group_size=4, dropout=0.3).eval()
    pos = torch.rand(128, 3)
    batch = torch.arange(2).repeat_interleave(64)
    model(None, pos, batch)
    assert [shape for shape, p in calls if p == 0.3] == [(2, 16, 32)] * 3
    model.reset_classifier(5, dropout=0.2)
    assert model.encoder.dropout == 0.2


@pytest.mark.parametrize(
    "name, dropout",
    [
        pytest.param("point-mamba-base.modelnet40.dingkang-liang", 0.0, id="modelnet40"),
        pytest.param("point-mamba-base.scanobjectnn-objbg.dingkang-liang", 0.0, id="objbg"),
        pytest.param("point-mamba-base.scanobjectnn-objonly.dingkang-liang", 0.1, id="objonly"),
        pytest.param("point-mamba-base.scanobjectnn-hardest.dingkang-liang", 0.0, id="hardest"),
    ],
)
def test_point_mamba_entries_register_the_released_dropout(name: str, dropout: float) -> None:
    assert model_info(name)["hparams"]["dropout"] == dropout


def test_point_mamba_modelnet40_transform_runs_without_normals() -> None:
    transform = model_info("point-mamba-base.modelnet40.dingkang-liang")["transform"]
    assert transform is not None
    data = transform({DataKeys.POS: torch.rand(3000, 3)})
    assert data[DataKeys.POS].shape == (1024, 3)
