from typing import Any, Dict

import pytest
import torch
from torch import Tensor, nn

from torch_pointcloud.layers.randlanet_blocks import RandLANetResidualBlock
from torch_pointcloud.models.randlanet import (
    RandLANetClassification,
    RandLANetDecoder,
    RandLANetEncoder,
    RandLANetSegmentation,
    pyg_example_checkpoint_filter_fn,
    random_max_pool,
    random_pool,
)
from torch_pointcloud.utils.imports import _PYG_LIB_AVAILABLE

# See: https://docs.pytest.org/en/stable/how-to/skipping.html#summary
pytestmark = pytest.mark.skipif(
    not _PYG_LIB_AVAILABLE,
    reason="pyg-lib is not installed",
)


@pytest.fixture
def data() -> Dict[str, Tensor]:
    torch.manual_seed(42)
    lengths = torch.tensor([256, 512])
    pos = torch.randn(int(lengths.sum()), 3)
    features = torch.randn(int(lengths.sum()), 6)
    batch = torch.repeat_interleave(torch.arange(len(lengths)), lengths)
    return dict(features=features, pos=pos, batch=batch)


@pytest.fixture
def mlp_kwargs() -> Dict[str, Any]:
    return dict(act="relu", norm="batch_norm", bias=False)


@pytest.fixture
def model_clf() -> RandLANetClassification:
    return RandLANetClassification(
        in_channels=6,
        num_classes=10,
        stem_channels=8,
        encoder_channels=[16, 32, 64],
        decimation=2,
        num_neighbors=8,
    )


@pytest.fixture
def model_seg() -> RandLANetSegmentation:
    return RandLANetSegmentation(
        in_channels=6,
        num_classes=10,
        stem_channels=8,
        encoder_channels=[16, 32, 64],
        fp_channels=[32, 16, 8],
        head_channels=[16, 8],
        decimation=2,
        num_neighbors=8,
    )


def test_randlanet_random_pool(data: Dict[str, Tensor]) -> None:
    pooled, pos_decim, batch_decim = random_pool(
        data["features"], data["pos"], data["batch"], factor=4, num_neighbors=8
    )
    expected_n = 256 // 4 + 512 // 4
    assert pooled.shape == (expected_n, data["features"].shape[1])
    assert pos_decim.shape == (expected_n, 3)
    assert batch_decim.shape == (expected_n,)
    assert batch_decim.dtype == torch.long


def test_randlanet_random_pool_without_reduce_keeps_sampled_features(data: Dict[str, Tensor]) -> None:
    x = torch.cat([data["pos"], data["features"]], dim=1)
    sampled, pos_decim, batch_decim = random_pool(x, data["pos"], data["batch"], factor=4, num_neighbors=8, reduce=None)
    assert sampled.shape == (256 // 4 + 512 // 4, 9)
    # Each kept point carries its own features, not a pool over its neighbors.
    assert torch.equal(sampled[:, :3], pos_decim)
    assert torch.equal(torch.bincount(batch_decim), torch.tensor([64, 128]))


def test_randlanet_random_max_pool_is_deprecated(data: Dict[str, Tensor]) -> None:
    torch.manual_seed(0)
    expected = random_pool(data["features"], data["pos"], data["batch"], factor=4, num_neighbors=8)
    torch.manual_seed(0)
    with pytest.warns(FutureWarning, match="random_pool"):
        actual = random_max_pool(data["features"], data["pos"], data["batch"], factor=4, num_neighbors=8)
    for a, e in zip(actual, expected):
        assert torch.equal(a, e)


def test_randlanet_pyg_example_checkpoint_filter_fn() -> None:
    lse_weight = torch.arange(40.0).reshape(4, 10)
    state_dict = {
        "fc0.weight": torch.zeros(1),
        "block1.mlp1.norms.0.module.running_mean": torch.zeros(1),
        "block2.lfa1.mlp_encoder.lins.0.weight": lse_weight,
        "block2.lfa2.mlp_attention.lins.0.weight": torch.zeros(1),
        "block3.lfa1.mlp_post_attention.norms.0.module.weight": torch.zeros(1),
        "mlp_summit.lins.0.bias": torch.zeros(1),
        "fp4.nn.lins.0.weight": torch.zeros(1),
        "fp1.nn.norms.0.module.num_batches_tracked": torch.zeros(1),
        "mlp_classif.lins.1.bias": torch.zeros(1),
        "fc_classif.weight": torch.zeros(1),
    }
    result = pyg_example_checkpoint_filter_fn(state_dict, nn.Module())
    assert list(result) == [
        "stem.lins.0.weight",
        "encoder.blocks.0.mlp1.norms.0.module.running_mean",
        "encoder.blocks.1.lfa.lse1.mlp.lins.0.weight",
        "encoder.blocks.1.lfa.att_pooling_2.fc.weight",
        "encoder.blocks.2.lfa.att_pooling_1.mlp.norms.0.module.weight",
        "aggr.lins.0.bias",
        "decoder.fp_blocks.0.mlp.lins.0.weight",
        "decoder.fp_blocks.3.mlp.norms.0.module.num_batches_tracked",
        "head.lins.1.bias",
        "head.lins.2.weight",
    ]
    # [pos_i, pos_j, pos_j - pos_i, dist] -> [dist, pos_i - pos_j, pos_i, pos_j]
    expected = torch.cat([lse_weight[:, 9:10], -lse_weight[:, 6:9], lse_weight[:, 0:3], lse_weight[:, 3:6]], dim=1)
    assert torch.equal(result["encoder.blocks.1.lfa.lse1.mlp.lins.0.weight"], expected)


def test_randlanet_encoder_forward(data: Dict[str, Tensor], mlp_kwargs: Dict[str, Any]) -> None:
    encoder = RandLANetEncoder(
        in_channels=6,
        encoder_channels=[16, 32, 64],
        decimation=2,
        num_neighbors=8,
        **mlp_kwargs,
    )
    x, pos, batch = encoder(data["features"], data["pos"], data["batch"])
    assert x.shape[1] == 64
    assert pos.shape[1] == 3
    assert x.shape[0] == pos.shape[0] == batch.shape[0]


def test_randlanet_encoder_intermediates(data: Dict[str, Tensor], mlp_kwargs: Dict[str, Any]) -> None:
    encoder = RandLANetEncoder(
        in_channels=6,
        encoder_channels=[16, 32, 64],
        decimation=2,
        num_neighbors=8,
        **mlp_kwargs,
    )
    _, _, _, intermediates = encoder(data["features"], data["pos"], data["batch"], return_intermediates=True)
    assert len(intermediates) == 3
    for inter in intermediates:
        assert inter["x"].shape[0] == inter["pos"].shape[0] == inter["batch"].shape[0]
    # First intermediate is the full-resolution skip (pre-decimation block-0 output).
    assert intermediates[0]["x"].shape == (data["pos"].shape[0], 16)


def test_randlanet_encoder_pyg_example_switches(data: Dict[str, Tensor], mlp_kwargs: Dict[str, Any]) -> None:
    encoder = RandLANetEncoder(
        in_channels=6,
        encoder_channels=[16, 32, 64],
        decimation=2,
        num_neighbors=8,
        lse_channels=[2, 4, 8],
        lse_input="pos",
        pool=None,
        **mlp_kwargs,
    )
    for block, lse_channels in zip(encoder.blocks, [2, 4, 8]):
        assert isinstance(block, RandLANetResidualBlock)
        assert block.lfa.lse_channels == lse_channels
        assert block.lfa.lse2.in_channels == 10
    x, pos, batch = encoder(data["features"], data["pos"], data["batch"])
    assert x.shape == (256 // 8 + 512 // 8, 64)
    assert x.shape[0] == pos.shape[0] == batch.shape[0]


def test_randlanet_encoder_bad_lse_channels_length_raises(mlp_kwargs: Dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        RandLANetEncoder(in_channels=6, encoder_channels=[16, 32, 64], lse_channels=[2, 4], **mlp_kwargs)


def test_randlanet_encoder_odd_channels_raises(mlp_kwargs: Dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="must be even"):
        RandLANetEncoder(in_channels=6, encoder_channels=[16, 33], **mlp_kwargs)


def test_randlanet_encoder_bad_decimation_length_raises(mlp_kwargs: Dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        RandLANetEncoder(
            in_channels=6,
            encoder_channels=[16, 32, 64],
            decimation=[2, 2],
            **mlp_kwargs,
        )


def test_randlanet_decoder_forward(data: Dict[str, Tensor], mlp_kwargs: Dict[str, Any]) -> None:
    encoder = RandLANetEncoder(
        in_channels=6,
        encoder_channels=[16, 32, 64],
        decimation=2,
        num_neighbors=8,
        **mlp_kwargs,
    )
    decoder = RandLANetDecoder(
        in_channels=64,
        skip_channels=[32, 16, 16],
        fp_channels=[32, 16, 8],
        **mlp_kwargs,
    )
    x, pos, batch, intermediates = encoder(data["features"], data["pos"], data["batch"], return_intermediates=True)
    x, pos, batch = decoder(x, pos, batch, intermediates)
    assert x.shape == (data["pos"].shape[0], 8)


def test_randlanet_decoder_length_mismatch_raises(mlp_kwargs: Dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="must match"):
        RandLANetDecoder(
            in_channels=64,
            skip_channels=[32, 16],
            fp_channels=[32, 16, 8],
            **mlp_kwargs,
        )


def test_randlanet_classification_forward(model_clf: RandLANetClassification, data: Dict[str, Tensor]) -> None:
    logits = model_clf(data["features"], data["pos"], data["batch"])
    assert logits.shape == (int(data["batch"].max()) + 1, model_clf.num_classes)
    assert logits.dtype == data["features"].dtype


def test_randlanet_segmentation_forward(model_seg: RandLANetSegmentation, data: Dict[str, Tensor]) -> None:
    logits = model_seg(data["features"], data["pos"], data["batch"])
    assert logits.shape == (data["pos"].shape[0], model_seg.num_classes)
    assert logits.dtype == data["features"].dtype


def test_randlanet_classification_reset_classifier(model_clf: RandLANetClassification, data: Dict[str, Tensor]) -> None:
    model_clf.reset_classifier(num_classes=42)
    logits = model_clf(data["features"], data["pos"], data["batch"])
    assert logits.shape == (int(data["batch"].max()) + 1, 42)


def test_randlanet_classification_reset_classifier_keeps_current_pooling(model_clf: RandLANetClassification) -> None:
    model_clf.reset_classifier(10, global_pool="mean")
    pool = model_clf.global_pool
    model_clf.reset_classifier(7)
    assert model_clf.global_pool is pool
    assert type(pool).__name__ == "MeanPool"


def test_randlanet_segmentation_reset_classifier(model_seg: RandLANetSegmentation, data: Dict[str, Tensor]) -> None:
    model_seg.reset_classifier(num_classes=42)
    logits = model_seg(data["features"], data["pos"], data["batch"])
    assert logits.shape == (data["pos"].shape[0], 42)


def test_randlanet_segmentation_reset_classifier_ignores_extra_kwargs(model_seg: RandLANetSegmentation) -> None:
    model_seg.reset_classifier(num_classes=7, global_pool="mean")
    assert model_seg.num_classes == 7


def test_randlanet_segmentation_num_classes_zero_head_is_identity(model_seg: RandLANetSegmentation) -> None:
    model_seg.reset_classifier(num_classes=0)
    assert isinstance(model_seg.head, torch.nn.Identity)


def test_randlanet_segmentation_no_stem(data: Dict[str, Tensor]) -> None:
    model = RandLANetSegmentation(
        in_channels=6,
        num_classes=10,
        stem_channels=None,
        encoder_channels=[16, 32, 64],
        fp_channels=[32, 16, 8],
        decimation=2,
        num_neighbors=8,
    )
    assert model.stem is None
    logits = model(data["features"], data["pos"], data["batch"])
    assert logits.shape == (data["pos"].shape[0], 10)


def test_randlanet_segmentation_plain_stem(data: Dict[str, Tensor]) -> None:
    model = RandLANetSegmentation(
        in_channels=6,
        num_classes=10,
        stem_channels=8,
        stem_plain_last=True,
        encoder_channels=[16, 32, 64],
        fp_channels=[32, 16, 8],
        decimation=2,
        num_neighbors=8,
        lse_channels=[2, 4, 8],
        lse_input="pos",
        pool=None,
    )
    assert model.stem is not None
    assert len(model.stem.norms) == 0
    logits = model(data["features"], data["pos"], data["batch"])
    assert logits.shape == (data["pos"].shape[0], 10)


def test_randlanet_segmentation_with_aggr(data: Dict[str, Tensor]) -> None:
    model = RandLANetSegmentation(
        in_channels=6,
        num_classes=10,
        stem_channels=8,
        encoder_channels=[16, 32, 64],
        fp_channels=[32, 16, 8],
        aggr_channels=[128, 64],
        decimation=2,
        num_neighbors=8,
    )
    assert model.aggr is not None
    logits = model(data["features"], data["pos"], data["batch"])
    assert logits.shape == (data["pos"].shape[0], 10)


def test_randlanet_classification_with_aggr(data: Dict[str, Tensor]) -> None:
    model = RandLANetClassification(
        in_channels=6,
        num_classes=10,
        stem_channels=8,
        encoder_channels=[16, 32, 64],
        aggr_channels=[128, 64],
        decimation=2,
        num_neighbors=8,
    )
    assert model.aggr is not None
    assert model.num_features == 64
    logits = model(data["features"], data["pos"], data["batch"])
    assert logits.shape == (int(data["batch"].max()) + 1, 10)
