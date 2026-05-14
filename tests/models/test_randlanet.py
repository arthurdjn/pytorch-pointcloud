from typing import Any, Dict

import pytest
import torch
from torch import Tensor

from torch_pointcloud.models.randlanet import (
    DilatedResidualBlock,
    RandLANetClassification,
    RandLANetDecoder,
    RandLANetEncoder,
    RandLANetIntermediate,
    RandLANetSegmentation,
    random_max_pool,
)
from torch_pointcloud.utils.imports import _TORCH_CLUSTER_AVAILABLE, _TORCH_SCATTER_AVAILABLE

# See: https://docs.pytest.org/en/stable/how-to/skipping.html#summary
pytestmark = pytest.mark.skipif(
    not (_TORCH_SCATTER_AVAILABLE and _TORCH_CLUSTER_AVAILABLE),
    reason="torch-scatter or torch-cluster is not installed",
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


def test_randlanet_intermediate_namedtuple() -> None:
    intermediate = RandLANetIntermediate(
        x=torch.randn(10, 4),
        pos=torch.randn(10, 3),
        batch=torch.zeros(10, dtype=torch.long),
    )
    assert intermediate.x.shape == (10, 4)
    assert intermediate.pos.shape == (10, 3)
    assert intermediate.batch.shape == (10,)


def test_randlanet_random_max_pool(data: Dict[str, Tensor]) -> None:
    pooled, pos_decim, batch_decim = random_max_pool(
        data["features"], data["pos"], data["batch"], factor=4, num_neighbors=8
    )
    expected_n = 256 // 4 + 512 // 4
    assert pooled.shape == (expected_n, data["features"].shape[1])
    assert pos_decim.shape == (expected_n, 3)
    assert batch_decim.shape == (expected_n,)
    assert batch_decim.dtype == torch.long


def test_randlanet_dilated_residual_block(data: Dict[str, Tensor], mlp_kwargs: Dict[str, Any]) -> None:
    block = DilatedResidualBlock(d_in=6, d_out=8, num_neighbors=8, **mlp_kwargs)
    x, pos, batch = block(data["features"], data["pos"], data["batch"])
    assert x.shape == (data["features"].shape[0], 16)
    assert pos.shape == data["pos"].shape
    assert batch.shape == data["batch"].shape


def test_randlanet_dilated_residual_block_odd_d_out_raises(mlp_kwargs: Dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="must be even"):
        DilatedResidualBlock(d_in=6, d_out=7, num_neighbors=8, **mlp_kwargs)


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
        assert isinstance(inter, RandLANetIntermediate)
        assert inter.x.shape[0] == inter.pos.shape[0] == inter.batch.shape[0]
    # First intermediate is the full-resolution skip (pre-decimation block-0 output).
    assert intermediates[0].x.shape == (data["pos"].shape[0], 16)


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


def test_randlanet_segmentation_reset_classifier(model_seg: RandLANetSegmentation, data: Dict[str, Tensor]) -> None:
    model_seg.reset_classifier(num_classes=42)
    logits = model_seg(data["features"], data["pos"], data["batch"])
    assert logits.shape == (data["pos"].shape[0], 42)


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
