from typing import Callable, Dict

import pytest
import torch
from torch import Tensor, nn

from torch_pointcloud.models.spformer_unet import SPFormerUNetDecoder, SPFormerUNetEncoder, SPFormerUNetSegmentation
from torch_pointcloud.utils.imports import _SPCONV_AVAILABLE

pytestmark = pytest.mark.skipif(not _SPCONV_AVAILABLE, reason="spconv is not installed")

CHANNELS = (16, 32, 64)
DEPTHS = (1, 1, 1)


@pytest.fixture
def data() -> Dict[str, Tensor]:
    torch.manual_seed(42)
    lengths = torch.tensor([256, 512])
    pos_grid = torch.randint(0, 64, (int(lengths.sum()), 3))
    x = torch.randn(int(lengths.sum()), 6)
    batch = torch.repeat_interleave(torch.arange(len(lengths)), lengths)
    return dict(x=x, pos_grid=pos_grid, batch=batch)


@pytest.fixture
def model(spconv_to_device: Callable[[nn.Module, str], None]) -> SPFormerUNetSegmentation:
    model = SPFormerUNetSegmentation(
        in_channels=6,
        num_classes=10,
        channels=CHANNELS,
        depths=DEPTHS,
        spatial_padding=64,
    )
    spconv_to_device(model, "cpu")
    return model


def test_spformer_unet_forward(model: SPFormerUNetSegmentation, data: Dict[str, Tensor]) -> None:
    logits = model(data["x"], data["pos_grid"], data["batch"])
    assert logits.shape == (data["pos_grid"].shape[0], model.num_classes)


def test_spformer_unet_identity_head_returns_features(
    data: Dict[str, Tensor],
    spconv_to_device: Callable[[nn.Module, str], None],
) -> None:
    model = SPFormerUNetSegmentation(
        in_channels=6,
        num_classes=0,
        channels=CHANNELS,
        depths=DEPTHS,
        spatial_padding=64,
    )
    spconv_to_device(model, "cpu")
    assert isinstance(model.head, nn.Identity)
    feats = model(data["x"], data["pos_grid"], data["batch"])
    assert feats.shape == (data["pos_grid"].shape[0], model.num_features)
    assert model.num_features == CHANNELS[0]


def test_spformer_unet_reset_classifier(
    model: SPFormerUNetSegmentation,
    data: Dict[str, Tensor],
    spconv_to_device: Callable[[nn.Module, str], None],
) -> None:
    model.reset_classifier(num_classes=42)
    spconv_to_device(model, "cpu")
    logits = model(data["x"], data["pos_grid"], data["batch"])
    assert logits.shape == (data["pos_grid"].shape[0], 42)


def test_spformer_unet_forward_features_decoder_head(model: SPFormerUNetSegmentation, data: Dict[str, Tensor]) -> None:
    bottleneck, skips = model.forward_features(data["x"], data["pos_grid"], data["batch"], return_intermediates=True)
    assert len(skips) == len(CHANNELS) - 1
    assert bottleneck.features.shape[1] == CHANNELS[-1]
    sparse_x = model.forward_decoder(bottleneck, skips)
    assert sparse_x.features.shape[1] == CHANNELS[0]
    logits = model.forward_head(sparse_x)
    assert logits.shape == (data["pos_grid"].shape[0], model.num_classes)


def test_spformer_unet_forward_head_pre_logits(model: SPFormerUNetSegmentation, data: Dict[str, Tensor]) -> None:
    bottleneck, skips = model.forward_features(data["x"], data["pos_grid"], data["batch"], return_intermediates=True)
    sparse_x = model.forward_decoder(bottleneck, skips)
    feats = model.forward_head(sparse_x, pre_logits=True)
    assert torch.equal(feats, sparse_x.features)
    assert feats.shape[1] == CHANNELS[0]


def test_spformer_unet_encoder_decoder_roundtrip(
    data: Dict[str, Tensor],
    spconv_to_device: Callable[[nn.Module, str], None],
) -> None:
    encoder = SPFormerUNetEncoder(6, CHANNELS, DEPTHS, spatial_padding=64)
    decoder = SPFormerUNetDecoder(CHANNELS, DEPTHS)
    spconv_to_device(encoder, "cpu")
    spconv_to_device(decoder, "cpu")
    bottleneck, skips = encoder(data["x"], data["pos_grid"], data["batch"], return_intermediates=True)
    out = decoder(bottleneck, skips)
    assert len(skips) == len(CHANNELS) - 1
    assert out.features.shape[1] == CHANNELS[0]
