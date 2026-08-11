from typing import Dict

import pytest
import torch
from torch import Tensor

from torch_pointcloud.models.spformer_unet import SPFormerUNetDecoder, SPFormerUNetEncoder, SPFormerUNetSegmentation
from torch_pointcloud.utils.imports import _CUDA_AVAILABLE, _SPCONV_AVAILABLE

pytestmark = [
    pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA is not available"),
    pytest.mark.skipif(not _SPCONV_AVAILABLE, reason="spconv is not installed"),
]

CHANNELS = (16, 32, 64)
LAYERS = (1, 1, 1)


@pytest.fixture
def data() -> Dict[str, Tensor]:
    torch.manual_seed(42)
    lengths = torch.tensor([256, 512])
    pos_grid = torch.randint(0, 64, (int(lengths.sum()), 3))
    x = torch.randn(int(lengths.sum()), 6)
    batch = torch.repeat_interleave(torch.arange(len(lengths)), lengths)
    return dict(x=x.cuda(), pos_grid=pos_grid.cuda(), batch=batch.cuda())


@pytest.fixture
def model() -> SPFormerUNetSegmentation:
    return SPFormerUNetSegmentation(
        in_channels=6,
        num_classes=10,
        channels=CHANNELS,
        layers=LAYERS,
        spatial_padding=64,
    ).cuda()


def test_spformer_unet_forward(model: SPFormerUNetSegmentation, data: Dict[str, Tensor]) -> None:
    logits = model(data["x"], data["pos_grid"], data["batch"])
    assert logits.shape == (data["pos_grid"].shape[0], model.num_classes)


def test_spformer_unet_identity_head_returns_features(data: Dict[str, Tensor]) -> None:
    model = SPFormerUNetSegmentation(
        in_channels=6,
        num_classes=0,
        channels=CHANNELS,
        layers=LAYERS,
        spatial_padding=64,
    ).cuda()
    feats = model(data["x"], data["pos_grid"], data["batch"])
    assert feats.shape == (data["pos_grid"].shape[0], CHANNELS[0])


def test_spformer_unet_reset_classifier(model: SPFormerUNetSegmentation, data: Dict[str, Tensor]) -> None:
    model.reset_classifier(num_classes=42)
    model.cuda()
    logits = model(data["x"], data["pos_grid"], data["batch"])
    assert logits.shape == (data["pos_grid"].shape[0], 42)


def test_spformer_unet_forward_features_decoder_head(model: SPFormerUNetSegmentation, data: Dict[str, Tensor]) -> None:
    bottleneck, skips = model.forward_features(data["x"], data["pos_grid"], data["batch"])
    assert len(skips) == len(CHANNELS) - 1
    assert bottleneck.features.shape[1] == CHANNELS[-1]
    sparse_x = model.forward_decoder(bottleneck, skips)
    assert sparse_x.features.shape[1] == CHANNELS[0]
    logits = model.forward_head(sparse_x)
    assert logits.shape == (data["pos_grid"].shape[0], model.num_classes)


def test_spformer_unet_forward_head_pre_logits(model: SPFormerUNetSegmentation, data: Dict[str, Tensor]) -> None:
    bottleneck, skips = model.forward_features(data["x"], data["pos_grid"], data["batch"])
    sparse_x = model.forward_decoder(bottleneck, skips)
    feats = model.forward_head(sparse_x, pre_logits=True)
    assert torch.equal(feats, sparse_x.features)
    assert feats.shape[1] == CHANNELS[0]


def test_spformer_unet_encoder_decoder_roundtrip(data: Dict[str, Tensor]) -> None:
    encoder = SPFormerUNetEncoder(6, CHANNELS, LAYERS, spatial_padding=64).cuda()
    decoder = SPFormerUNetDecoder(CHANNELS, LAYERS).cuda()
    bottleneck, skips = encoder(data["x"], data["pos_grid"], data["batch"], return_intermediates=True)
    out = decoder(bottleneck, skips)
    assert len(skips) == len(CHANNELS) - 1
    assert out.features.shape[1] == CHANNELS[0]
