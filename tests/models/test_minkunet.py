from typing import Dict

import pytest
import torch
from torch import Tensor, nn

from torch_pointcloud.models.minkunet import MinkUNetSegmentation
from torch_pointcloud.utils.imports import _CUDA_AVAILABLE, _SPCONV_AVAILABLE

pytestmark = pytest.mark.skipif(not _SPCONV_AVAILABLE, reason="spconv is not installed")
_REQUIRES_CUDA = pytest.mark.skipif(not _CUDA_AVAILABLE, reason="spconv convolutions need CUDA")


@pytest.fixture
def model() -> MinkUNetSegmentation:
    return MinkUNetSegmentation(
        in_channels=4,
        num_classes=10,
        stem_channels=8,
        encoder_channels=(8, 16, 32, 64),
        decoder_channels=(64, 32, 16, 16),
        encoder_depths=(1, 2, 1, 1),
        decoder_depths=(1, 1, 1, 1),
    )


@pytest.fixture
def data() -> Dict[str, Tensor]:
    torch.manual_seed(42)
    lengths = torch.tensor([256, 512])
    pos_grid = torch.randint(0, 64, (int(lengths.sum()), 3))
    x = torch.randn(int(lengths.sum()), 4)
    batch = torch.repeat_interleave(torch.arange(len(lengths)), lengths)
    return dict(x=x.cuda(), pos_grid=pos_grid.cuda(), batch=batch.cuda())


def test_minkunet_stage_layout(model: MinkUNetSegmentation) -> None:
    # The strided convolution keeps the width of the previous stage; the first residual block widens it.
    shapes = {key: tuple(value.shape) for key, value in model.state_dict().items()}
    assert shapes["encoder.conv_input.1.0.weight"] == (8, 3, 3, 3, 8)
    assert shapes["encoder.down.1.0.weight"] == (8, 2, 2, 2, 8)
    assert shapes["encoder.enc.1.block0.conv1.weight"] == (16, 3, 3, 3, 8)
    assert shapes["encoder.enc.1.block0.proj.0.weight"] == (16, 1, 1, 1, 8)
    assert "encoder.enc.1.block1.conv1.weight" in shapes
    assert "encoder.enc.1.block2.conv1.weight" not in shapes
    assert shapes["decoder.up.0.0.weight"] == (16, 2, 2, 2, 16)
    assert shapes["decoder.dec.0.block0.conv1.weight"] == (16, 3, 3, 3, 16 + 8)


def test_minkunet_head(model: MinkUNetSegmentation) -> None:
    assert model.num_features == 16
    assert isinstance(model.head, nn.Linear)
    assert model.head.out_features == 10


def test_minkunet_reset_classifier(model: MinkUNetSegmentation) -> None:
    model.reset_classifier(num_classes=0)
    assert isinstance(model.head, nn.Identity)
    model.reset_classifier(num_classes=3)
    assert isinstance(model.head, nn.Linear)
    assert model.head.out_features == 3


def test_minkunet_rejects_mismatched_stages() -> None:
    with pytest.raises(ValueError, match="decoder_channels"):
        MinkUNetSegmentation(in_channels=4, num_classes=2, decoder_channels=(64, 32, 16))
    with pytest.raises(ValueError, match="encoder_depths"):
        MinkUNetSegmentation(in_channels=4, num_classes=2, encoder_depths=(2, 2))


@_REQUIRES_CUDA
def test_minkunet_forward(model: MinkUNetSegmentation, data: Dict[str, Tensor]) -> None:
    model = model.cuda()
    logits = model(data["x"], data["pos_grid"], data["batch"])
    assert logits.shape == (data["pos_grid"].shape[0], 10)
    logits.sum().backward()


@_REQUIRES_CUDA
def test_minkunet_forward_without_features(data: Dict[str, Tensor]) -> None:
    model = MinkUNetSegmentation(in_channels=3, num_classes=2, encoder_depths=(1, 1, 1, 1)).cuda()
    logits = model(None, data["pos_grid"], data["batch"])
    assert logits.shape == (data["pos_grid"].shape[0], 2)


@_REQUIRES_CUDA
def test_minkunet_rows_follow_the_input(model: MinkUNetSegmentation, data: Dict[str, Tensor]) -> None:
    model = model.cuda().eval()
    perm = torch.randperm(data["pos_grid"].shape[0], device="cuda")
    with torch.no_grad():
        logits = model(data["x"], data["pos_grid"], data["batch"])
        logits_perm = model(data["x"][perm], data["pos_grid"][perm], data["batch"][perm])
    torch.testing.assert_close(logits_perm, logits[perm], atol=1e-4, rtol=1e-4)


@_REQUIRES_CUDA
def test_minkunet_features_decoder_head(model: MinkUNetSegmentation, data: Dict[str, Tensor]) -> None:
    model = model.cuda()
    sparse_x, skips = model.forward_features(data["x"], data["pos_grid"], data["batch"], return_intermediates=True)
    assert len(skips) == 4
    assert [skip["x"].shape[1] for skip in skips] == [8, 8, 16, 32]
    features = model.forward_decoder(sparse_x, skips)
    assert features.shape == (data["pos_grid"].shape[0], model.num_features)
    assert torch.equal(model.forward_head(features, pre_logits=True), features)
    assert model.forward_head(features).shape == (data["pos_grid"].shape[0], 10)


def test_minkunet_head_stages_widen_the_head() -> None:
    model = MinkUNetSegmentation(in_channels=4, num_classes=5, head_stages=(0, 2, 4))
    assert model.num_features == 256 + 128 + 96
    assert isinstance(model.head, nn.Linear)
    assert model.head.in_features == 256 + 128 + 96


def test_minkunet_rejects_unknown_head_stages() -> None:
    with pytest.raises(ValueError, match="head_stages"):
        MinkUNetSegmentation(in_channels=4, num_classes=2, head_stages=(0, 5))


@_REQUIRES_CUDA
def test_minkunet_head_stages_concatenate_the_decoding_stages(data: Dict[str, Tensor]) -> None:
    torch.manual_seed(0)
    plain = MinkUNetSegmentation(in_channels=4, num_classes=0, encoder_depths=(1, 1, 1, 1)).cuda().eval()
    multiscale = (
        MinkUNetSegmentation(in_channels=4, num_classes=0, encoder_depths=(1, 1, 1, 1), head_stages=(0, 2, 4))
        .cuda()
        .eval()
    )
    multiscale.load_state_dict(plain.state_dict())
    with torch.no_grad():
        features = plain(data["x"], data["pos_grid"], data["batch"])
        features_multiscale = multiscale(data["x"], data["pos_grid"], data["batch"])
    assert features_multiscale.shape == (data["pos_grid"].shape[0], 256 + 128 + 96)
    torch.testing.assert_close(features_multiscale[:, -96:], features)


@_REQUIRES_CUDA
def test_minkunet_headless(data: Dict[str, Tensor]) -> None:
    model = MinkUNetSegmentation(in_channels=4, num_classes=0).cuda()
    features = model(data["x"], data["pos_grid"], data["batch"])
    assert features.shape == (data["pos_grid"].shape[0], model.num_features)
