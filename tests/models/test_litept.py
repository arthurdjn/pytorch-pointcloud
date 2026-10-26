from typing import Dict, Tuple
from unittest.mock import Mock

import pytest
import torch
from torch import Tensor, nn

from torch_pointcloud.models.litept import LitePTBlock, LitePTSegmentation, segmentation_checkpoint_filter_fn
from torch_pointcloud.utils.imports import _CUDA_AVAILABLE, _SPCONV_AVAILABLE

requires_spconv = pytest.mark.skipif(not _SPCONV_AVAILABLE, reason="spconv is not installed")
requires_cuda = pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA is not available")

# Released checkpoint keys of a model with two decoder stages, and the `LitePTSegmentation` keys they load into.
RELEASED_KEYS = {
    "module.seg_head.weight": "head.weight",
    "module.backbone.embedding.stem.conv.weight": "encoder.stem.stem.weight",
    "module.backbone.embedding.stem.norm.running_mean": "encoder.stem.norm.module.running_mean",
    "module.backbone.enc.enc1.down.proj.weight": "encoder.blocks.1.downsample.proj.weight",
    "module.backbone.enc.enc1.down.norm.0.running_var": "encoder.blocks.1.downsample.norm.module.running_var",
    "module.backbone.enc.enc0.block0.conv.0.weight": "encoder.blocks.0.blocks.0.cpe_conv.weight",
    "module.backbone.enc.enc0.block0.conv.1.bias": "encoder.blocks.0.blocks.0.cpe_proj.bias",
    "module.backbone.enc.enc0.block0.conv.2.weight": "encoder.blocks.0.blocks.0.cpe_norm.weight",
    "module.backbone.enc.enc2.block1.norm0.0.weight": "encoder.blocks.2.blocks.1.norm0.weight",
    "module.backbone.enc.enc2.block1.norm1.0.bias": "encoder.blocks.2.blocks.1.norm1.bias",
    "module.backbone.enc.enc2.block1.attn.qkv.weight": "encoder.blocks.2.blocks.1.attn.qkv.weight",
    "module.backbone.enc.enc2.block1.attn.proj.bias": "encoder.blocks.2.blocks.1.attn.proj.bias",
    "module.backbone.enc.enc2.block1.norm2.0.weight": "encoder.blocks.2.blocks.1.norm2.weight",
    "module.backbone.enc.enc2.block1.mlp.0.fc1.weight": "encoder.blocks.2.blocks.1.mlp.0.weight",
    "module.backbone.enc.enc2.block1.mlp.0.fc2.bias": "encoder.blocks.2.blocks.1.mlp.3.bias",
    "module.backbone.dec.dec1.up.proj.0.weight": "decoder.blocks.0.upsample.proj.weight",
    "module.backbone.dec.dec1.up.proj.1.running_mean": "decoder.blocks.0.upsample.norm.module.running_mean",
    "module.backbone.dec.dec1.up.proj_skip.0.bias": "decoder.blocks.0.upsample.proj_skip.bias",
    "module.backbone.dec.dec1.up.proj_skip.1.weight": "decoder.blocks.0.upsample.norm_skip.module.weight",
    "module.backbone.dec.dec1.block0.attn.qkv.bias": "decoder.blocks.0.blocks.0.attn.qkv.bias",
    "module.backbone.dec.dec0.up.proj.0.weight": "decoder.blocks.1.upsample.proj.weight",
    "module.backbone.dec.dec0.block0.conv.0.weight": "decoder.blocks.1.blocks.0.cpe_conv.weight",
}


def _litept(decoder_depths: Tuple[int, ...], use_flash_attn: bool = False) -> LitePTSegmentation:
    return LitePTSegmentation(
        in_channels=6,
        num_classes=10,
        strides=(2, 2),
        encoder_depths=(1, 1, 2),
        encoder_channels=(12, 24, 36),
        encoder_num_heads=(2, 4, 6),
        encoder_patch_size=(16, 16, 16),
        encoder_conv=(True, True, False),
        encoder_attention=(False, False, True),
        decoder_depths=decoder_depths,
        decoder_channels=(24, 12),
        decoder_num_heads=(4, 2),
        decoder_patch_size=(16, 16),
        decoder_conv=(False, True),
        decoder_attention=(True, False),
        use_flash_attn=use_flash_attn,
    )


@pytest.fixture
def data() -> Dict[str, Tensor]:
    torch.manual_seed(42)
    lengths = torch.tensor([256, 512])
    pos_grid = torch.randint(0, 64, (int(lengths.sum()), 3))
    x = torch.randn(int(lengths.sum()), 6)
    batch = torch.repeat_interleave(torch.arange(len(lengths)), lengths)
    return dict(
        x=x.cuda(),
        pos_grid=pos_grid.cuda(),
        batch=batch.cuda(),
    )


@requires_cuda
@requires_spconv
@pytest.mark.parametrize("decoder_depths", [(0, 0), (1, 1)], ids=["no-decoder-blocks", "decoder-blocks"])
def test_litept_segmentation_forward(decoder_depths: Tuple[int, ...], data: Dict[str, Tensor]) -> None:
    model = _litept(decoder_depths).cuda()
    logits = model(data["x"], data["pos_grid"], data["batch"])
    assert logits.shape == (data["x"].shape[0], 10)
    assert torch.isfinite(logits).all()


@requires_cuda
@requires_spconv
def test_litept_segmentation_serializes_attention_stages_only(data: Dict[str, Tensor]) -> None:
    model = _litept((0, 0)).cuda()
    _, _, _, intermediates = model.forward_features(
        data["x"], data["pos_grid"], data["batch"], return_intermediates=True
    )
    assert len(intermediates) == 2
    assert all("serialized_order" not in skip for skip in intermediates)


@requires_cuda
@requires_spconv
def test_litept_segmentation_headless(data: Dict[str, Tensor]) -> None:
    model = _litept((0, 0))
    model.reset_classifier(num_classes=0)
    model.cuda()
    assert isinstance(model.head, nn.Identity)
    features = model(data["x"], data["pos_grid"], data["batch"])
    assert features.shape == (data["x"].shape[0], model.num_features)
    assert model.num_features == 12


def test_litept_block_requires_conv_or_attention() -> None:
    with pytest.raises(ValueError, match="conv=False, attention=False"):
        LitePTBlock(channels=12, num_heads=2, conv=False, attention=False)


def test_litept_attention_block_submodules() -> None:
    block = LitePTBlock(channels=12, num_heads=2, conv=False, attention=True, use_flash_attn=False)
    assert hasattr(block, "norm0") and hasattr(block, "attn") and hasattr(block, "mlp")
    assert not hasattr(block, "cpe_conv")


@requires_spconv
def test_litept_conv_block_submodules() -> None:
    block = LitePTBlock(channels=12, num_heads=2, conv=True, attention=False, use_flash_attn=False)
    assert hasattr(block, "cpe_conv") and hasattr(block, "cpe_norm")
    assert not hasattr(block, "attn") and not hasattr(block, "norm0")


def test_segmentation_checkpoint_filter_fn() -> None:
    model = Mock(spec=LitePTSegmentation)
    model.decoder = Mock(blocks=[Mock(), Mock()])
    rope_key = "encoder.blocks.2.blocks.0.attn.rope.inv_freq"
    model.state_dict.return_value = {"head.weight": torch.zeros(1), rope_key: torch.ones(3)}
    state_dict = {key: torch.zeros(1) for key in RELEASED_KEYS}

    converted = segmentation_checkpoint_filter_fn(state_dict, model)

    assert set(converted) == set(RELEASED_KEYS.values()) | {rope_key}
    assert torch.equal(converted[rope_key], torch.ones(3))


@requires_spconv
def test_segmentation_checkpoint_filter_fn_targets_exist() -> None:
    model = _litept((1, 1))
    assert set(RELEASED_KEYS.values()) <= set(model.state_dict())
