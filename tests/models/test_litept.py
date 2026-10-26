from typing import Dict, Tuple
from unittest.mock import Mock

import pytest
import torch
from torch import Tensor, nn

from torch_pointcloud.models import model_info
from torch_pointcloud.models.litept import LitePTBlock, LitePTSegmentation, segmentation_checkpoint_filter_fn
from torch_pointcloud.utils.data import DataKeys
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
def test_litept_segmentation_headless_forward(data: Dict[str, Tensor]) -> None:
    model = _litept((0, 0))
    model.reset_classifier(num_classes=0)
    model.cuda()
    features = model(data["x"], data["pos_grid"], data["batch"])
    assert features.shape == (data["x"].shape[0], model.num_features)


@requires_spconv
def test_litept_segmentation_headless() -> None:
    model = _litept((0, 0))
    model.reset_classifier(num_classes=0)
    assert isinstance(model.head, nn.Identity)
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


@pytest.mark.parametrize(
    "name",
    [
        pytest.param("litept-small.scannet20.prs-eth", id="scannet20"),
        pytest.param("litept-small.structured3d.prs-eth", id="structured3d-small"),
        pytest.param("litept-base.structured3d.prs-eth", id="structured3d-base"),
        pytest.param("litept-large.structured3d.prs-eth", id="structured3d-large"),
    ],
)
def test_litept_indoor_registered_transform_runs_without_labels(name: str) -> None:
    """The indoor transform runs on an unlabeled cloud and feeds colors in $[0, 1]$, the scale of the training data."""
    transform = model_info(name)["transform"]
    assert transform is not None
    torch.manual_seed(0)
    color = torch.randint(0, 256, (500, 3), dtype=torch.uint8)
    color[0] = 255
    data = {
        DataKeys.POS: torch.rand(500, 3) * 2.0,
        DataKeys.COLOR: color,
        DataKeys.NORMAL: nn.functional.normalize(torch.randn(500, 3), dim=1),
    }

    out = transform(data)

    assert DataKeys.SEGMENT not in out
    assert out[DataKeys.X].shape == (out[DataKeys.POS_GRID].size(0), 6)
    assert out[DataKeys.INVERSE].shape == (500,)
    assert out[DataKeys.X][:, :3].min() >= 0.0
    assert out[DataKeys.X][:, :3].max() == 1.0


@pytest.mark.parametrize(
    "name",
    [
        pytest.param("litept-small.nuscenes.prs-eth", id="small"),
        pytest.param("litept-large.nuscenes.prs-eth", id="large"),
    ],
)
def test_litept_nuscenes_registered_transform_runs_without_labels(name: str) -> None:
    """The nuScenes transform runs on an unlabeled scan and scales the intensities to $[0, 1]$."""
    transform = model_info(name)["transform"]
    assert transform is not None
    torch.manual_seed(0)
    intensity = torch.randint(0, 256, (500, 1)).float()
    intensity[0] = 255.0
    data = {
        DataKeys.POS: torch.rand(500, 3) * 50.0,
        DataKeys.INTENSITY: intensity,
    }

    out = transform(data)

    assert DataKeys.SEGMENT not in out
    assert out[DataKeys.X].shape == (out[DataKeys.POS_GRID].size(0), 4)
    assert out[DataKeys.INVERSE].shape == (500,)
    assert out[DataKeys.X][:, 3].min() >= 0.0
    assert out[DataKeys.X][:, 3].max() == 1.0
