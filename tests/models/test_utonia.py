from typing import Dict

import pytest
import torch
from torch import Tensor

from torch_pointcloud.models.utonia import UtoniaSegmentation
from torch_pointcloud.utils.imports import _CUDA_AVAILABLE, _SPCONV_AVAILABLE, _TORCH_SCATTER_AVAILABLE

# See: https://docs.pytest.org/en/stable/how-to/skipping.html#summary
pytestmark = [
    pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA is not available"),
    pytest.mark.skipif(not _SPCONV_AVAILABLE, reason="spconv is not installed"),
    pytest.mark.skipif(not _TORCH_SCATTER_AVAILABLE, reason="torch-scatter is not installed"),
]


@pytest.fixture
def data() -> Dict[str, Tensor]:
    torch.manual_seed(42)
    lengths = torch.tensor([256, 512])
    pos_grid = torch.randint(0, 64, (int(lengths.sum()), 3))
    pos = pos_grid.float() * 0.02
    x = torch.randn(int(lengths.sum()), 6)
    batch = torch.repeat_interleave(torch.arange(len(lengths)), lengths)
    return dict(
        x=x.cuda(),
        pos=pos.cuda(),
        pos_grid=pos_grid.cuda(),
        batch=batch.cuda(),
    )


@pytest.fixture
def model() -> UtoniaSegmentation:
    return UtoniaSegmentation(
        in_channels=6,
        num_classes=10,
        serialization_orders=("z", "z-trans", "hilbert", "hilbert-trans"),
        shuffle_serialization_orders=True,
        strides=(2, 2, 2, 2),
        encoder_depths=(1, 1, 1, 1, 1),
        encoder_channels=(12, 24, 48, 96, 192),
        encoder_num_heads=(1, 2, 4, 8, 16),
        encoder_patch_size=(16, 16, 16, 16, 16),
        norm="layer_norm",
        act="gelu",
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        use_flash_attn=False,
        upcast_attn=False,
        upcast_softmax=False,
        rope_base=10.0,
        dropout=0.0,
        pooling="grid",
        stem_type="linear",
        act_kwargs=None,
        norm_kwargs=None,
    ).cuda()


def test_utonia_segmentation_forward(model: UtoniaSegmentation, data: Dict[str, Tensor]) -> None:
    logits = model(data["x"], data["pos"], data["pos_grid"], data["batch"])
    assert logits.shape == (data["pos_grid"].shape[0], model.num_classes)


def test_utonia_segmentation_reset_classifier(model: UtoniaSegmentation, data: Dict[str, Tensor]) -> None:
    model.reset_classifier(num_classes=42)
    model.cuda()
    logits = model(data["x"], data["pos"], data["pos_grid"], data["batch"])
    assert logits.shape == (data["pos_grid"].shape[0], 42)


def test_utonia_segmentation_forward_features_decoder_head(model: UtoniaSegmentation, data: Dict[str, Tensor]) -> None:
    x, _, _, intermediates = model.forward_features(
        data["x"], data["pos_grid"], data["batch"], return_intermediates=True, pos=data["pos"]
    )
    assert len(intermediates) > 0
    x, _, _ = model.forward_decoder(x, intermediates)
    logits = model.forward_head(x)
    assert logits.shape[1] == model.num_classes


def test_utonia_segmentation_default_norm_no_batch_leak() -> None:
    torch.manual_seed(0)
    model = UtoniaSegmentation(
        in_channels=6,
        num_classes=5,
        encoder_depths=(1, 1, 1, 1, 1),
        encoder_channels=(12, 24, 48, 96, 192),
        encoder_num_heads=(1, 2, 4, 8, 16),
        encoder_patch_size=(16, 16, 16, 16, 16),
        use_flash_attn=False,
    ).cuda()
    model.eval()

    lengths = (128, 160)
    pos_grids = [torch.randint(0, 64, (n, 3)) for n in lengths]
    for pos_grid in pos_grids:
        pos_grid[0] = torch.tensor([63, 63, 63])  # same serialization depth batched and separate
    xs = [torch.randn(n, 6) for n in lengths]
    batch = torch.repeat_interleave(torch.arange(2), torch.tensor(lengths))

    with torch.no_grad():
        batched = model(
            torch.cat(xs).cuda(),
            torch.cat(pos_grids).float().cuda() * 0.02,
            torch.cat(pos_grids).cuda(),
            batch.cuda(),
        )
        separate = model(
            xs[0].cuda(),
            pos_grids[0].float().cuda() * 0.02,
            pos_grids[0].cuda(),
            torch.zeros(lengths[0], dtype=torch.long).cuda(),
        )
    assert torch.allclose(batched[: lengths[0]], separate, atol=1e-5)
