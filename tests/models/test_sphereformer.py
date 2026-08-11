import os
from pathlib import Path
from typing import Dict

import pytest
import torch
from torch import Tensor

from torch_pointcloud.config import DATA_DIR, MODELS_DIR
from torch_pointcloud.models.sphereformer import (
    SphereFormerSegmentation,
    SphereFormerUBlock,
    WindowedRelPosAttention,
    exponential_split,
)
from torch_pointcloud.utils.imports import (
    _CUDA_AVAILABLE,
    _SPCONV_AVAILABLE,
    _SPTR_AVAILABLE,
    _TORCH_SCATTER_AVAILABLE,
)

# See: https://docs.pytest.org/en/stable/how-to/skipping.html#summary
requires_cuda = pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA is not available")
requires_spconv = pytest.mark.skipif(not _SPCONV_AVAILABLE, reason="spconv is not installed")
requires_torch_scatter = pytest.mark.skipif(not _TORCH_SCATTER_AVAILABLE, reason="torch_scatter is not installed")
requires_sptr = pytest.mark.skipif(not _SPTR_AVAILABLE, reason="sptr is not installed")

_WEIGHTS = Path(MODELS_DIR, "sphereformer", "sphereformer.semantickitti.pt")
_SEMANTICKITTI = Path(DATA_DIR, "SemanticKITTI", "raw", "sequences", "08", "velodyne")


@pytest.fixture
def data() -> Dict[str, Tensor]:
    torch.manual_seed(42)
    lengths = torch.tensor([400, 600])
    num_points = int(lengths.sum())
    pos = (torch.rand(num_points, 3) * 2 - 1) * torch.tensor([20.0, 20.0, 3.0])
    pos_grid = ((pos - pos.min(0).values) / 0.05).floor().long()
    x = torch.cat([pos, torch.rand(num_points, 1)], dim=1)
    batch = torch.repeat_interleave(torch.arange(len(lengths)), lengths)
    return dict(
        x=x.cuda(),
        pos=pos.cuda(),
        pos_grid=pos_grid.cuda(),
        batch=batch.cuda(),
    )


@pytest.fixture
def model_seg() -> SphereFormerSegmentation:
    # The sptr kernel requires head_dim == 16, and each window-attention branch needs at least one head
    # (num_heads >= 2), so base_channels must be >= 32.
    return SphereFormerSegmentation(
        in_channels=4,
        num_classes=10,
        base_channels=32,
        layers=(32, 64, 128),
        block_reps=1,
        head_dim=16,
        min_spatial_shape=64,
    ).cuda()


@requires_cuda
@requires_spconv
@requires_torch_scatter
@requires_sptr
def test_sphereformer_segmentation_forward(model_seg: SphereFormerSegmentation, data: Dict[str, Tensor]) -> None:
    model_seg.eval()
    logits = model_seg(data["x"], data["pos"], data["pos_grid"], data["batch"])
    assert logits.shape == (data["pos_grid"].shape[0], model_seg.num_classes)
    assert torch.isfinite(logits).all()


@requires_cuda
@requires_spconv
@requires_torch_scatter
@requires_sptr
def test_sphereformer_forward_keyword_order(model_seg: SphereFormerSegmentation, data: Dict[str, Tensor]) -> None:
    model_seg.eval()
    logits = model_seg(x=data["x"], pos=data["pos"], pos_grid=data["pos_grid"], batch=data["batch"])
    assert logits.shape == (data["pos_grid"].shape[0], model_seg.num_classes)


@requires_cuda
@requires_spconv
@requires_torch_scatter
@requires_sptr
def test_sphereformer_forward_head_pre_logits(model_seg: SphereFormerSegmentation, data: Dict[str, Tensor]) -> None:
    model_seg.eval()
    sparse_x = model_seg.forward_features(data["x"], data["pos"], data["pos_grid"], data["batch"])
    sparse_x = model_seg.forward_decoder(sparse_x)
    feats = model_seg.forward_head(sparse_x, pre_logits=True)
    assert torch.equal(feats, sparse_x.features)
    assert feats.shape == (data["pos_grid"].shape[0], model_seg.base_channels)


@requires_cuda
@requires_spconv
@requires_torch_scatter
@requires_sptr
def test_sphereformer_segmentation_reset_classifier(
    model_seg: SphereFormerSegmentation, data: Dict[str, Tensor]
) -> None:
    model_seg.reset_classifier(num_classes=7)
    model_seg.cuda().eval()
    logits = model_seg(data["x"], data["pos"], data["pos_grid"], data["batch"])
    assert logits.shape == (data["pos_grid"].shape[0], 7)


@requires_cuda
@requires_spconv
@requires_torch_scatter
@requires_sptr
@pytest.mark.skipif(not _WEIGHTS.exists(), reason=f"pretrained weights not found at {_WEIGHTS}")
@pytest.mark.skipif(not _SEMANTICKITTI.is_dir(), reason="SemanticKITTI not available")
def test_sphereformer_semantickitti_smoke() -> None:
    """Load the pretrained checkpoint and run a single SemanticKITTI scan end-to-end."""
    from torch_pointcloud.datasets import SemanticKITTI
    from torch_pointcloud.models import create_model
    from torch_pointcloud.utils.data import DataKeys, collate

    model, info = create_model("sphereformer.semantickitti", task="segmentation", pretrained=True, return_info=True)
    model = model.cuda().eval()

    dataset = SemanticKITTI(root=os.fspath(Path(DATA_DIR)), split="val", transform=info["transform"])
    sample = collate([dataset[0]])

    x = sample[DataKeys.X].cuda()
    pos = sample[DataKeys.POS].cuda()
    pos_grid = sample[DataKeys.POS_GRID].cuda()
    batch = sample[DataKeys.BATCH].cuda()

    with torch.inference_mode():
        logits = model(x, pos, pos_grid, batch)

    assert logits.shape == (x.shape[0], model.num_classes)
    assert torch.isfinite(logits).all()


def _radial_bins(gaps: Tensor, offset: int = 24) -> Tensor:
    """Run `exponential_split` on query/key pairs whose radial gaps are exactly `gaps`."""
    radii = torch.cat([gaps, gaps.new_zeros(1)])
    pos = torch.zeros(radii.numel(), 3)
    pos[:, 2] = radii
    index_query = torch.arange(gaps.numel())
    index_key = torch.full((gaps.numel(),), gaps.numel())
    relative_position_index = torch.zeros(gaps.numel(), 3, dtype=torch.long)
    out = exponential_split(pos, index_query, index_key, relative_position_index, offset=offset)
    return out[:, 2]


@pytest.mark.parametrize(
    ("gap", "expected"),
    [
        pytest.param(0.0, 24, id="zero-gap"),
        pytest.param(0.0124, 24, id="just-below-base-bin-width"),
        pytest.param(0.0125, 25, id="base-bin-width-boundary"),
        pytest.param(-0.001, 23, id="first-negative-bin"),
        pytest.param(-0.0125, 22, id="negative-base-bin-width-boundary"),
        pytest.param(76.7, 46, id="second-to-last-positive-bin"),
        pytest.param(102.37, 47, id="last-positive-bin"),
        pytest.param(102.38, 47, id="beyond-last-positive-bin-clamped"),
        pytest.param(-102.37, 0, id="last-negative-bin"),
        pytest.param(-102.38, 0, id="beyond-last-negative-bin-clamped"),
    ],
)
def test_exponential_split_bin_boundaries(gap: float, expected: int) -> None:
    bins = _radial_bins(torch.tensor([gap]))
    assert bins.item() == expected


def test_exponential_split_nuscenes_window_stays_in_table() -> None:
    # The nuscenes config pairs a 120 m radial window with a 48-row radial table (offset 24);
    # every gap inside the window must index a valid row.
    gaps = torch.linspace(-120.0, 120.0, 4001)
    bins = _radial_bins(gaps, offset=24)
    assert int(bins.min()) >= 0
    assert int(bins.max()) <= 47


def test_exponential_split_overwrites_radial_column_only() -> None:
    gaps = torch.tensor([0.5, -3.0])
    radii = torch.cat([gaps, gaps.new_zeros(1)])
    pos = torch.zeros(3, 3)
    pos[:, 2] = radii
    relative_position_index = torch.full((2, 3), 7, dtype=torch.long)
    out = exponential_split(pos, torch.arange(2), torch.full((2,), 2), relative_position_index)
    assert out is relative_position_index
    assert torch.equal(out[:, 0], torch.full((2,), 7))
    assert torch.equal(out[:, 1], torch.full((2,), 7))


@requires_spconv
def test_sphereformer_ublock_head_dim_validation() -> None:
    window = torch.tensor([0.3, 0.3, 0.3])
    with pytest.raises(ValueError, match="must be divisible by `head_dim`"):
        SphereFormerUBlock(
            planes=(24,),
            block_reps=1,
            window_size=window,
            window_size_sphere=window,
            quant_size=window / 24,
            quant_size_sphere=window / 24,
            head_dim=16,
        )


@pytest.mark.skipif(_SPTR_AVAILABLE, reason="sptr is installed")
def test_windowed_attention_without_sptr_raises_import_error() -> None:
    window = torch.tensor([0.3, 0.3, 0.3])
    with pytest.raises(ImportError, match="sptr"):
        WindowedRelPosAttention(
            embed_dim=32,
            num_heads=2,
            window_size=window,
            window_size_sphere=window,
            quant_size=window / 24,
            quant_size_sphere=window / 24,
        )
