import pytest
import torch

from torch_pointcloud.models import create_model, list_models
from torch_pointcloud.models.voxel_mamba import (
    VoxelMambaDetection,
    build_hilbert_template,
    hilbert_serialize,
)
from torch_pointcloud.utils.box3d import nms3d
from torch_pointcloud.utils.imports import (
    _CUDA_AVAILABLE,
    _MAMBA_SSM_AVAILABLE,
    _SPCONV_AVAILABLE,
    _TORCH_SCATTER_AVAILABLE,
)

RANGE = (-74.88, -74.88, -2.0, 74.88, 74.88, 4.0)
DEVICE = "cuda" if _CUDA_AVAILABLE else "cpu"

_FULL_STACK = _CUDA_AVAILABLE and _MAMBA_SSM_AVAILABLE and _SPCONV_AVAILABLE and _TORCH_SCATTER_AVAILABLE


def test_build_hilbert_template_is_a_permutation() -> None:
    """A truncated Hilbert template indexes each kept voxel to a distinct curve position."""
    template = build_hilbert_template(rank=4, z_max=4)
    assert template.shape == (16 * 16 * 4,)
    assert template.dtype == torch.long
    # Every kept voxel maps to a unique Hilbert position (the curve visits each cell once).
    assert torch.unique(template).numel() == template.numel()


def test_hilbert_serialize_round_trip() -> None:
    """`forward[inverse]` is the identity, so Mamba outputs scatter back to voxel order."""
    template = build_hilbert_template(rank=4, z_max=4)
    n = 200
    torch.manual_seed(0)
    coords = torch.stack(
        [
            torch.zeros(n, dtype=torch.long),
            torch.randint(0, 4, (n,)),
            torch.randint(0, 16, (n,)),
            torch.randint(0, 16, (n,)),
        ],
        dim=1,
    )
    forward, inverse = hilbert_serialize(template, coords, batch_size=1, rank=4, shift=0)
    assert torch.equal(forward[0][inverse[0]], torch.arange(n))
    # The forward order sorts voxels by ascending Hilbert position.
    flat = coords[:, 1] * 16 * 16 + coords[:, 2] * 16 + coords[:, 3]
    positions = template[flat]
    assert torch.equal(positions[forward[0]], torch.sort(positions).values)


def test_voxel_mamba_registered_variant() -> None:
    assert "voxel-mamba.waymo" in list_models("voxel-mamba*", task="detection")


@pytest.mark.skipif(not (_MAMBA_SSM_AVAILABLE and _SPCONV_AVAILABLE), reason="mamba_ssm or spconv is not installed")
def test_voxel_mamba_create_model_hparams() -> None:
    model = create_model("voxel-mamba.waymo", task="detection")
    assert isinstance(model, VoxelMambaDetection)
    assert model.in_channels == 5
    assert model.num_classes == 3
    assert model.grid_size == (468, 468, 32)
    assert model.backbone_3d.sparse_shape == [33, 468, 468]
    assert model.backbone.num_bev_features == 384
    assert len(model.backbone_3d.block_list) == 6


def _make_inputs(n_per_scene: int = 6000, batch_size: int = 1) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    pos, x, batch = [], [], []
    for i in range(batch_size):
        p = torch.rand(n_per_scene, 3)
        for d in range(3):
            p[:, d] = p[:, d] * (RANGE[d + 3] - RANGE[d]) + RANGE[d]
        pos.append(p)
        x.append(torch.rand(n_per_scene, 2))
        batch.append(torch.full((n_per_scene,), i, dtype=torch.long))
    return torch.cat(pos).to(DEVICE), torch.cat(x).to(DEVICE), torch.cat(batch).to(DEVICE)


@pytest.mark.skipif(not _FULL_STACK, reason="Voxel Mamba forward needs CUDA + mamba_ssm + spconv + torch_scatter")
def test_voxel_mamba_forward_shapes() -> None:
    model = create_model("voxel-mamba.waymo", task="detection").to(DEVICE).eval()
    assert isinstance(model, VoxelMambaDetection)
    pos, x, batch = _make_inputs()
    with torch.no_grad():
        out = model(x, pos, batch)
    nx = ny = 468
    assert out["heatmap"].shape == (1, 3, ny, nx)
    assert out["center"].shape == (1, 2, ny, nx)
    assert out["dim"].shape == (1, 3, ny, nx)
    assert out["rot"].shape == (1, 2, ny, nx)
    assert torch.isfinite(out["heatmap"]).all()


@pytest.mark.skipif(not _FULL_STACK, reason="Voxel Mamba forward needs CUDA + mamba_ssm + spconv + torch_scatter")
def test_voxel_mamba_decode() -> None:
    model = create_model("voxel-mamba.waymo", task="detection").to(DEVICE).eval()
    assert isinstance(model, VoxelMambaDetection)
    pos, x, batch = _make_inputs()
    with torch.no_grad():
        out = model(x, pos, batch)
        det = model.decode(out, score_threshold=0.1, top_k=100)
        raw = model.decode(out, top_k=100)
        idx = nms3d(det["boxes"], det["scores"], 0.7, labels=det["labels"], batch=det["batch"])
    assert det["boxes"].shape[1] == 7
    assert det["scores"].shape[0] == det["boxes"].shape[0] == det["labels"].shape[0] == det["batch"].shape[0]
    assert idx.numel() <= det["boxes"].shape[0]
    assert torch.isfinite(det["boxes"]).all()
    # The default decode is non-filtering: it returns every gathered peak (the eval layer thresholds).
    assert raw["boxes"].shape[0] == 100
    assert raw["boxes"].shape[0] >= det["boxes"].shape[0]


@pytest.mark.skipif(not (_MAMBA_SSM_AVAILABLE and _SPCONV_AVAILABLE), reason="mamba_ssm or spconv is not installed")
def test_voxel_mamba_registered_without_pretrained_weights() -> None:
    """Voxel Mamba has no public trained weights, so `pretrained=True` warns and returns the model unloaded."""
    with pytest.warns(UserWarning, match="No pretrained weights"):
        model = create_model("voxel-mamba.waymo", task="detection", pretrained=True)
    assert isinstance(model, VoxelMambaDetection)
