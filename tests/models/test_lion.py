from pathlib import Path
from typing import Sequence

import pytest
import torch

from torch_pointcloud.config import MODELS_DIR
from torch_pointcloud.models import create_model, list_models
from torch_pointcloud.models.lion import LIONDetection, TransFusionHead
from torch_pointcloud.utils.imports import (
    _CUDA_AVAILABLE,
    _MAMBA_SSM_AVAILABLE,
    _SPCONV_AVAILABLE,
    _TORCH_SCATTER_AVAILABLE,
)

RANGE = (-54.0, -54.0, -5.0, 54.0, 54.0, 3.0)
DEVICE = "cuda" if _CUDA_AVAILABLE else "cpu"

_FULL_STACK = _CUDA_AVAILABLE and _MAMBA_SSM_AVAILABLE and _SPCONV_AVAILABLE and _TORCH_SCATTER_AVAILABLE
_WEIGHTS = Path(MODELS_DIR, "lion", "lion-mamba-happinesslz.nuscenes.pt")


def test_lion_registered_variant() -> None:
    assert "lion-mamba-happinesslz.nuscenes" in list_models("lion*", task="detection")


@pytest.mark.skipif(not _SPCONV_AVAILABLE, reason="spconv is not installed")
def test_lion_create_model_hparams() -> None:
    model = create_model("lion-mamba-happinesslz.nuscenes", task="detection")
    assert isinstance(model, LIONDetection)
    assert model.in_channels == 5
    assert model.num_classes == 10
    assert model.grid_size == (360, 360, 32)
    assert model.backbone_3d.sparse_shape == [32, 360, 360]
    assert model.backbone.num_bev_features == 384
    assert model.head.feature_map_stride == 2


def test_lion_head_decode_packs_detections() -> None:
    """`TransFusionHead.decode` turns raw predictions into PyG-packed detections without needing CUDA."""
    torch.manual_seed(0)
    head = TransFusionHead(384, 10, (360, 360, 32), RANGE, (0.3, 0.3, 0.25)).eval()
    num_proposals = head.num_proposals
    batch_size = 2
    out = {
        "center": torch.rand(batch_size, 2, num_proposals),
        "height": torch.rand(batch_size, 1, num_proposals),
        "dim": torch.rand(batch_size, 3, num_proposals),
        "rot": torch.randn(batch_size, 2, num_proposals),
        "vel": torch.randn(batch_size, 2, num_proposals),
        "iou": torch.rand(batch_size, 1, num_proposals),
        "heatmap": torch.randn(batch_size, 10, num_proposals),
        "query_heatmap_score": torch.rand(batch_size, 10, num_proposals),
        "query_labels": torch.randint(0, 10, (batch_size, num_proposals)),
    }
    det = head.decode(out)
    assert det["boxes"].shape[1] == 7
    assert det["scores"].shape[0] == det["boxes"].shape[0] == det["labels"].shape[0] == det["batch"].shape[0]
    assert torch.isfinite(det["boxes"]).all()
    assert set(det["batch"].tolist()) <= {0, 1}


def _make_inputs(scene_sizes: Sequence[int] = (30000,)) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    pos, x, batch = [], [], []
    for i, num_points in enumerate(scene_sizes):
        p = torch.rand(num_points, 3)
        for d in range(3):
            p[:, d] = p[:, d] * (RANGE[d + 3] - RANGE[d]) + RANGE[d]
        pos.append(p)
        x.append(torch.rand(num_points, 2))
        batch.append(torch.full((num_points,), i, dtype=torch.long))
    return torch.cat(pos).to(DEVICE), torch.cat(x).to(DEVICE), torch.cat(batch).to(DEVICE)


@pytest.mark.skipif(not _FULL_STACK, reason="LION forward needs CUDA + mamba_ssm + spconv + torch_scatter")
def test_lion_forward_and_decode() -> None:
    model = create_model("lion-mamba-happinesslz.nuscenes", task="detection").to(DEVICE).eval()
    assert isinstance(model, LIONDetection)
    pos, x, batch = _make_inputs()
    with torch.no_grad():
        out = model(x, pos, batch)
        det = model.decode(out)
    assert out["heatmap"].shape == (1, 10, model.head.num_proposals)
    assert out["dense_heatmap"].shape[1] == 10
    assert torch.isfinite(out["heatmap"]).all()
    assert det["boxes"].shape[1] == 7
    assert det["scores"].shape[0] == det["boxes"].shape[0] == det["labels"].shape[0] == det["batch"].shape[0]
    assert torch.isfinite(det["boxes"]).all()

    # decode must stay a pure function of the output dict, even after a later forward
    with torch.no_grad():
        model(torch.rand_like(x), pos, batch)
        det_again = model.decode(out)
    assert torch.equal(det_again["boxes"], det["boxes"])
    assert torch.equal(det_again["scores"], det["scores"])
    assert torch.equal(det_again["labels"], det["labels"])


@pytest.mark.skipif(not _FULL_STACK, reason="LION forward needs CUDA + mamba_ssm + spconv + torch_scatter")
@pytest.mark.skipif(not _WEIGHTS.exists(), reason="LION nuScenes weights not in local cache")
def test_lion_pretrained_smoke() -> None:
    """Pretrained LION strict-loads and produces finite, in-range detections on a random scene."""
    model = create_model("lion-mamba-happinesslz.nuscenes", task="detection", pretrained=True).to(DEVICE).eval()
    assert isinstance(model, LIONDetection)
    pos, x, batch = _make_inputs()
    with torch.no_grad():
        det = model.decode(model(x, pos, batch))
    assert torch.isfinite(det["scores"]).all()
    assert (det["labels"] >= 0).all() and (det["labels"] < 10).all()
