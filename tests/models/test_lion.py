from pathlib import Path
from typing import Optional, Sequence

import pytest
import torch
import torch.nn as nn
from torch import Tensor

from torch_pointcloud.config import MODELS_DIR
from torch_pointcloud.models import create_model, list_models
from torch_pointcloud.models.lion import (
    FlattenedWindowMapping,
    LION3DBackbone,
    LIONDetection,
    PatchMerging3D,
    TransFusionHead,
    TransFusionHeadOutput,
)
from torch_pointcloud.utils.imports import (
    _CUDA_AVAILABLE,
    _MAMBA_SSM_AVAILABLE,
    _SPCONV_AVAILABLE,
    _TORCH_SCATTER_AVAILABLE,
)

RANGE = (-54.0, -54.0, -5.0, 54.0, 54.0, 3.0)
DEVICE = "cuda" if _CUDA_AVAILABLE else "cpu"

_FULL_STACK = _CUDA_AVAILABLE and _MAMBA_SSM_AVAILABLE and _SPCONV_AVAILABLE and _TORCH_SCATTER_AVAILABLE
_WEIGHTS = Path(MODELS_DIR, "lion", "lion-mamba.nuscenes.zhe-liu.safetensors")


def test_lion_registered_variant() -> None:
    assert "lion-mamba.nuscenes.zhe-liu" in list_models("lion*", task="detection")


@pytest.mark.skipif(not _SPCONV_AVAILABLE, reason="spconv is not installed")
def test_lion_create_model_hparams() -> None:
    model = create_model("lion-mamba.nuscenes.zhe-liu", task="detection")
    assert isinstance(model, LIONDetection)
    assert model.in_channels == 5
    assert model.num_classes == 10
    assert model.grid_size == (360, 360, 32)
    assert model.backbone_3d.sparse_shape == [32, 360, 360]
    assert model.backbone.num_bev_features == 384
    assert model.num_features == 384
    assert model.head.feature_map_stride == 2


def test_lion_head_decode_packs_detections() -> None:
    """`TransFusionHead.decode` turns raw predictions into PyG-packed detections without needing CUDA."""
    torch.manual_seed(0)
    head = TransFusionHead(384, 10, (360, 360, 32), RANGE, (0.3, 0.3, 0.25)).eval()
    num_proposals = head.num_proposals
    batch_size = 2
    out: TransFusionHeadOutput = {
        "center": torch.rand(batch_size, 2, num_proposals),
        "height": torch.rand(batch_size, 1, num_proposals),
        "dim": torch.rand(batch_size, 3, num_proposals),
        "rot": torch.randn(batch_size, 2, num_proposals),
        "vel": torch.randn(batch_size, 2, num_proposals),
        "iou": torch.rand(batch_size, 1, num_proposals),
        "heatmap": torch.randn(batch_size, 10, num_proposals),
        "query_heatmap_score": torch.rand(batch_size, 10, num_proposals),
        "query_labels": torch.randint(0, 10, (batch_size, num_proposals)),
        "dense_heatmap": torch.randn(batch_size, 10, 180, 180),
    }
    det = head.decode(out)
    assert det["boxes"].shape[1] == 7
    assert det["scores"].shape[0] == det["boxes"].shape[0] == det["labels"].shape[0] == det["batch"].shape[0]
    assert torch.isfinite(det["boxes"]).all()
    assert set(det["batch"].tolist()) <= {0, 1}


def test_lion_head_decode_returns_velocity() -> None:
    """Decoded candidates carry the head's predicted BEV velocity unchanged under `velocity`."""
    torch.manual_seed(0)
    head = TransFusionHead(384, 10, (360, 360, 32), RANGE, (0.3, 0.3, 0.25)).eval()
    num_proposals = head.num_proposals
    batch_size = 2
    vel = torch.empty(batch_size, 2, num_proposals)
    vel[:, 0] = 1.5
    vel[:, 1] = -0.5
    out: TransFusionHeadOutput = {
        "center": torch.rand(batch_size, 2, num_proposals),
        "height": torch.rand(batch_size, 1, num_proposals),
        "dim": torch.rand(batch_size, 3, num_proposals),
        "rot": torch.randn(batch_size, 2, num_proposals),
        "vel": vel,
        "iou": torch.rand(batch_size, 1, num_proposals),
        "heatmap": torch.randn(batch_size, 10, num_proposals),
        "query_heatmap_score": torch.rand(batch_size, 10, num_proposals),
        "query_labels": torch.randint(0, 10, (batch_size, num_proposals)),
        "dense_heatmap": torch.randn(batch_size, 10, 180, 180),
    }
    det = head.decode(out)
    assert det["boxes"].shape[0] > 0
    assert det["velocity"].shape == (det["boxes"].shape[0], 2)
    assert torch.all(det["velocity"][:, 0] == 1.5)
    assert torch.all(det["velocity"][:, 1] == -0.5)


def test_lion_head_predict_supports_non_nuscenes_num_classes() -> None:
    """`predict` must not assume the nuScenes 10-class layout (regression: hardcoded class indices 8/9)."""
    torch.manual_seed(0)
    head = TransFusionHead(16, 3, (40, 40, 32), RANGE, (0.3, 0.3, 0.25), num_proposals=20, query_radius=2).eval()
    x_size = head.grid_size[0] // head.feature_map_stride
    y_size = head.grid_size[1] // head.feature_map_stride
    out = head.predict(torch.randn(2, 16, x_size, y_size))
    assert int(out["query_labels"].max()) < 3


class DummyDecoder(nn.Module):
    def forward(
        self, query: Tensor, key: Tensor, query_pos: Tensor, key_pos: Tensor, key_padding_mask: Optional[Tensor] = None
    ) -> Tensor:
        self.query_pos = query_pos
        self.key_pos = key_pos
        self.key_padding_mask = key_padding_mask
        return query


def test_lion_head_predict_key_indices_on_non_square_grid(monkeypatch: pytest.MonkeyPatch) -> None:
    """The BEV flat index is x-major (x * y_size + y): each query's (0, 0)-offset key must be its own cell."""
    torch.manual_seed(0)
    radius = 2
    head = TransFusionHead(16, 3, (32, 64, 1), RANGE, (0.3, 0.3, 0.25), num_proposals=10, query_radius=radius).eval()
    capture = DummyDecoder()
    monkeypatch.setattr(head, "decoder", capture)
    x_size = head.grid_size[0] // head.feature_map_stride
    y_size = head.grid_size[1] // head.feature_map_stride
    assert x_size != y_size

    with torch.no_grad():
        out = head.predict(torch.randn(2, 16, x_size, y_size))

    center = (2 * radius + 1) * radius + radius
    assert capture.key_padding_mask is not None
    assert not capture.key_padding_mask[:, center].any()
    assert torch.allclose(capture.key_pos[:, center, :], capture.query_pos[:, 0, :])
    assert torch.isfinite(out["heatmap"]).all()


def test_lion_head_local_max_classes_decode() -> None:
    """`decode` is class-count agnostic: a 3-class head with `local_max_classes` set still packs raw detections."""
    torch.manual_seed(0)
    head = TransFusionHead(384, 3, (360, 360, 32), RANGE, (0.3, 0.3, 0.25), local_max_classes=(1, 2)).eval()
    assert head.local_max_classes == (1, 2)
    num_proposals = head.num_proposals
    batch_size = 2
    out: TransFusionHeadOutput = {
        "center": torch.rand(batch_size, 2, num_proposals),
        "height": torch.rand(batch_size, 1, num_proposals),
        "dim": torch.rand(batch_size, 3, num_proposals),
        "rot": torch.randn(batch_size, 2, num_proposals),
        "vel": torch.randn(batch_size, 2, num_proposals),
        "iou": torch.rand(batch_size, 1, num_proposals),
        "heatmap": torch.randn(batch_size, 3, num_proposals),
        "query_heatmap_score": torch.rand(batch_size, 3, num_proposals),
        "query_labels": torch.randint(0, 3, (batch_size, num_proposals)),
        "dense_heatmap": torch.randn(batch_size, 3, 180, 180),
    }
    det = head.decode(out)
    assert det["boxes"].shape[1] == 7
    assert torch.isfinite(det["boxes"]).all()


def test_lion_window_mapping_empty_batch_element() -> None:
    torch.manual_seed(0)
    pos = torch.cat(
        [
            torch.stack(
                [
                    torch.full((n,), b, dtype=torch.long),
                    torch.randint(0, 8, (n,)),
                    torch.randint(0, 32, (n,)),
                    torch.randint(0, 32, (n,)),
                ],
                dim=1,
            )
            for b, n in [(0, 30), (2, 40)]
        ]
    )
    mapping = FlattenedWindowMapping(window_shape=[13, 13, 32], group_size=16, shift=False)

    out = mapping(pos, batch_size=3, sparse_shape=(8, 32, 32))
    assert out["win2flat"].shape == (70,)
    assert out["flat2win"].shape == (80,)  # 30 and 40 voxels each padded up to a multiple of group_size


@pytest.mark.skipif(not _SPCONV_AVAILABLE, reason="spconv is not installed")
def test_lion_patch_merging_invalid_diffusion_scale_raises() -> None:
    import spconv.pytorch as spconv

    merge = PatchMerging3D(8)
    x = spconv.SparseConvTensor(torch.randn(4, 8), torch.zeros(4, 4, dtype=torch.int32), [4, 4, 4], batch_size=1)
    with pytest.raises(ValueError, match="diffusion_scale"):
        merge(x, diffusion_scale=3)


def test_lion_backbone_3d_num_layers_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="num_layers"):
        LION3DBackbone((360, 360, 32), num_layers=2, depths=(2, 2, 2))


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
    model = create_model("lion-mamba.nuscenes.zhe-liu", task="detection").to(DEVICE).eval()
    assert isinstance(model, LIONDetection)
    pos, x, batch = _make_inputs()
    with torch.no_grad():
        out = model(x, pos, batch)
        det = model.decode(out)
        feat = model.forward_features(x, pos, batch)
    assert feat.shape[1] == model.num_features
    assert out["heatmap"].shape == (1, 10, model.head.num_proposals)
    assert out["dense_heatmap"].shape[1] == 10
    assert torch.isfinite(out["heatmap"]).all()
    assert det["boxes"].shape[1] == 7
    assert det["velocity"].shape == (det["boxes"].shape[0], 2)
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
    model = create_model("lion-mamba.nuscenes.zhe-liu", task="detection", pretrained=True).to(DEVICE).eval()
    assert isinstance(model, LIONDetection)
    pos, x, batch = _make_inputs()
    with torch.no_grad():
        det = model.decode(model(x, pos, batch))
    assert torch.isfinite(det["scores"]).all()
    assert (det["labels"] >= 0).all() and (det["labels"] < 10).all()
