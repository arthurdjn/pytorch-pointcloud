from typing import Dict, List

import pytest
import torch
from torch import Tensor

from torch_pointcloud.models import create_model, list_models
from torch_pointcloud.models.oneformer3d import OneFormer3DQueryDecoder, OneFormer3DSegmentation
from torch_pointcloud.utils.imports import _CUDA_AVAILABLE, _SPCONV_AVAILABLE, _TORCH_SCATTER_AVAILABLE

pytestmark = [
    pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA is not available"),
    pytest.mark.skipif(not _SPCONV_AVAILABLE, reason="spconv is not installed"),
    pytest.mark.skipif(not _TORCH_SCATTER_AVAILABLE, reason="torch_scatter is not installed"),
]


def _make_inputs(
    num_voxels_per_scene: int = 64,
    batch_size: int = 2,
    grid_size: int = 16,
    in_channels: int = 6,
    device: str = "cuda",
) -> Dict[str, Tensor]:
    """Build deterministic per-batch unique voxel inputs for OneFormer3D."""
    torch.manual_seed(0)
    pos_list: List[Tensor] = []
    feat_list: List[Tensor] = []
    batch_list: List[Tensor] = []
    for b in range(batch_size):
        flat = torch.randperm(grid_size**3)[:num_voxels_per_scene]
        c = torch.stack(
            [flat // (grid_size**2), (flat // grid_size) % grid_size, flat % grid_size],
            dim=1,
        )
        pos_list.append(c)
        feat_list.append(torch.randn(num_voxels_per_scene, in_channels))
        batch_list.append(torch.full((num_voxels_per_scene,), b, dtype=torch.long))
    pos_grid = torch.cat(pos_list, dim=0).long().to(device)
    feats = torch.cat(feat_list, dim=0).to(device)
    batch = torch.cat(batch_list, dim=0).to(device)

    # Each voxel is its own superpoint, with batch offsets so ids are globally contiguous.
    n_pts = feats.shape[0]
    inverse = torch.arange(n_pts, device=device)
    superpoint = torch.zeros(n_pts, dtype=torch.long, device=device)
    running = 0
    for b in range(batch_size):
        mask = batch == b
        nb = int(mask.sum())
        superpoint[mask] = torch.arange(nb, device=device) + running
        running += nb

    return dict(
        x=feats,
        pos_grid=pos_grid,
        batch=batch,
        superpoint=superpoint,
        inverse=inverse,
    )


@pytest.fixture
def tiny_model() -> OneFormer3DSegmentation:
    return OneFormer3DSegmentation(
        in_channels=6,
        num_classes=20,
        num_instance_classes=18,
        num_channels=16,
        num_levels=3,
        block_reps=1,
        d_model=32,
        num_layers=2,
        num_heads=4,
        hidden_dim=64,
        dropout=0.0,
        iter_pred=True,
        attn_mask=True,
        objectness_flag=False,
        num_semantic_linears=1,
    ).cuda()


def test_oneformer3d_query_decoder_forward() -> None:
    decoder = OneFormer3DQueryDecoder(
        in_channels=16,
        num_instance_classes=10,
        num_semantic_classes=12,
        d_model=32,
        num_layers=2,
        num_heads=4,
        hidden_dim=64,
        iter_pred=False,
        attn_mask=False,
        objectness_flag=True,
    ).cuda()
    feats = [torch.randn(20, 16, device="cuda"), torch.randn(15, 16, device="cuda")]
    out = decoder(feats, feats)
    assert len(out["cls_preds"]) == 2
    assert out["cls_preds"][0].shape == (20, 11)
    assert out["cls_preds"][1].shape == (15, 11)
    assert "sem_preds" in out
    assert out["sem_preds"][0].shape == (20, 13)
    assert out["scores"][0] is not None
    assert out["scores"][0].shape == (20, 1)


def test_oneformer3d_forward_shapes(tiny_model: OneFormer3DSegmentation) -> None:
    data = _make_inputs(num_voxels_per_scene=32, batch_size=2, grid_size=16)
    tiny_model.eval()
    with torch.no_grad():
        out = tiny_model(data["x"], data["pos_grid"], data["batch"], data["superpoint"], data["inverse"])
    assert len(out["cls_preds"]) == 2
    for cls_pred in out["cls_preds"]:
        assert cls_pred.shape[-1] == tiny_model.num_instance_classes + 1
    for sem_pred in out["sem_preds"]:
        assert sem_pred.shape[-1] == tiny_model.num_semantic_classes + 1
    for mask in out["masks"]:
        assert mask.ndim == 2
    assert "aux_outputs" in out
    assert len(out["aux_outputs"]) == tiny_model.head.num_layers


def test_oneformer3d_iter_pred_off() -> None:
    model = OneFormer3DSegmentation(
        in_channels=6,
        num_classes=20,
        num_instance_classes=18,
        num_channels=16,
        num_levels=3,
        block_reps=1,
        d_model=32,
        num_layers=2,
        num_heads=4,
        hidden_dim=64,
        iter_pred=False,
        attn_mask=False,
    ).cuda()
    data = _make_inputs(num_voxels_per_scene=32, batch_size=2, grid_size=16)
    model.eval()
    with torch.no_grad():
        out = model(data["x"], data["pos_grid"], data["batch"], data["superpoint"], data["inverse"])
    assert "aux_outputs" not in out
    assert "sem_preds" in out
    assert len(out["sem_preds"]) == 2


def test_oneformer3d_reset_classifier(tiny_model: OneFormer3DSegmentation) -> None:
    tiny_model.reset_classifier(num_classes=42)
    tiny_model.cuda().eval()
    data = _make_inputs(num_voxels_per_scene=32, batch_size=2, grid_size=16)
    with torch.no_grad():
        out = tiny_model(data["x"], data["pos_grid"], data["batch"], data["superpoint"], data["inverse"])
    assert out["cls_preds"][0].shape[-1] == 43


def test_oneformer3d_predict_instance_and_semantic(tiny_model: OneFormer3DSegmentation) -> None:
    data = _make_inputs(num_voxels_per_scene=32, batch_size=2, grid_size=16)
    tiny_model.eval()
    with torch.no_grad():
        out = tiny_model(data["x"], data["pos_grid"], data["batch"], data["superpoint"], data["inverse"])
    sp_per_point = data["superpoint"][data["batch"][data["inverse"]] == 0]
    sem = tiny_model.predict_semantic(out, sp_per_point - int(sp_per_point.min()))
    assert sem.shape == sp_per_point.shape
    masks, labels, scores = tiny_model.predict_instance(
        out,
        sp_per_point - int(sp_per_point.min()),
        topk=64,
        npoint_threshold=0,
    )
    assert masks.dtype == torch.bool
    assert labels.dtype == torch.long
    assert scores.ndim == 1
    assert masks.shape[0] == labels.shape[0] == scores.shape[0]


def test_oneformer3d_registered_variants() -> None:
    names = list_models("oneformer3d*", task="segmentation")
    assert "oneformer3d-base.scannet20" in names
    assert "oneformer3d-base.scannet200" in names
    assert "oneformer3d-base.s3dis-area5" in names


def test_oneformer3d_create_model_no_pretrained() -> None:
    model = create_model("oneformer3d-base.scannet20", task="segmentation")
    assert isinstance(model, OneFormer3DSegmentation)
    assert model.num_instance_classes == 18
    assert model.num_semantic_classes == 20
