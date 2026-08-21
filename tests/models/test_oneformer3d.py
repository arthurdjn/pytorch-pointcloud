from typing import Dict, List

import pytest
import torch
from torch import Tensor

from torch_pointcloud.models import create_model, list_models
from torch_pointcloud.models.oneformer3d import OneFormer3DOutput, OneFormer3DQueryDecoder, OneFormer3DSegmentation
from torch_pointcloud.utils.imports import _CUDA_AVAILABLE, _SPCONV_AVAILABLE, _TORCH_SCATTER_AVAILABLE
from torch_pointcloud.utils.metrics import instance_average_precision, instance_matches

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
        channels=(16, 32, 64),
        layers=1,
        embed_dim=32,
        num_layers=2,
        num_heads=4,
        mlp_dim=64,
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
        embed_dim=32,
        num_layers=2,
        num_heads=4,
        mlp_dim=64,
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


def test_oneformer3d_forward_head_pre_logits(tiny_model: OneFormer3DSegmentation) -> None:
    data = _make_inputs(num_voxels_per_scene=32, batch_size=2, grid_size=16)
    tiny_model.eval()
    with torch.no_grad():
        feats = tiny_model.forward_features(data["x"], data["pos_grid"], data["batch"])
        sources = tiny_model.forward_decoder(feats, data["batch"], data["superpoint"], data["inverse"])
        pre = tiny_model.forward_head(sources, pre_logits=True)
    assert pre is sources


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
        channels=(16, 32, 64),
        layers=1,
        embed_dim=32,
        num_layers=2,
        num_heads=4,
        mlp_dim=64,
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


def test_oneformer3d_s3dis_mode_no_pooling() -> None:
    # S3DIS-style config: no superpoint pooling, learned queries, no out_sem head.
    model = OneFormer3DSegmentation(
        in_channels=6,
        num_classes=13,
        channels=(16, 32, 64),
        layers=1,
        embed_dim=32,
        num_layers=2,
        num_instance_queries=40,
        num_semantic_queries=13,
        num_heads=4,
        mlp_dim=64,
        objectness_flag=True,
        semantic_head=False,
        superpoint_pooling=False,
    ).cuda()
    data = _make_inputs(num_voxels_per_scene=32, batch_size=2, grid_size=16)
    model.eval()
    with torch.no_grad():
        out = model(data["x"], data["pos_grid"], data["batch"])
    assert "sem_preds" not in out
    assert len(out["cls_preds"]) == 2
    for cls_pred in out["cls_preds"]:
        assert cls_pred.shape == (40 + 13, model.num_instance_classes + 1)
    for score in out["scores"]:
        assert score is not None and score.shape == (40 + 13, 1)


def test_oneformer3d_reset_classifier(tiny_model: OneFormer3DSegmentation) -> None:
    tiny_model.reset_classifier(num_classes=42)
    tiny_model.cuda().eval()
    data = _make_inputs(num_voxels_per_scene=32, batch_size=2, grid_size=16)
    with torch.no_grad():
        out = tiny_model(data["x"], data["pos_grid"], data["batch"], data["superpoint"], data["inverse"])
    assert out["cls_preds"][0].shape[-1] == 43
    assert out["sem_preds"][0].shape[-1] == 43


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
    assert "oneformer3d-base.scannet20.danila-rukhovich" in names
    assert "oneformer3d-base.scannet200.danila-rukhovich" in names
    assert "oneformer3d-base.s3dis-area5.danila-rukhovich" in names


def test_oneformer3d_scannet200_registered_without_weights() -> None:
    pretrained = list_models("oneformer3d*", task="segmentation", pretrained=True)
    assert "oneformer3d-base.scannet200.danila-rukhovich" not in pretrained
    assert "oneformer3d-base.scannet20.danila-rukhovich" in pretrained


def test_oneformer3d_create_model_no_pretrained() -> None:
    model = create_model("oneformer3d-base.scannet20.danila-rukhovich", task="segmentation")
    assert isinstance(model, OneFormer3DSegmentation)
    assert model.num_instance_classes == 18
    assert model.num_semantic_classes == 20


def _make_decode_model(**overrides: object) -> OneFormer3DSegmentation:
    hparams: Dict[str, object] = dict(
        in_channels=6,
        num_classes=3,
        num_instance_classes=2,
        channels=(16, 32, 64),
        layers=1,
        embed_dim=16,
        num_layers=1,
        num_heads=2,
        mlp_dim=32,
    )
    hparams.update(overrides)
    return OneFormer3DSegmentation(**hparams)  # type: ignore[arg-type]


def test_oneformer3d_predict_instance_synthetic_decode() -> None:
    """Hand-crafted decoder output: two confident queries decode to two known instance masks.

    Query 0 is class 0 on superpoints 0-1, query 1 is class 1 on superpoints 2-3. The expected score
    is the class softmax times the mean sigmoid of the positive mask entries (objectness
    normalization). The decoded masks then score a perfect instance mAP against matching GT.
    """
    model = _make_decode_model()
    output: OneFormer3DOutput = {
        "cls_preds": [torch.tensor([[10.0, 0.0, -10.0], [0.0, 10.0, -10.0]])],
        "masks": [torch.tensor([[5.0, 5.0, -5.0, -5.0], [-5.0, -5.0, 5.0, 5.0]])],
        "scores": [None],
    }
    superpoint_per_point = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    masks, labels, scores = model.predict_instance(
        output, superpoint_per_point, topk=4, score_threshold=0.05, npoint_threshold=0
    )
    assert masks.shape == (2, 8)
    assert sorted(labels.tolist()) == [0, 1]
    expected_mask = torch.tensor([True, True, True, True, False, False, False, False])
    assert torch.equal(masks[labels == 0][0], expected_mask)
    assert torch.equal(masks[labels == 1][0], ~expected_mask)
    expected_score = float(torch.softmax(torch.tensor([10.0, 0.0, -10.0]), 0)[0] * torch.sigmoid(torch.tensor(5.0)))
    assert scores[0].item() == pytest.approx(expected_score, abs=1e-4)
    assert scores[1].item() == pytest.approx(expected_score, abs=1e-4)

    gt_instance = torch.tensor([0] * 4 + [1] * 4)
    gt_label = torch.tensor([0] * 4 + [1] * 4)
    match = instance_matches(masks, labels, scores, gt_instance, gt_label)
    out = instance_average_precision([match], num_classes=2, min_points=1)
    assert out["mAP"] == pytest.approx(1.0)
    assert out["mAP@0.25"] == pytest.approx(1.0)


def test_oneformer3d_predict_instance_semantic_query_slicing() -> None:
    """S3DIS-style decode: semantic queries are excluded, objectness scales scores, sigmoid-threshold
    objectness normalization averages every mask entry above `obj_normalization_threshold`.

    The two semantic queries carry full masks and huge class-0 logits: without the slicing they would
    dominate the top-k. Query 0's mask has a logit at sigmoid 0.12, included in the normalization mean
    at threshold 0.01 (it would be excluded at the ScanNet default 0.5) but cut from the final mask by
    `sp_score_threshold=0.15`.
    """
    model = _make_decode_model(
        num_classes=2,
        num_instance_classes=2,
        num_instance_queries=3,
        num_semantic_queries=2,
        semantic_head=False,
        objectness_flag=True,
        superpoint_pooling=False,
    )
    output: OneFormer3DOutput = {
        "cls_preds": [
            torch.tensor(
                [
                    [10.0, 0.0, -10.0],
                    [0.0, 10.0, -10.0],
                    [-10.0, -10.0, 10.0],
                    [20.0, 0.0, -20.0],
                    [20.0, 0.0, -20.0],
                ]
            )
        ],
        "masks": [
            torch.tensor(
                [
                    [5.0, -2.0, -5.0, -5.0, -5.0, -5.0],
                    [-5.0, -5.0, 5.0, 5.0, -5.0, -5.0],
                    [5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
                    [5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
                    [5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
                ]
            )
        ],
        "scores": [torch.tensor([[0.5], [1.0], [1.0], [1.0], [1.0]])],
    }
    masks, labels, scores = model.predict_instance(
        output,
        torch.arange(6),
        topk=6,
        score_threshold=0.05,
        sp_score_threshold=0.15,
        npoint_threshold=0,
        obj_normalization_threshold=0.01,
    )
    assert masks.shape == (2, 6)
    assert sorted(labels.tolist()) == [0, 1]
    assert not masks.all(dim=1).any(), "a full mask means a semantic query leaked into the instance decode"
    assert torch.equal(masks[labels == 0][0], torch.tensor([True, False, False, False, False, False]))
    softmax0 = torch.softmax(torch.tensor([10.0, 0.0, -10.0]), 0)[0]
    mask_mean = (torch.sigmoid(torch.tensor(5.0)) + torch.sigmoid(torch.tensor(-2.0))) / 2
    assert scores[labels == 0].item() == pytest.approx(float(softmax0 * 0.5 * mask_mean), abs=1e-4)


def test_oneformer3d_predict_instance_filters() -> None:
    """The score and point-count thresholds drop instances after mask expansion."""
    model = _make_decode_model()
    output: OneFormer3DOutput = {
        "cls_preds": [torch.tensor([[10.0, 0.0, -10.0], [0.0, 10.0, -10.0]])],
        "masks": [torch.tensor([[5.0, 5.0, -5.0, -5.0], [-5.0, -5.0, 5.0, 5.0]])],
        "scores": [None],
    }
    superpoint_per_point = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    masks, _, _ = model.predict_instance(output, superpoint_per_point, topk=4, score_threshold=0.999)
    assert masks.shape[0] == 0
    masks, _, _ = model.predict_instance(output, superpoint_per_point, topk=4, npoint_threshold=4)
    assert masks.shape[0] == 0
    masks, _, _ = model.predict_instance(output, superpoint_per_point, topk=4, score_threshold=0.05, npoint_threshold=3)
    assert masks.shape[0] == 2


def test_oneformer3d_predict_instance_topk_above_pair_count() -> None:
    """`topk` larger than `num_queries x num_instance_classes` (small scenes) must clamp, not raise."""
    model = _make_decode_model()
    output: OneFormer3DOutput = {
        "cls_preds": [torch.tensor([[10.0, 0.0, -10.0], [0.0, 10.0, -10.0]])],
        "masks": [torch.tensor([[5.0, 5.0, -5.0, -5.0], [-5.0, -5.0, 5.0, 5.0]])],
        "scores": [None],
    }
    superpoint_per_point = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    masks, labels, scores = model.predict_instance(
        output, superpoint_per_point, topk=600, score_threshold=0.05, npoint_threshold=0
    )
    assert masks.shape == (2, 8)
    assert sorted(labels.tolist()) == [0, 1]
