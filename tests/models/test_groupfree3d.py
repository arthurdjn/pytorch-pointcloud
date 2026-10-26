from typing import Any, Dict

import pytest
import torch
from torch import Tensor

from torch_pointcloud.layers.transformer import TransformerDecoderLayer
from torch_pointcloud.models import create_model, list_models
from torch_pointcloud.models.groupfree3d import GroupFree3DDetection, GroupFree3DPredictionHead
from torch_pointcloud.ops.box3d import class_to_angle
from torch_pointcloud.utils.imports import _PYG_LIB_AVAILABLE

requires_pyg_lib = pytest.mark.skipif(not _PYG_LIB_AVAILABLE, reason="pyg-lib is not installed")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _create_groupfree3d(**overrides: Any) -> GroupFree3DDetection:
    kwargs: Dict[str, Any] = dict(
        in_channels=0,
        num_classes=18,
        num_heading_bins=1,
        num_size_clusters=18,
        mean_sizes=[[1.0, 1.0, 1.0]] * 18,
        size_class_agnostic=False,
        num_proposals=16,
        sa_channels=[[16, 16, 32], [32, 32, 64], [64, 64, 64], [64, 64, 64]],
        sa_num_points=[256, 128, 64, 32],
        sa_radii=[0.2, 0.4, 0.8, 1.2],
        sa_num_neighbors=[16, 16, 16, 16],
        fp_channels=[[64, 64], [64, 32]],
        num_decoder_layers=2,
        num_heads=4,
        mlp_dim=64,
        dropout=0.1,
    )
    kwargs.update(overrides)
    return GroupFree3DDetection(**kwargs)


def _make_inputs(n_per_scene: int = 3000, batch_size: int = 2) -> Dict[str, Tensor]:
    """Two scenes of `n_per_scene` points each (>= the 2048 the first SA layer samples)."""
    torch.manual_seed(0)
    pos = torch.rand(n_per_scene * batch_size, 3) * 4.0
    batch = torch.arange(batch_size).repeat_interleave(n_per_scene)
    return {"pos": pos.to(DEVICE), "batch": batch.to(DEVICE)}


def _assert_stage_shapes(out: Dict[str, Tensor], stages: int, batch_size: int, q: int, nh: int, nc: int) -> None:
    assert out["objectness_scores"].shape == (stages, batch_size, q, 1)
    assert out["center"].shape == (stages, batch_size, q, 3)
    assert out["heading_scores"].shape == (stages, batch_size, q, nh)
    assert out["heading_residuals"].shape == (stages, batch_size, q, nh)
    assert out["size"].shape == (stages, batch_size, q, 3)
    assert out["sem_cls_scores"].shape == (stages, batch_size, q, nc)
    assert out["query_pos"].shape == (batch_size, q, 3)
    assert out["query_indices"].shape == (batch_size, q)


@requires_pyg_lib
def test_groupfree3d_scannet_forward_shapes() -> None:
    model = create_model("groupfree3d-l6-o256.scannet.ze-liu", task="detection").to(DEVICE).eval()
    data = _make_inputs()
    with torch.no_grad():
        out = model(None, data["pos"], data["batch"])
        x_seed, _, _, _ = model.forward_features(None, data["pos"], data["batch"])
    assert x_seed.shape[1] == model.num_features == 288
    _assert_stage_shapes(out, stages=7, batch_size=2, q=256, nh=1, nc=18)
    assert out["size_scores"].shape == (7, 2, 256, 18)
    assert out["size_residuals"].shape == (7, 2, 256, 18, 3)
    # Seeds are the 1024 SA2 points per scene; every candidate is one of its scene's seeds.
    assert out["pos_seed"].shape == (2 * 1024, 3)
    assert out["seed_objectness_logits"].shape == (2 * 1024,)
    assert torch.equal(out["pos_seed"][out["query_indices"]], out["query_pos"])
    assert torch.isfinite(out["center"]).all()


@requires_pyg_lib
def test_groupfree3d_sunrgbd_forward_shapes() -> None:
    # SUN RGB-D uses 12 heading bins, 10 classes and a class-agnostic size head.
    model = create_model("groupfree3d-l6-o256.sunrgbd.ze-liu", task="detection").to(DEVICE).eval()
    data = _make_inputs()
    with torch.no_grad():
        out = model(None, data["pos"], data["batch"])
    _assert_stage_shapes(out, stages=7, batch_size=2, q=256, nh=12, nc=10)
    assert "size_scores" not in out
    assert model.mean_sizes is None


@requires_pyg_lib
def test_groupfree3d_candidates_are_the_top_scoring_seeds() -> None:
    model = _create_groupfree3d().to(DEVICE).eval()
    data = _make_inputs()
    with torch.no_grad():
        out = model(None, data["pos"], data["batch"])
    scores = out["seed_objectness_logits"].view(2, -1)
    offsets = scores.size(1) * torch.arange(2, device=DEVICE).unsqueeze(1)
    expected = scores.topk(model.num_proposals, dim=1).indices + offsets
    assert torch.equal(out["query_indices"].sort(dim=1).values, expected.sort(dim=1).values)


@requires_pyg_lib
def test_groupfree3d_eval_is_deterministic() -> None:
    model = _create_groupfree3d().to(DEVICE).eval()
    data = _make_inputs()
    with torch.no_grad():
        a = model(None, data["pos"], data["batch"])
        b = model(None, data["pos"], data["batch"])
    for key in ("center", "objectness_scores", "sem_cls_scores", "size"):
        if DEVICE == "cpu":
            assert torch.equal(a[key], b[key]), f"{key} not bit-identical on CPU"
        else:
            assert torch.allclose(a[key], b[key], atol=1e-5), f"{key} drifted beyond float-atomics on CUDA"


@requires_pyg_lib
def test_groupfree3d_train_mode_backward() -> None:
    model = _create_groupfree3d().to(DEVICE).train()
    data = _make_inputs()
    out = model(None, data["pos"], data["batch"])
    loss = out["center"].sum() + out["objectness_scores"].sum() + out["seed_objectness_logits"].sum()
    loss.backward()
    assert model.proposal_head.center_head.weight.grad is not None
    layer = model.decoder[0]
    assert isinstance(layer, TransformerDecoderLayer)
    assert layer.linear1.weight.grad is not None


@requires_pyg_lib
def test_groupfree3d_reset_classifier() -> None:
    model = _create_groupfree3d()
    model.reset_classifier(num_classes=5)
    assert model.num_classes == 5
    for head in [model.proposal_head, *model.prediction_heads]:
        assert isinstance(head, GroupFree3DPredictionHead)
        assert head.sem_cls_head.out_features == 5
    model = model.to(DEVICE).eval()
    data = _make_inputs()
    with torch.no_grad():
        out = model(None, data["pos"], data["batch"])
    assert out["sem_cls_scores"].shape[-1] == 5


def test_groupfree3d_class_aware_size_requires_mean_sizes() -> None:
    with pytest.raises(ValueError, match="mean_sizes"):
        _create_groupfree3d(mean_sizes=None)
    with pytest.raises(ValueError, match="mean_sizes"):
        _create_groupfree3d(num_size_clusters=3, mean_sizes=[[1.0, 1.0, 1.0]])


def test_groupfree3d_mean_sizes_not_persisted() -> None:
    model = create_model("groupfree3d-l6-o256.scannet.ze-liu", task="detection")
    assert isinstance(model, GroupFree3DDetection)
    assert "mean_sizes" not in model.state_dict()
    assert model.mean_sizes is not None
    assert model.mean_sizes.shape == (18, 3)


def test_groupfree3d_prediction_head_class_aware_size() -> None:
    torch.manual_seed(0)
    head = GroupFree3DPredictionHead(
        8, num_classes=4, num_heading_bins=3, num_size_clusters=2, size_class_agnostic=False
    )
    head = head.eval()
    mean_sizes = torch.tensor([[1.0, 2.0, 3.0], [2.0, 2.0, 2.0]])
    x, pos = torch.randn(2, 5, 8), torch.randn(2, 5, 3)
    with torch.no_grad():
        out = head(x, pos, mean_sizes)
    assert out["center"].shape == (2, 5, 3)
    assert torch.allclose(out["center"] - pos, head.center_head(head.mlp(x.view(-1, 8))).view(2, 5, 3))
    size_class = out["size_scores"].argmax(-1)
    expected = (
        mean_sizes[size_class]
        + out["size_residuals"].gather(2, size_class[..., None, None].expand(-1, -1, 1, 3))[:, :, 0]
    )
    assert torch.allclose(out["size"], expected)
    assert torch.allclose(out["size_residuals"], out["size_residuals_normalized"] * mean_sizes)


@requires_pyg_lib
def test_groupfree3d_decode_uses_last_stage_and_negates_heading() -> None:
    torch.manual_seed(0)
    model = _create_groupfree3d(num_heading_bins=12).to(DEVICE).eval()
    data = _make_inputs()
    with torch.no_grad():
        out = model(None, data["pos"], data["batch"])
    det = model.decode(out)

    b, q = 2, model.num_proposals
    heading_class = out["heading_scores"][-1].argmax(dim=-1)
    heading_residual = out["heading_residuals"][-1].gather(2, heading_class.unsqueeze(-1)).squeeze(-1)
    native = class_to_angle(heading_class, heading_residual, 12)
    assert det["boxes"].shape == (b * q, 7)
    assert torch.allclose(det["boxes"][:, 6], -native.reshape(-1))
    assert torch.equal(det["boxes"][:, :3], out["center"][-1].reshape(-1, 3))
    assert torch.equal(det["boxes"][:, 3:6], out["size"][-1].reshape(-1, 3))
    assert torch.allclose(det["scores"], out["objectness_scores"][-1][..., 0].sigmoid().reshape(-1))
    assert torch.equal(det["labels"], out["sem_cls_scores"][-1].argmax(-1).reshape(-1))
    assert torch.equal(det["batch"], torch.arange(b, device=DEVICE).repeat_interleave(q))


@requires_pyg_lib
def test_groupfree3d_decode_ensembles_stages_scene_by_scene() -> None:
    model = _create_groupfree3d().to(DEVICE).eval()
    data = _make_inputs()
    with torch.no_grad():
        out = model(None, data["pos"], data["batch"])
    det = model.decode(out, stages=[-1, 0])

    b, q = 2, model.num_proposals
    assert det["boxes"].shape == (b * 2 * q, 7)
    assert torch.equal(det["batch"], torch.arange(b, device=DEVICE).repeat_interleave(2 * q))
    centers = det["boxes"][:, :3].view(b, 2, q, 3)
    assert torch.equal(centers[:, 0], out["center"][-1])
    assert torch.equal(centers[:, 1], out["center"][0])


@requires_pyg_lib
def test_groupfree3d_decode_rejects_unknown_stages() -> None:
    model = _create_groupfree3d().to(DEVICE).eval()
    data = _make_inputs()
    with torch.no_grad():
        out = model(None, data["pos"], data["batch"])
    with pytest.raises(IndexError, match="stages"):
        model.decode(out, stages=[model.num_decoder_layers + 1])
    with pytest.raises(IndexError, match="stages"):
        model.decode(out, stages=[])


def test_groupfree3d_registered_variants() -> None:
    names = list_models("groupfree3d*", task="detection")
    assert names == [
        "groupfree3d-l12-o256.scannet.mmdet3d",
        "groupfree3d-l12-o256.scannet.ze-liu",
        "groupfree3d-l6-o256.scannet.mmdet3d",
        "groupfree3d-l6-o256.scannet.ze-liu",
        "groupfree3d-l6-o256.sunrgbd.ze-liu",
        "groupfree3d-w2x-l12-o256.scannet.mmdet3d",
        "groupfree3d-w2x-l12-o256.scannet.ze-liu",
        "groupfree3d-w2x-l12-o512.scannet.mmdet3d",
        "groupfree3d-w2x-l12-o512.scannet.ze-liu",
    ]


def test_groupfree3d_w2x_doubles_the_backbone() -> None:
    model = create_model("groupfree3d-w2x-l12-o512.scannet.ze-liu", task="detection")
    assert isinstance(model, GroupFree3DDetection)
    assert model.num_decoder_layers == 12
    assert model.num_proposals == 512
    assert model.num_features == 288
    state_dict = model.state_dict()
    assert state_dict["backbone.sa_modules.0.convs.0.local_nn.lins.2.weight"].shape == (256, 128)
    assert state_dict["backbone.fp_modules.1.mlp.lins.1.weight"].shape == (288, 512)
