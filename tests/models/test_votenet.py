from typing import Any, Dict

import pytest
import torch
from torch import Tensor

from torch_pointcloud.layers.pointnet2_blocks import SAModule
from torch_pointcloud.models import create_model, list_models
from torch_pointcloud.models.votenet import VoteNetDetection, VotingModule
from torch_pointcloud.utils.imports import _TORCH_CLUSTER_AVAILABLE, _TORCH_SCATTER_AVAILABLE

pytestmark = [
    pytest.mark.skipif(not _TORCH_CLUSTER_AVAILABLE, reason="torch-cluster is not installed"),
    pytest.mark.skipif(not _TORCH_SCATTER_AVAILABLE, reason="torch-scatter is not installed"),
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _votenet_kwargs(**overrides: Any) -> Dict[str, Any]:
    """A complete small-architecture VoteNet config (no defaults live on the class itself)."""
    kwargs: Dict[str, Any] = dict(
        in_channels=1,
        num_classes=18,
        num_heading_bin=1,
        num_size_cluster=18,
        mean_size_arr=[[1.0, 1.0, 1.0]] * 18,
        num_proposal=16,
        vote_factor=1,
        sampling="vote_fps",
        sa_channels=[[16, 16, 32], [32, 32, 64], [64, 64, 64], [64, 64, 64]],
        sa_npoints=[256, 128, 64, 32],
        sa_radii=[0.2, 0.4, 0.8, 1.2],
        sa_num_neighbors=[16, 16, 16, 16],
        fp_channels=[[64, 64], [64, 64]],
        vote_aggr_channels=[64, 64, 64],
        vote_aggr_radius=0.3,
        vote_aggr_num_neighbors=16,
    )
    kwargs.update(overrides)
    return kwargs


def _make_inputs(n_per_scene: int = 3000, batch_size: int = 2, in_channels: int = 1) -> Dict[str, Tensor]:
    """Two scenes of `n_per_scene` points each (>= the 2048 the first SA layer samples)."""
    torch.manual_seed(0)
    n = n_per_scene * batch_size
    pos = torch.rand(n, 3) * 4.0
    x = torch.rand(n, in_channels)
    batch = torch.arange(batch_size).repeat_interleave(n_per_scene)
    return {"x": x.to(DEVICE), "pos": pos.to(DEVICE), "batch": batch.to(DEVICE)}


def _assert_proposal_shapes(
    out: Dict[str, Tensor], batch_size: int, num_proposal: int, nh: int, ns: int, nc: int
) -> None:
    assert out["objectness_scores"].shape == (batch_size, num_proposal, 2)
    assert out["center"].shape == (batch_size, num_proposal, 3)
    assert out["heading_scores"].shape == (batch_size, num_proposal, nh)
    assert out["heading_residuals"].shape == (batch_size, num_proposal, nh)
    assert out["size_scores"].shape == (batch_size, num_proposal, ns)
    assert out["size_residuals"].shape == (batch_size, num_proposal, ns, 3)
    assert out["sem_cls_scores"].shape == (batch_size, num_proposal, nc)
    assert out["aggregated_vote_pos"].shape == (batch_size, num_proposal, 3)


def test_votenet_scannet_forward_shapes() -> None:
    model = create_model("votenet-fair-base.scannet", task="detection").to(DEVICE).eval()
    data = _make_inputs(in_channels=model.in_channels)
    with torch.no_grad():
        out = model(data["x"], data["pos"], data["batch"])
    _assert_proposal_shapes(out, batch_size=2, num_proposal=256, nh=1, ns=18, nc=18)
    # Seeds are the 1024 SA2 points per scene; votes are 1:1 with seeds.
    assert out["seed_pos"].shape == (2 * 1024, 3)
    assert out["vote_pos"].shape == (2 * 1024, 3)
    assert torch.isfinite(out["center"]).all()


def test_votenet_sunrgbd_forward_shapes() -> None:
    # SUN RGB-D uses 12 heading bins, 10 classes and seed_fps sampling.
    model = create_model("votenet-fair-base.sunrgbd", task="detection").to(DEVICE).eval()
    data = _make_inputs(in_channels=model.in_channels)
    with torch.no_grad():
        out = model(data["x"], data["pos"], data["batch"])
    _assert_proposal_shapes(out, batch_size=2, num_proposal=256, nh=12, ns=10, nc=10)


def test_votenet_eval_is_deterministic() -> None:
    model = create_model("votenet-fair-base.scannet", task="detection").to(DEVICE).eval()
    data = _make_inputs()
    with torch.no_grad():
        a = model(data["x"], data["pos"], data["batch"])
        b = model(data["x"], data["pos"], data["batch"])
    for key in ("center", "objectness_scores", "sem_cls_scores", "size_residuals"):
        if DEVICE == "cpu":
            assert torch.equal(a[key], b[key]), f"{key} not bit-identical on CPU"
        else:
            # On CUDA, scatter-add atomics make the FP interpolation reproducible only up to
            # float ordering (~1e-7); the same caveat applies to every packed model here.
            assert torch.allclose(a[key], b[key], atol=1e-5), f"{key} drifted beyond float-atomics on CUDA"


def test_votenet_reset_classifier() -> None:
    model = create_model("votenet-fair-base.scannet", task="detection")
    assert isinstance(model, VoteNetDetection)
    model.reset_classifier(num_classes=5)
    assert model.num_classes == 5
    assert model.proposal.mlp.lins[-1].out_features == 2 + 3 + 1 * 2 + 18 * 4 + 5
    model = model.to(DEVICE).eval()
    data = _make_inputs()
    with torch.no_grad():
        out = model(data["x"], data["pos"], data["batch"])
    assert out["sem_cls_scores"].shape[-1] == 5


def test_votenet_seed_fps_requires_unit_vote_factor() -> None:
    with pytest.raises(ValueError, match="vote_factor"):
        VoteNetDetection(**_votenet_kwargs(sampling="seed_fps", vote_factor=2))


def test_votenet_bad_mean_size_arr_shape() -> None:
    with pytest.raises(ValueError, match="mean_size_arr"):
        VoteNetDetection(**_votenet_kwargs(num_size_cluster=3, mean_size_arr=[[1.0, 1.0, 1.0]]))


def test_votenet_mean_size_arr_not_persisted() -> None:
    # The reference rebuilds mean_size_arr on the fly, so it must stay out of the checkpoint.
    model = create_model("votenet-fair-base.scannet", task="detection")
    assert isinstance(model, VoteNetDetection)
    assert "mean_size_arr" not in model.state_dict()
    # ...but it still moves with the module and drives size decoding.
    assert model.mean_size_arr.shape == (18, 3)


def test_sa_module_num_points_and_precomputed_idx() -> None:
    sa = SAModule(in_channels=1, channels=[16, 16], num_points=64, radii=0.4, num_neighbors=16, pos_first=True).eval()
    pos = torch.rand(500, 3)
    x = torch.rand(500, 1)
    batch = torch.zeros(500, dtype=torch.long)
    with torch.no_grad():
        new_x, new_pos, new_batch = sa(x, pos, batch)
    assert new_x.shape == (64, 16)
    assert new_pos.shape == (64, 3)
    assert new_batch.shape == (64,)
    # A precomputed sampling index is honoured verbatim.
    idx = torch.arange(64)
    with torch.no_grad():
        nx, npos, _ = sa(x, pos, batch, idx)
    assert torch.equal(npos, pos[idx])
    assert nx.shape == (64, 16)


def test_sa_module_requires_exactly_one_sampling_spec() -> None:
    with pytest.raises(ValueError, match="ratio"):
        SAModule(in_channels=1, channels=[16], radii=0.4, num_neighbors=16)
    with pytest.raises(ValueError, match="ratio"):
        SAModule(in_channels=1, channels=[16], ratio=0.5, num_points=64, radii=0.4, num_neighbors=16)


def test_votenet_voting_module_residual() -> None:
    vgen = VotingModule(vote_factor=1, seed_feature_dim=8).eval()
    seed_pos = torch.rand(10, 3)
    seed_x = torch.rand(10, 8)
    seed_batch = torch.zeros(10, dtype=torch.long)
    with torch.no_grad():
        vote_pos, vote_x, vote_batch = vgen(seed_pos, seed_x, seed_batch)
    assert vote_pos.shape == (10, 3)
    assert vote_x.shape == (10, 8)
    assert torch.equal(vote_batch, seed_batch)


def test_votenet_registered_variants() -> None:
    names = list_models("votenet*", task="detection")
    assert "votenet-fair-base.scannet" in names
    assert "votenet-fair-base.sunrgbd" in names


def test_votenet_create_model_no_pretrained() -> None:
    model = create_model("votenet-fair-base.sunrgbd", task="detection")
    assert isinstance(model, VoteNetDetection)
    assert model.num_classes == 10
    assert model.num_heading_bin == 12
    assert model.num_size_cluster == 10
    assert model.sampling == "seed_fps"
