from typing import Any, Dict

import pytest
import torch
from torch import Tensor

from torch_pointcloud.layers.pointnet2_blocks import SAModule
from torch_pointcloud.losses import VoteNetLoss
from torch_pointcloud.models import create_model, list_models
from torch_pointcloud.models.votenet import VoteNetDetection, VoteNetOutput, VotingModule
from torch_pointcloud.transforms.functional import class_to_angle, class_to_size
from torch_pointcloud.utils.imports import _TORCH_CLUSTER_AVAILABLE, _TORCH_SCATTER_AVAILABLE

pytestmark = [
    pytest.mark.skipif(not _TORCH_CLUSTER_AVAILABLE, reason="torch-cluster is not installed"),
    pytest.mark.skipif(not _TORCH_SCATTER_AVAILABLE, reason="torch-scatter is not installed"),
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _create_votenet(**overrides: Any) -> VoteNetDetection:
    kwargs: Dict[str, Any] = dict(
        in_channels=1,
        num_classes=18,
        num_heading_bin=1,
        num_size_cluster=18,
        mean_sizes=[[1.0, 1.0, 1.0]] * 18,
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
    return VoteNetDetection(**kwargs)


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
    assert out["pos_vote_aggr"].shape == (batch_size, num_proposal, 3)


def test_votenet_scannet_forward_shapes() -> None:
    model = create_model("votenet.scannet.fair", task="detection").to(DEVICE).eval()
    data = _make_inputs(in_channels=model.in_channels)
    with torch.no_grad():
        out = model(data["x"], data["pos"], data["batch"])
    _assert_proposal_shapes(out, batch_size=2, num_proposal=256, nh=1, ns=18, nc=18)
    # Seeds are the 1024 SA2 points per scene; votes are 1:1 with seeds.
    assert out["pos_seed"].shape == (2 * 1024, 3)
    assert out["pos_vote"].shape == (2 * 1024, 3)
    assert torch.isfinite(out["center"]).all()


def test_votenet_sunrgbd_forward_shapes() -> None:
    # SUN RGB-D uses 12 heading bins, 10 classes and seed_fps sampling.
    model = create_model("votenet.sunrgbd.fair", task="detection").to(DEVICE).eval()
    data = _make_inputs(in_channels=model.in_channels)
    with torch.no_grad():
        out = model(data["x"], data["pos"], data["batch"])
    _assert_proposal_shapes(out, batch_size=2, num_proposal=256, nh=12, ns=10, nc=10)


def test_votenet_eval_is_deterministic() -> None:
    model = create_model("votenet.scannet.fair", task="detection").to(DEVICE).eval()
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
    model = create_model("votenet.scannet.fair", task="detection")
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
        _create_votenet(sampling="seed_fps", vote_factor=2)


def test_votenet_bad_mean_sizes_shape() -> None:
    with pytest.raises(ValueError, match="mean_sizes"):
        _create_votenet(num_size_cluster=3, mean_sizes=[[1.0, 1.0, 1.0]])


def test_votenet_mean_sizes_not_persisted() -> None:
    # The reference rebuilds mean_sizes on the fly, so it must stay out of the checkpoint.
    model = create_model("votenet.scannet.fair", task="detection")
    assert isinstance(model, VoteNetDetection)
    assert "mean_sizes" not in model.state_dict()
    # ...but it still moves with the module and drives size decoding.
    assert model.mean_sizes.shape == (18, 3)


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
    pos_seed = torch.rand(10, 3)
    x_seed = torch.rand(10, 8)
    batch_seed = torch.zeros(10, dtype=torch.long)
    with torch.no_grad():
        pos_vote, x_vote, batch_vote = vgen(x_seed, pos_seed, batch_seed)
    assert pos_vote.shape == (10, 3)
    assert x_vote.shape == (10, 8)
    assert torch.equal(batch_vote, batch_seed)


def test_votenet_registered_variants() -> None:
    names = list_models("votenet*", task="detection")
    assert "votenet.scannet.fair" in names
    assert "votenet.sunrgbd.fair" in names


def test_votenet_create_model_no_pretrained() -> None:
    model = create_model("votenet.sunrgbd.fair", task="detection")
    assert isinstance(model, VoteNetDetection)
    assert model.num_classes == 10
    assert model.num_heading_bin == 12
    assert model.num_size_cluster == 10
    assert model.sampling == "seed_fps"


def test_votenet_decode_negates_native_heading() -> None:
    """`decode` returns counter-clockwise headings: the negated bin-decoded angle; all else is unchanged."""
    torch.manual_seed(0)
    model = _create_votenet(num_heading_bin=12, num_classes=10, num_size_cluster=10, mean_sizes=[[1.0, 1.0, 1.0]] * 10)
    b, k, nh, ns = 2, model.num_proposal, 12, 10
    out: VoteNetOutput = {
        "objectness_scores": torch.randn(b, k, 2),
        "center": torch.randn(b, k, 3),
        "heading_scores": torch.randn(b, k, nh),
        "heading_residuals_normalized": torch.randn(b, k, nh),
        "heading_residuals": torch.randn(b, k, nh) * 0.2,
        "size_scores": torch.randn(b, k, ns),
        "size_residuals_normalized": torch.randn(b, k, ns, 3),
        "size_residuals": torch.randn(b, k, ns, 3) * 0.1,
        "sem_cls_scores": torch.randn(b, k, 10),
        "pos_vote_aggr": torch.randn(b, k, 3),
        "pos_seed": torch.randn(b * 4, 3),
        "batch_seed": torch.arange(b).repeat_interleave(4),
        "seed_indices": torch.zeros(b * 4, dtype=torch.long),
        "pos_vote": torch.randn(b * 4, 3),
        "batch_vote": torch.arange(b).repeat_interleave(4),
    }
    det = model.decode(out)

    heading_class = out["heading_scores"].argmax(dim=-1)
    heading_residual = out["heading_residuals"].gather(2, heading_class.unsqueeze(-1)).squeeze(-1)
    native = class_to_angle(heading_class, heading_residual, nh)
    size_class = out["size_scores"].argmax(dim=-1)
    size_residual = out["size_residuals"].gather(2, size_class.view(b, k, 1, 1).expand(-1, -1, 1, 3)).squeeze(2)
    size = class_to_size(size_class.reshape(-1), size_residual.reshape(-1, 3), model.mean_sizes)

    assert torch.allclose(det["boxes"][:, 6], -native.reshape(-1))
    assert torch.equal(det["boxes"][:, :3], out["center"].reshape(-1, 3))
    assert torch.allclose(det["boxes"][:, 3:6], size)
    assert torch.equal(det["labels"], out["sem_cls_scores"].softmax(-1).argmax(-1).reshape(-1))


def test_votenet_output_feeds_loss_directly() -> None:
    """The model's raw packed output feeds `VoteNetLoss` with no glue: the loss self-densifies."""
    torch.manual_seed(0)
    model = _create_votenet().to(DEVICE).eval()
    data = _make_inputs(in_channels=model.in_channels)
    with torch.no_grad():
        output = model(data["x"], data["pos"], data["batch"])

    loss_fn = VoteNetLoss(
        num_heading_bin=model.num_heading_bin,
        num_size_cluster=model.num_size_cluster,
        num_classes=int(model.num_classes),
        mean_sizes=model.mean_sizes,
    ).to(DEVICE)
    batch_size, num_point, max_obj = 2, 3000, 4
    box_label_mask = torch.zeros(batch_size, max_obj, device=DEVICE)
    box_label_mask[:, :2] = 1.0
    gt: Dict[str, Tensor] = {
        "center_label": torch.randn(batch_size, max_obj, 3, device=DEVICE),
        "heading_class_label": torch.zeros(batch_size, max_obj, dtype=torch.long, device=DEVICE),
        "heading_residual_label": torch.randn(batch_size, max_obj, device=DEVICE),
        "size_class_label": torch.randint(0, model.num_size_cluster, (batch_size, max_obj), device=DEVICE),
        "size_residual_label": torch.randn(batch_size, max_obj, 3, device=DEVICE),
        "sem_cls_label": torch.randint(0, int(model.num_classes), (batch_size, max_obj), device=DEVICE),
        "box_label_mask": box_label_mask,
        "vote_label": torch.randn(batch_size, num_point, 9, device=DEVICE),
        "vote_label_mask": (torch.rand(batch_size, num_point, device=DEVICE) > 0.5).long(),
        "batch": data["batch"],
    }
    result = loss_fn(output, gt)
    assert result["loss"].ndim == 0
    assert torch.isfinite(result["loss"])
