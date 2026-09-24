import math
from typing import Any, Dict

import pytest
import torch
from torch import Tensor

import torch_pointcloud.transforms.functional as F
from torch_pointcloud.losses import VoteNetLoss
from torch_pointcloud.models import create_model, list_models
from torch_pointcloud.models.votenet import (
    EncodeVoteNetTargets,
    GenerateVoteLabels,
    VoteNetDetection,
    VoteNetOutput,
    VotingModule,
)
from torch_pointcloud.utils.box3d import class_to_angle, class_to_size
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
        num_heading_bins=1,
        num_size_clusters=18,
        mean_sizes=[[1.0, 1.0, 1.0]] * 18,
        num_proposals=16,
        vote_factor=1,
        sampling="vote_fps",
        sa_channels=[[16, 16, 32], [32, 32, 64], [64, 64, 64], [64, 64, 64]],
        sa_num_points=[256, 128, 64, 32],
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
    out: Dict[str, Tensor],
    batch_size: int,
    num_proposals: int,
    nh: int,
    ns: int,
    nc: int,
) -> None:
    assert out["objectness_scores"].shape == (batch_size, num_proposals, 2)
    assert out["center"].shape == (batch_size, num_proposals, 3)
    assert out["heading_scores"].shape == (batch_size, num_proposals, nh)
    assert out["heading_residuals"].shape == (batch_size, num_proposals, nh)
    assert out["size_scores"].shape == (batch_size, num_proposals, ns)
    assert out["size_residuals"].shape == (batch_size, num_proposals, ns, 3)
    assert out["sem_cls_scores"].shape == (batch_size, num_proposals, nc)
    assert out["pos_vote_aggr"].shape == (batch_size, num_proposals, 3)


def test_votenet_scannet_forward_shapes() -> None:
    model = create_model("votenet.scannet.fair", task="detection").to(DEVICE).eval()
    data = _make_inputs(in_channels=model.in_channels)
    with torch.no_grad():
        out = model(data["x"], data["pos"], data["batch"])
        x_seed, _, _, _ = model.forward_features(data["x"], data["pos"], data["batch"])
    assert x_seed.shape[1] == model.num_features
    _assert_proposal_shapes(out, batch_size=2, num_proposals=256, nh=1, ns=18, nc=18)
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
    _assert_proposal_shapes(out, batch_size=2, num_proposals=256, nh=12, ns=10, nc=10)


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
        _create_votenet(num_size_clusters=3, mean_sizes=[[1.0, 1.0, 1.0]])


def test_votenet_mean_sizes_not_persisted() -> None:
    # The reference rebuilds mean_sizes on the fly, so it must stay out of the checkpoint.
    model = create_model("votenet.scannet.fair", task="detection")
    assert isinstance(model, VoteNetDetection)
    assert "mean_sizes" not in model.state_dict()
    # ...but it still moves with the module and drives size decoding.
    assert model.mean_sizes.shape == (18, 3)


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
    assert model.num_heading_bins == 12
    assert model.num_size_clusters == 10
    assert model.sampling == "seed_fps"


def test_votenet_scannet_registered_seed_fps() -> None:
    # The reference eval command samples proposal centers with seed_fps on ScanNet too.
    model = create_model("votenet.scannet.fair", task="detection")
    assert isinstance(model, VoteNetDetection)
    assert model.sampling == "seed_fps"
    assert model.vote_factor == 1


def test_votenet_sampling_is_weight_free() -> None:
    vote = _create_votenet(sampling="vote_fps")
    seed = _create_votenet(sampling="seed_fps")
    assert list(vote.state_dict().keys()) == list(seed.state_dict().keys())


def test_votenet_decode_negates_native_heading() -> None:
    """`decode` returns counter-clockwise headings: the negated bin-decoded angle; all else is unchanged."""
    torch.manual_seed(0)
    model = _create_votenet(
        num_heading_bins=12,
        num_classes=10,
        num_size_clusters=10,
        mean_sizes=[[1.0, 1.0, 1.0]] * 10,
    )
    b, k, nh, ns = 2, model.num_proposals, 12, 10
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
        num_heading_bins=model.num_heading_bins,
        num_size_clusters=model.num_size_clusters,
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
        "size_class_label": torch.randint(0, model.num_size_clusters, (batch_size, max_obj), device=DEVICE),
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


def _box(heading: float = 0.0) -> Tensor:
    return torch.tensor([[1.0, 0.5, 0.3, 0.8, 0.6, 0.4, heading]])


def test_generate_vote_labels_overlapping_boxes_get_distinct_votes() -> None:
    boxes = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
            [0.25, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
        ]
    )
    pos = torch.tensor([[0.1, 0.0, 0.0], [-0.4, 0.0, 0.0], [5.0, 5.0, 5.0]])
    out = GenerateVoteLabels(oriented=False)({"pos": pos, "box": boxes})
    votes = out["vote_label"]
    # Inside both boxes: slot 0 votes for box 0, slot 1 for box 1, slot 2 repeats the first vote.
    assert torch.allclose(votes[0, 0:3], boxes[0, 0:3] - pos[0])
    assert torch.allclose(votes[0, 3:6], boxes[1, 0:3] - pos[0])
    assert torch.allclose(votes[0, 6:9], votes[0, 0:3])
    # Inside box 0 only: its offset is tiled into all three slots.
    assert torch.allclose(votes[1], (boxes[0, 0:3] - pos[1]).repeat(3))
    assert out["vote_label_mask"].tolist() == [1, 1, 0]


def test_generate_vote_labels_oriented_containment_is_counterclockwise() -> None:
    heading = math.pi / 4
    rotation = F.rotation_matrix(heading, axis=2)
    box = torch.tensor([[0.0, 0.0, 0.0, 2.0, 0.5, 1.0, heading]])
    pos = torch.tensor([[0.9, 0.2, 0.0]]) @ rotation.T  # inside only if the heading rotates counterclockwise
    out = GenerateVoteLabels(oriented=True)({"pos": pos, "box": box})
    assert out["vote_label_mask"].tolist() == [1]


def test_generate_vote_labels_marks_in_and_out() -> None:
    box = _box(heading=0.0)
    pts = torch.tensor([[1.0, 0.5, 0.3], [5.0, 5.0, 5.0]])
    data = {"pos": pts, "box": box.clone()}
    out = GenerateVoteLabels(oriented=True, gt_vote_factor=3)(data)
    assert out["vote_label"].shape == (2, 9)
    assert out["vote_label_mask"].tolist() == [1, 0]
    assert torch.allclose(out["vote_label"][0, 0:3], box[0, 0:3] - pts[0])
    assert torch.allclose(out["vote_label"][0, 0:3], out["vote_label"][0, 3:6])
    assert torch.allclose(out["vote_label"][0, 0:3], out["vote_label"][0, 6:9])
    assert torch.allclose(out["vote_label"][1], torch.zeros(9))


def test_generate_vote_labels_oriented_vs_axis_aligned() -> None:
    box = _box(heading=math.pi / 4)
    corner = torch.tensor([[1.0 + 0.38, 0.5 + 0.28, 0.3]])
    data_axis = {"pos": corner.clone(), "box": box.clone()}
    data_oriented = {"pos": corner.clone(), "box": box.clone()}
    out_axis = GenerateVoteLabels(oriented=False)(data_axis)
    out_oriented = GenerateVoteLabels(oriented=True)(data_oriented)
    assert out_axis["vote_label_mask"].item() == 1
    assert out_oriented["vote_label_mask"].item() == 0


def test_encode_votenet_targets_shapes_and_roundtrip() -> None:
    mean = torch.ones(10, 3) * 0.5
    box = _box(heading=0.6)
    data = {"box": box.clone(), "label": torch.tensor([2])}
    out = EncodeVoteNetTargets(num_heading_bins=12, mean_sizes=mean, max_num_obj=64)(data)
    assert out["center_label"].shape == (64, 3)
    assert out["heading_class_label"].shape == (64,)
    assert out["heading_residual_label"].shape == (64,)
    assert out["size_class_label"].shape == (64,)
    assert out["size_residual_label"].shape == (64, 3)
    assert out["sem_cls_label"].shape == (64,)
    assert out["box_label_mask"].shape == (64,)
    assert out["box_label_mask"].sum().item() == 1
    assert out["size_class_label"][0].item() == 2
    assert out["sem_cls_label"][0].item() == 2
    assert torch.allclose(out["center_label"][0], box[0, 0:3])
    recovered = class_to_size(out["size_class_label"][:1], out["size_residual_label"][:1], mean)
    assert torch.allclose(recovered[0], box[0, 3:6], atol=1e-5)


def test_encode_votenet_targets_truncates_to_max_num_obj() -> None:
    mean = torch.ones(10, 3) * 0.5
    boxes = _box(heading=0.0).repeat(5, 1)
    data = {"box": boxes, "label": torch.ones(5, dtype=torch.long)}
    out = EncodeVoteNetTargets(num_heading_bins=12, mean_sizes=mean, max_num_obj=3)(data)
    assert out["center_label"].shape == (3, 3)
    assert out["box_label_mask"].sum().item() == 3


def test_vote_then_encode_keeps_boxes_and_writes_all_labels() -> None:
    mean = torch.ones(10, 3) * 0.5
    pos = torch.rand(2048, 3) * 4
    boxes = torch.tensor(
        [
            [1.0, 1.0, 1.0, 1.0, 0.8, 0.6, 0.2],
            [2.0, 2.0, 1.0, 1.2, 1.0, 0.8, 0.0],
        ]
    )
    data = {"pos": pos.clone(), "box": boxes.clone(), "label": torch.tensor([3, 1])}
    data = GenerateVoteLabels(pos_key="pos", box_key="box")(data)
    out = EncodeVoteNetTargets(box_key="box", num_heading_bins=12, mean_sizes=mean, max_num_obj=64)(data)

    assert torch.equal(out["box"], boxes)
    assert out["vote_label"].shape == (2048, 9)
    assert out["vote_label_mask"].shape == (2048,)

    shapes = {
        "center_label": (64, 3),
        "heading_class_label": (64,),
        "heading_residual_label": (64,),
        "size_class_label": (64,),
        "size_residual_label": (64, 3),
        "sem_cls_label": (64,),
        "box_label_mask": (64,),
    }
    for key, shape in shapes.items():
        assert out[key].shape == shape

    assert out["heading_class_label"].dtype == torch.long
    assert out["size_class_label"].dtype == torch.long
    assert out["sem_cls_label"].dtype == torch.long
    assert out["center_label"].dtype == torch.float32
    assert out["size_residual_label"].dtype == torch.float32
    assert out["box_label_mask"].dtype == torch.float32

    assert out["box_label_mask"][:2].tolist() == [1.0, 1.0]
    assert out["box_label_mask"][2:].sum().item() == 0.0
    assert out["sem_cls_label"][:2].tolist() == [3, 1]
    assert out["size_class_label"][:2].tolist() == [3, 1]
