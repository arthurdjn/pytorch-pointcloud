"""VoteNet detection model."""

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TypedDict, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import MLP

import torch_pointcloud.transforms as T
import torch_pointcloud.transforms.functional as F
from torch_pointcloud.datasets.scannet import SCANNET_DETECTION_CLASSES
from torch_pointcloud.datasets.sunrgbd import SUNRGBD_CLASSES
from torch_pointcloud.layers.pointnet2_blocks import FPModule, SAModule
from torch_pointcloud.utils.cluster import fps
from torch_pointcloud.utils.conversion import ensure_list
from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.types import Detection3D, OptTensor

from ._base import DetectionModel
from ._registry import WeightsDict, register_model


class VoteNetOutput(TypedDict):
    """Decoded VoteNet proposals for a batch of $B$ scenes with $K$ proposals each."""

    objectness_scores: Tensor
    center: Tensor
    heading_scores: Tensor
    heading_residuals_normalized: Tensor
    heading_residuals: Tensor
    size_scores: Tensor
    size_residuals_normalized: Tensor
    size_residuals: Tensor
    sem_cls_scores: Tensor
    pos_vote_aggr: Tensor
    pos_seed: Tensor
    batch_seed: Tensor
    seed_indices: Tensor
    pos_vote: Tensor
    batch_vote: Tensor


class VoteNetBackbone(nn.Module):
    r"""PointNet++ single-scale-grouping backbone (set abstraction + feature propagation).

    Reuses [`SAModule`][torch_pointcloud.layers.pointnet2_blocks.SAModule] /
    [`FPModule`][torch_pointcloud.layers.pointnet2_blocks.FPModule]; every layer width, sample count,
    radius and neighbor cap is a constructor argument (no hardcoded sizes). The seeds are the points at
    the second SA resolution, recovered by the feature-propagation layers. Their indices into the
    original packed input are tracked through the first two SA samplings (`idx1[idx2]`) for the voting
    loss, so the SA samplings of those two blocks are computed here and threaded in.

    Args:
        in_channels: Input feature channels per point (excluding xyz).
        sa_channels: Per-SA-block MLP channel lists, e.g. `[[64, 64, 128], ...]`.
        sa_npoints: Per-SA-block farthest-point-sample counts.
        sa_radii: Per-SA-block ball-query radii.
        sa_num_neighbors: Per-SA-block neighbor caps.
        fp_channels: Per-FP-block MLP channel lists. The $i$-th FP block skips to the
            $(\text{n\_sa} - 2 - i)$-th SA output, so two FP blocks recover the SA2 resolution.
        act: Activation for every block.
        act_kwargs: Extra activation arguments.
        norm: Normalization for every block.
        norm_kwargs: Extra normalization arguments.
    """

    def __init__(
        self,
        in_channels: int,
        *,
        sa_channels: Sequence[Sequence[int]],
        sa_npoints: Sequence[int],
        sa_radii: Sequence[float],
        sa_num_neighbors: Sequence[int],
        fp_channels: Sequence[Sequence[int]],
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.sa_npoints = ensure_list(sa_npoints)

        skip_channels: List[int] = []

        self.sa_modules = nn.ModuleList()
        for channels, npoint, radius, num_neighbors in zip(sa_channels, sa_npoints, sa_radii, sa_num_neighbors):
            block_channels = ensure_list(channels)
            sa_block = SAModule(
                in_channels=in_channels,
                channels=block_channels,
                num_points=npoint,
                radii=radius,
                num_neighbors=num_neighbors,
                use_pos=True,
                normalize_pos=True,
                pos_first=True,
                sort_neighbors=True,
                pool="max",
                bias=False,
                act=act,
                act_kwargs=act_kwargs,
                norm=norm,
                norm_kwargs=norm_kwargs,
            )

            self.sa_modules.append(sa_block)
            in_channels = block_channels[-1]
            skip_channels.append(in_channels)

        num_sa_blocks = len(self.sa_modules)

        self.fp_modules = nn.ModuleList()
        for i, channels in enumerate(fp_channels):
            block_channels = ensure_list(channels)
            fp_block = FPModule(
                in_channels=in_channels + skip_channels[num_sa_blocks - 2 - i],
                channels=block_channels,
                k=3,
                weighting="inverse",
                eps=1e-8,
                bias=False,
                act=act,
                act_kwargs=act_kwargs,
                norm=norm,
                norm_kwargs=norm_kwargs,
            )

            self.fp_modules.append(fp_block)
            in_channels = block_channels[-1]

        self.out_channels = in_channels

    def forward(self, x: Tensor, pos: Tensor, batch: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        # Sample the first two SA blocks here so the seed indices can be traced to the input
        # (`seed_indices = idx1[idx2]`); deeper blocks sample internally as their indices are unused.
        head_idx: List[Tensor] = []
        intermediates: List[Tuple[Tensor, Tensor, Tensor]] = []
        for i, sa_block in enumerate(self.sa_modules):
            if i < 2:
                idx = fps(pos, batch, num_nodes=self.sa_npoints[i], random_start=self.training)
                x, pos, batch = sa_block(x, pos, batch, idx)
                head_idx.append(idx)
            else:
                x, pos, batch = sa_block(x, pos, batch)

            intermediates.append((x, pos, batch))

        num_sa_blocks = len(self.sa_modules)
        x, pos, batch = intermediates[-1]
        for i, fp_block in enumerate(self.fp_modules):
            x_skip, pos_skip, batch_skip = intermediates[num_sa_blocks - 2 - i]
            x, pos, batch = fp_block(x, pos, batch, x_skip, pos_skip, batch_skip)

        seed_indices = head_idx[0][head_idx[1]]
        return x, pos, batch, seed_indices


class VotingModule(nn.Module):
    r"""Hough voting: each seed predicts a vote offset and a residual feature.

    Mirrors the reference `VotingModule`. Because the residual is added to the seed feature, the input
    and output feature dims are equal.

    Args:
        vote_factor: Number of votes generated per seed.
        seed_feature_dim: Channel count of the seed features.
        act: Activation type or callable.
        act_kwargs: Extra activation arguments.
        norm: Normalization type or callable.
        norm_kwargs: Extra normalization arguments.
    """

    def __init__(
        self,
        vote_factor: int,
        seed_feature_dim: int,
        *,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.vote_factor = vote_factor
        self.in_dim = seed_feature_dim
        self.out_dim = seed_feature_dim
        self.mlp = MLP(
            [self.in_dim, self.in_dim, self.in_dim, (3 + self.out_dim) * self.vote_factor],
            act=act,
            act_kwargs=act_kwargs,
            norm=norm,
            norm_kwargs=norm_kwargs,
            plain_last=True,
        )

    def forward(self, x: Tensor, pos: Tensor, batch: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        preds = self.mlp(x)  # (N, (3 + out_dim) * vote_factor)

        n = pos.size(0)
        preds = preds.view(n, self.vote_factor, 3 + self.out_dim)
        pos_vote = (pos.unsqueeze(1) + preds[:, :, 0:3]).reshape(n * self.vote_factor, 3)
        x_vote = (x.unsqueeze(1) + preds[:, :, 3:]).reshape(n * self.vote_factor, self.out_dim)
        batch_vote = batch.repeat_interleave(self.vote_factor)
        return pos_vote, x_vote, batch_vote


class VoteNetProposalModule(nn.Module):
    r"""Vote aggregation and proposal generation.

    Mirrors the reference `ProposalModule`: cluster the votes with a single set-abstraction layer
    ([`SAModule`][torch_pointcloud.layers.pointnet2_blocks.SAModule]), then a 3-layer linear head
    decodes objectness, center, heading and size bins/residuals and semantic class per proposal.

    Args:
        num_classes: Number of semantic classes.
        num_heading_bin: Number of heading-angle bins.
        num_size_cluster: Number of size templates.
        num_proposal: Number of proposals (= aggregation centroids) per scene.
        sampling: Aggregation-center sampling, `"vote_fps"` or `"seed_fps"`.
        seed_channels: Channel count of the (vote) input features.
        vote_aggr_channels: MLP channels of the vote-aggregation set-abstraction layer.
        vote_aggr_radius: Ball-query radius of the vote-aggregation layer.
        vote_aggr_num_neighbors: Neighbor cap of the vote-aggregation layer.
        act: Activation type or callable.
        act_kwargs: Extra activation arguments.
        norm: Normalization type or callable.
        norm_kwargs: Extra normalization arguments.
    """

    def __init__(
        self,
        num_classes: int,
        num_heading_bin: int,
        num_size_cluster: int,
        num_proposal: int,
        sampling: str,
        seed_channels: int,
        *,
        vote_aggr_channels: Sequence[int],
        vote_aggr_radius: float,
        vote_aggr_num_neighbors: int,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.num_proposal = num_proposal
        self.sampling = sampling

        self.vote_aggr = SAModule(
            in_channels=seed_channels,
            channels=list(vote_aggr_channels),
            num_points=num_proposal,
            radii=vote_aggr_radius,
            num_neighbors=vote_aggr_num_neighbors,
            use_pos=True,
            normalize_pos=True,
            pos_first=True,
            sort_neighbors=True,
            pool="max",
            bias=False,
            act=act,
            act_kwargs=act_kwargs,
            norm=norm,
            norm_kwargs=norm_kwargs,
        )
        self.aggr_dim = vote_aggr_channels[-1]
        out_dim = 2 + 3 + num_heading_bin * 2 + num_size_cluster * 4 + num_classes
        self.mlp = MLP(
            [self.aggr_dim, self.aggr_dim, self.aggr_dim, out_dim],
            act=act,
            act_kwargs=act_kwargs,
            norm=norm,
            norm_kwargs=norm_kwargs,
            plain_last=True,
        )

    def forward(
        self,
        pos_vote: Tensor,
        x_vote: Tensor,
        batch_vote: Tensor,
        pos_seed: Tensor,
        batch_seed: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        if self.sampling == "vote_fps":
            idx = fps(pos_vote, batch_vote, num_nodes=self.num_proposal, random_start=self.training)
        elif self.sampling == "seed_fps":
            idx = fps(pos_seed, batch_seed, num_nodes=self.num_proposal, random_start=self.training)
        else:
            raise ValueError(f"Unknown sampling strategy {self.sampling!r}. Expected 'vote_fps' or 'seed_fps'.")

        x_aggr, pos_aggr, batch_aggr = self.vote_aggr(x_vote, pos_vote, batch_vote, idx)

        x_aggr = self.mlp(x_aggr)
        return x_aggr, pos_aggr, batch_aggr

    def decode(
        self,
        preds: Tensor,
        pos_aggr: Tensor,
        batch_aggr: Tensor,
        mean_sizes: Tensor,
    ) -> Dict[str, Tensor]:
        r"""Splits the flat head output into the per-proposal box attributes.

        Centers are predicted as offsets from the aggregation centroids, and heading and size residuals are
        denormalized by the bin width and the size templates respectively.

        Args:
            preds: Flat head output, shape $(B \cdot Q, C)$.
            pos_aggr: Aggregation centroid positions, shape $(B \cdot Q, 3)$.
            batch_aggr: Per-centroid scene index, shape $(B \cdot Q,)$.
            mean_sizes: Size templates, shape $(S, 3)$.

        Returns:
            The per-proposal objectness, center, heading, size and semantic predictions, each of shape
            $(B, Q, \ldots)$.
        """
        batch_size = int(batch_aggr.max().item()) + 1 if batch_aggr.numel() else 0
        num_proposal = self.num_proposal
        nh = self.num_heading_bin
        ns = self.num_size_cluster

        preds = preds.view(batch_size, num_proposal, -1)
        pos_vote_aggr = pos_aggr.view(batch_size, num_proposal, 3)

        objectness = preds[..., 0:2]
        center = pos_vote_aggr + preds[..., 2:5]
        heading_scores = preds[..., 5 : 5 + nh]
        heading_res_norm = preds[..., 5 + nh : 5 + nh * 2]
        size_scores = preds[..., 5 + nh * 2 : 5 + nh * 2 + ns]
        size_res_norm = preds[..., 5 + nh * 2 + ns : 5 + nh * 2 + ns * 4].view(batch_size, num_proposal, ns, 3)
        sem_cls_scores = preds[..., 5 + nh * 2 + ns * 4 :]

        return {
            "objectness_scores": objectness,
            "center": center,
            "heading_scores": heading_scores,
            "heading_residuals_normalized": heading_res_norm,
            "heading_residuals": heading_res_norm * (math.pi / nh),
            "size_scores": size_scores,
            "size_residuals_normalized": size_res_norm,
            "size_residuals": size_res_norm * mean_sizes.view(1, 1, ns, 3),
            "sem_cls_scores": sem_cls_scores,
            "pos_vote_aggr": pos_vote_aggr,
        }


class VoteNetDetection(DetectionModel):
    r"""VoteNet 3D object detector (packed point format).

    Reference: :arxiv: [Qi et al., 2019](https://arxiv.org/abs/1904.09664).
    Reference implementation: :github:
    [facebookresearch/votenet](https://github.com/facebookresearch/votenet).

    A PointNet++ backbone extracts seed points; a voting module shifts each seed toward its object
    center; a proposal module clusters the votes and decodes a fixed number of oriented (or
    axis-aligned) box proposals per scene.

    Args:
        in_channels: Input feature channels per point excluding xyz (e.g. $1$ for a floor-relative
            height feature, $4$ for height + RGB).
        num_classes: Number of semantic classes.
        num_heading_bin: Number of heading-angle bins ($1$ for axis-aligned ScanNet boxes, $12$ for
            oriented SUN RGB-D boxes).
        num_size_cluster: Number of size templates (one per class here).
        mean_sizes: Per-template mean box size, shape $(\text{num\_size\_cluster}, 3)$.
        num_proposal: Number of box proposals per scene.
        vote_factor: Votes generated per seed.
        sampling: Aggregation-center sampling, `"vote_fps"` or `"seed_fps"`.
        sa_channels: Per-SA-block MLP channel lists for the backbone.
        sa_npoints: Per-SA-block farthest-point-sample counts.
        sa_radii: Per-SA-block ball-query radii.
        sa_num_neighbors: Per-SA-block neighbor caps.
        fp_channels: Per-FP-block MLP channel lists for the backbone.
        vote_aggr_channels: MLP channels of the proposal vote-aggregation layer.
        vote_aggr_radius: Ball-query radius of the vote-aggregation layer.
        vote_aggr_num_neighbors: Neighbor cap of the vote-aggregation layer.
        act: Activation type or callable.
        act_kwargs: Extra activation arguments.
        norm: Normalization type or callable.
        norm_kwargs: Extra normalization arguments.
    """

    mean_sizes: Tensor  # registered as a non-persistent buffer in __init__

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        *,
        num_heading_bin: int,
        num_size_cluster: int,
        mean_sizes: Union[Tensor, List[List[float]]],
        num_proposal: int,
        vote_factor: int,
        sampling: str,
        sa_channels: Sequence[Sequence[int]],
        sa_npoints: Sequence[int],
        sa_radii: Sequence[float],
        sa_num_neighbors: Sequence[int],
        fp_channels: Sequence[Sequence[int]],
        vote_aggr_channels: Sequence[int],
        vote_aggr_radius: float,
        vote_aggr_num_neighbors: int,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(in_channels=in_channels, num_classes=num_classes)
        if sampling == "seed_fps" and vote_factor != 1:
            raise ValueError("'seed_fps' sampling requires vote_factor == 1.")

        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.num_proposal = num_proposal
        self.vote_factor = vote_factor
        self.sampling = sampling
        self.spatial_dim = 3

        mean = torch.as_tensor(mean_sizes, dtype=torch.float32)
        if mean.shape != (num_size_cluster, 3):
            raise ValueError(f"`mean_sizes` must have shape ({num_size_cluster}, 3), got {tuple(mean.shape)}.")

        # Not part of the checkpoint (the reference rebuilds it on the fly); persistent=False
        # keeps it out of the state dict while still moving with `.to(device)`.
        self.register_buffer("mean_sizes", mean, persistent=False)

        self.backbone = VoteNetBackbone(
            in_channels,
            sa_channels=sa_channels,
            sa_npoints=sa_npoints,
            sa_radii=sa_radii,
            sa_num_neighbors=sa_num_neighbors,
            fp_channels=fp_channels,
            act=act,
            act_kwargs=act_kwargs,
            norm=norm,
            norm_kwargs=norm_kwargs,
        )
        self.vgen = VotingModule(
            vote_factor,
            self.backbone.out_channels,
            act=act,
            act_kwargs=act_kwargs,
            norm=norm,
            norm_kwargs=norm_kwargs,
        )
        self.proposal = VoteNetProposalModule(
            num_classes=num_classes,
            num_heading_bin=num_heading_bin,
            num_size_cluster=num_size_cluster,
            num_proposal=num_proposal,
            sampling=sampling,
            seed_channels=self.backbone.out_channels,
            vote_aggr_channels=vote_aggr_channels,
            vote_aggr_radius=vote_aggr_radius,
            vote_aggr_num_neighbors=vote_aggr_num_neighbors,
            act=act,
            act_kwargs=act_kwargs,
            norm=norm,
            norm_kwargs=norm_kwargs,
        )

    def reset_classifier(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.proposal.num_classes = num_classes
        out_dim = 2 + 3 + self.num_heading_bin * 2 + self.num_size_cluster * 4 + num_classes
        self.proposal.mlp.lins[-1] = nn.Linear(self.proposal.aggr_dim, out_dim)

    def forward_features(self, x: OptTensor, pos: Tensor, batch: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        assert x is not None, "VoteNet requires input features (got x=None)."
        return self.backbone(x, pos, batch)

    def forward_head(
        self,
        x: Tensor,
        pos: Tensor,
        batch: Tensor,
        seed_indices: Tensor,
    ) -> VoteNetOutput:
        pos_vote, x_vote, batch_vote = self.vgen(x, pos, batch)
        x_vote = x_vote / x_vote.norm(p=2, dim=1, keepdim=True)

        preds, pos_aggr, batch_aggr = self.proposal(pos_vote, x_vote, batch_vote, pos, batch)
        decoded = self.proposal.decode(preds, pos_aggr, batch_aggr, self.mean_sizes)

        return {
            **decoded,  # type: ignore[typeddict-item]
            "pos_seed": pos,
            "batch_seed": batch,
            "seed_indices": seed_indices,
            "pos_vote": pos_vote,
            "batch_vote": batch_vote,
        }

    def forward(self, x: OptTensor, pos: Tensor, batch: Tensor) -> VoteNetOutput:
        x_seed, pos_seed, batch_seed, seed_indices = self.forward_features(x, pos, batch)
        return self.forward_head(x_seed, pos_seed, batch_seed, seed_indices)

    @torch.no_grad()
    def decode(self, out: VoteNetOutput) -> Detection3D:
        r"""Decode a forward output into raw per-proposal detections (no NMS, threshold, or filtering).

        Builds one oriented box per proposal from the predicted heading/size bins, scores it by objectness,
        and labels it by the argmax semantic class. The heading head predicts negated angles, so the decoded
        heading is negated to return counter-clockwise headings (the library box convention). The result is
        the full unfiltered proposal set; the evaluation pipeline applies point-count filtering, NMS, score
        thresholding, and the indoor per-class expansion (driven by the returned `class_probs`) via the
        `torch_pointcloud.utils.box3d` utilities.

        Args:
            out: A `VoteNetOutput` from `forward`.

        Returns:
            Packed proposals `{"boxes", "scores", "labels", "batch", "class_probs"}` (PyG layout), where the
            per-proposal score is objectness, the label is the argmax semantic class, and `class_probs` holds
            the softmaxed semantic-class probabilities.

        Shape:
            - boxes: $(B \cdot P, 7)$
            - scores / labels / batch: $(B \cdot P,)$
            - class_probs: $(B \cdot P, C)$
        """
        batch_size, num_proposal = out["center"].shape[:2]

        heading_class = out["heading_scores"].argmax(dim=-1)
        heading_residual = out["heading_residuals"].gather(2, heading_class.unsqueeze(-1)).squeeze(-1)
        angle = -F.class_to_angle(heading_class, heading_residual, self.num_heading_bin)

        size_class = out["size_scores"].argmax(dim=-1)
        size_gather = size_class.view(batch_size, num_proposal, 1, 1).expand(-1, -1, 1, 3)
        size_residual = out["size_residuals"].gather(2, size_gather).squeeze(2)
        size = F.class_to_size(size_class.reshape(-1), size_residual.reshape(-1, 3), self.mean_sizes)

        boxes = torch.cat([out["center"], size.view(batch_size, num_proposal, 3), angle.unsqueeze(-1)], dim=-1)
        objectness = out["objectness_scores"].softmax(dim=-1)[..., 1]
        class_probs = out["sem_cls_scores"].softmax(dim=-1)
        batch = torch.arange(batch_size, device=boxes.device).repeat_interleave(num_proposal)
        return {
            "boxes": boxes.reshape(-1, 7),
            "scores": objectness.reshape(-1),
            "labels": class_probs.argmax(dim=-1).reshape(-1),
            "batch": batch,
            "class_probs": class_probs.reshape(-1, class_probs.size(-1)),
        }


_SCANNET_MEAN_SIZES = [
    [0.769667, 0.811602, 0.925737],
    [1.876858, 1.842560, 1.193157],
    [0.613280, 0.614861, 0.718270],
    [1.395501, 1.512155, 0.834436],
    [0.979496, 1.067515, 0.632969],
    [0.531663, 0.595558, 1.750015],
    [0.962471, 0.724623, 1.148187],
    [0.832219, 1.049094, 1.687566],
    [0.211322, 0.420616, 0.537285],
    [1.444007, 1.897083, 0.269857],
    [1.029426, 1.404080, 0.875543],
    [1.376641, 0.655218, 1.681313],
    [0.665082, 0.711119, 1.298853],
    [0.419992, 0.379069, 1.751397],
    [0.593596, 0.591249, 0.739190],
    [0.508676, 0.506561, 0.301362],
    [1.151153, 1.054630, 0.497068],
    [0.475353, 0.492495, 0.580212],
]

_SUNRGBD_MEAN_SIZES = [
    [2.114256, 1.620300, 0.927272],
    [0.791118, 1.279516, 0.718182],
    [0.923508, 1.867419, 0.845495],
    [0.591958, 0.552978, 0.827272],
    [0.699104, 0.454178, 0.756250],
    [0.695190, 1.346299, 0.736364],
    [0.528526, 1.002642, 1.172878],
    [0.500618, 0.632163, 0.683424],
    [0.404671, 1.071108, 1.688889],
    [0.765840, 1.398258, 0.472728],
]


@register_model(
    "votenet.scannet.fair",
    task="detection",
    weights=WeightsDict(
        url="hf://torch-pointcloud/votenet/votenet.scannet.fair.safetensors",
        dataset="scannet",
        classes=SCANNET_DETECTION_CLASSES,
        author="fair",
        license="MIT",
    ),
    transform=T.Compose(
        [
            T.AxisMinOffset(keys=DataKeys.POS, axis=2, quantile=0.0099, dst_keys="height"),
            # `segment` / `instance` are carried along only when present (detection data is xyz-only).
            T.RandomSample(
                keys=[DataKeys.POS, "height", DataKeys.SEGMENT, DataKeys.INSTANCE],
                num_samples=40000,
                allow_missing_keys=True,
            ),
            T.Cat(keys=["height"], dst_key=DataKeys.X, dim=1),
        ]
    ),
    hparams=dict(
        in_channels=1,
        num_classes=18,
        sa_channels=[[64, 64, 128], [128, 128, 256], [128, 128, 256], [128, 128, 256]],
        sa_npoints=[2048, 1024, 512, 256],
        sa_radii=[0.2, 0.4, 0.8, 1.2],
        sa_num_neighbors=[64, 32, 16, 16],
        fp_channels=[[256, 256], [256, 256]],
        vote_aggr_channels=[128, 128, 128],
        vote_aggr_radius=0.3,
        vote_aggr_num_neighbors=16,
        num_proposal=256,
        vote_factor=1,
        num_heading_bin=1,
        num_size_cluster=18,
        mean_sizes=_SCANNET_MEAN_SIZES,
        sampling="seed_fps",
    ),
)
def votenet_fair_base_scannet(**hparams: Any) -> VoteNetDetection:
    return VoteNetDetection(**hparams)


@register_model(
    "votenet.sunrgbd.fair",
    task="detection",
    weights=WeightsDict(
        url="hf://torch-pointcloud/votenet/votenet.sunrgbd.fair.safetensors",
        dataset="sunrgbd",
        classes=SUNRGBD_CLASSES,
        author="fair",
        license="MIT",
    ),
    transform=T.Compose(
        [
            T.AxisMinOffset(keys=DataKeys.POS, axis=2, quantile=0.0099, dst_keys="height"),
            T.RandomSample(keys=[DataKeys.POS, "height"], num_samples=20000),
            T.Cat(keys=["height"], dst_key=DataKeys.X, dim=1),
        ]
    ),
    hparams=dict(
        in_channels=1,
        num_classes=10,
        sa_channels=[[64, 64, 128], [128, 128, 256], [128, 128, 256], [128, 128, 256]],
        sa_npoints=[2048, 1024, 512, 256],
        sa_radii=[0.2, 0.4, 0.8, 1.2],
        sa_num_neighbors=[64, 32, 16, 16],
        fp_channels=[[256, 256], [256, 256]],
        vote_aggr_channels=[128, 128, 128],
        vote_aggr_radius=0.3,
        vote_aggr_num_neighbors=16,
        num_proposal=256,
        vote_factor=1,
        num_heading_bin=12,
        num_size_cluster=10,
        mean_sizes=_SUNRGBD_MEAN_SIZES,
        sampling="seed_fps",
    ),
)
def votenet_fair_base_sunrgbd(**hparams: Any) -> VoteNetDetection:
    return VoteNetDetection(**hparams)
