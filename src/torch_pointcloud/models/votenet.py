r"""VoteNet: deep Hough voting for 3D object detection in point clouds.

Reference: :arxiv: [Qi et al., 2019](https://arxiv.org/abs/1904.09664).
Reference implementation: :github: [facebookresearch/votenet](https://github.com/facebookresearch/votenet).

This is a packed / flat-batch port of the original dense `(B, N, C)` implementation.
All point tensors use the PyG-style packed layout (`pos` $(N, 3)$, `x` $(N, C)$,
`batch` $(N,)$); proposals, which are a fixed `num_proposal` per scene, are returned
as dense $(B, K, \cdot)$ tensors so the decoded boxes feed directly into the standard
detection-AP pipeline.

The backbone reuses the shared PointNet++ blocks [`SAModule`][torch_pointcloud.layers.pointnet2_blocks.SAModule]
and [`FPModule`][torch_pointcloud.layers.pointnet2_blocks.FPModule]. Set abstraction keeps the $k$
smallest in-radius source indices (`sort_neighbors=True`, reproducing `query_ball_point`) and feature
propagation uses inverse-distance-weighted 3-NN ($k = 3$, $\epsilon = 10^{-8}$). Because every sampled
centroid belongs to the set it groups over, each ball is non-empty, so the packed `scatter` max-pool
matches the dense `max_pool2d` element-for-element in eval mode. The only sources of non-bit-exactness
against the reference CUDA kernels are tie-breaking in farthest-point sampling and in the 3-NN search,
both negligible.
"""

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TypedDict, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import MLP

import torch_pointcloud.transforms as T
from torch_pointcloud.layers.pointnet2_blocks import FPModule, SAModule
from torch_pointcloud.utils.cluster import fps
from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.types import OptTensor

from ._base import DetectionModel
from ._registry import register_model


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
    aggregated_vote_pos: Tensor
    seed_pos: Tensor
    seed_batch: Tensor
    seed_inds: Tensor
    vote_pos: Tensor
    vote_batch: Tensor


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
        self.sa_npoints = list(sa_npoints)
        self.sa_modules = nn.ModuleList()
        ch = in_channels
        sa_out: List[int] = []
        for channels, npoint, r, k in zip(sa_channels, sa_npoints, sa_radii, sa_num_neighbors):
            self.sa_modules.append(
                SAModule(
                    in_channels=ch,
                    channels=list(channels),
                    num_points=npoint,
                    radii=r,
                    num_neighbors=k,
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
            )
            ch = channels[-1]
            sa_out.append(ch)

        n_sa = len(self.sa_modules)
        self.fp_modules = nn.ModuleList()
        self.fp_skip_index: List[int] = []
        prev = sa_out[-1]
        for i, channels in enumerate(fp_channels):
            skip_idx = n_sa - 2 - i
            self.fp_modules.append(
                FPModule(
                    in_channels=prev + sa_out[skip_idx],
                    channels=list(channels),
                    k=3,
                    weighting="inverse",
                    eps=1e-8,
                    bias=False,
                    act=act,
                    act_kwargs=act_kwargs,
                    norm=norm,
                    norm_kwargs=norm_kwargs,
                )
            )
            self.fp_skip_index.append(skip_idx)
            prev = channels[-1]

        self.out_channels = prev

    def forward(self, x: Tensor, pos: Tensor, batch: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        cur_x, cur_pos, cur_batch = x, pos, batch
        levels: List[Tuple[Tensor, Tensor, Tensor]] = []
        # Sample the first two SA blocks here so the seed indices can be traced to the input
        # (`seed_inds = idx1[idx2]`); deeper blocks sample internally as their indices are unused.
        head_idx: List[Tensor] = []
        for i, sa in enumerate(self.sa_modules):
            if i < 2:
                idx = fps(cur_pos, cur_batch, num_nodes=self.sa_npoints[i], random_start=self.training)
                cur_x, cur_pos, cur_batch = sa(cur_x, cur_pos, cur_batch, idx)
                head_idx.append(idx)
            else:
                cur_x, cur_pos, cur_batch = sa(cur_x, cur_pos, cur_batch)
            levels.append((cur_x, cur_pos, cur_batch))

        x_, pos_, batch_ = levels[-1]
        for fp, skip_idx in zip(self.fp_modules, self.fp_skip_index):
            x_skip, pos_skip, batch_skip = levels[skip_idx]
            x_, pos_, batch_ = fp(x_, pos_, batch_, x_skip, pos_skip, batch_skip)

        seed_inds = head_idx[0][head_idx[1]]
        return x_, pos_, batch_, seed_inds


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

    def forward(self, seed_pos: Tensor, seed_x: Tensor, seed_batch: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        net = self.mlp(seed_x)  # (N, (3 + out_dim) * vote_factor)

        n = seed_pos.size(0)
        net = net.view(n, self.vote_factor, 3 + self.out_dim)
        vote_pos = (seed_pos.unsqueeze(1) + net[:, :, 0:3]).reshape(n * self.vote_factor, 3)
        vote_x = (seed_x.unsqueeze(1) + net[:, :, 3:]).reshape(n * self.vote_factor, self.out_dim)
        vote_batch = seed_batch.repeat_interleave(self.vote_factor)
        return vote_pos, vote_x, vote_batch


class VoteNetProposalModule(nn.Module):
    r"""Vote aggregation and proposal generation.

    Mirrors the reference `ProposalModule`: cluster the votes with a single set-abstraction layer
    ([`SAModule`][torch_pointcloud.layers.pointnet2_blocks.SAModule]), then a 3-layer linear head
    decodes objectness, center, heading and size bins/residuals and semantic class per proposal.

    Args:
        num_class: Number of semantic classes.
        num_heading_bin: Number of heading-angle bins.
        num_size_cluster: Number of size templates.
        num_proposal: Number of proposals (= aggregation centroids) per scene.
        sampling: Aggregation-center sampling, `"vote_fps"` or `"seed_fps"`.
        seed_feat_dim: Channel count of the (vote) input features.
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
        num_class: int,
        num_heading_bin: int,
        num_size_cluster: int,
        num_proposal: int,
        sampling: str,
        seed_feat_dim: int,
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
        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.num_proposal = num_proposal
        self.sampling = sampling

        self.vote_aggr = SAModule(
            in_channels=seed_feat_dim,
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
        out_dim = 2 + 3 + num_heading_bin * 2 + num_size_cluster * 4 + num_class
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
        vote_pos: Tensor,
        vote_x: Tensor,
        vote_batch: Tensor,
        seed_pos: Tensor,
        seed_batch: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        if self.sampling == "vote_fps":
            idx = fps(vote_pos, vote_batch, num_nodes=self.num_proposal, random_start=self.training)
        elif self.sampling == "seed_fps":
            # FPS on the seeds; with vote_factor == 1 the vote and seed orderings coincide.
            idx = fps(seed_pos, seed_batch, num_nodes=self.num_proposal, random_start=self.training)
        else:
            raise ValueError(f"Unknown sampling strategy {self.sampling!r}. Expected 'vote_fps' or 'seed_fps'.")

        aggr_x, aggr_pos, aggr_batch = self.vote_aggr(vote_x, vote_pos, vote_batch, idx)

        net = self.mlp(aggr_x)
        return net, aggr_pos, aggr_batch

    def decode(
        self,
        net: Tensor,
        aggr_pos: Tensor,
        aggr_batch: Tensor,
        mean_size_arr: Tensor,
    ) -> Dict[str, Tensor]:
        batch_size = int(aggr_batch.max().item()) + 1 if aggr_batch.numel() else 0
        num_proposal = self.num_proposal
        nh = self.num_heading_bin
        ns = self.num_size_cluster

        net = net.view(batch_size, num_proposal, -1)
        base_pos = aggr_pos.view(batch_size, num_proposal, 3)

        objectness = net[..., 0:2]
        center = base_pos + net[..., 2:5]
        heading_scores = net[..., 5 : 5 + nh]
        heading_res_norm = net[..., 5 + nh : 5 + nh * 2]
        size_scores = net[..., 5 + nh * 2 : 5 + nh * 2 + ns]
        size_res_norm = net[..., 5 + nh * 2 + ns : 5 + nh * 2 + ns * 4].view(batch_size, num_proposal, ns, 3)
        sem_cls_scores = net[..., 5 + nh * 2 + ns * 4 :]

        return {
            "objectness_scores": objectness,
            "center": center,
            "heading_scores": heading_scores,
            "heading_residuals_normalized": heading_res_norm,
            "heading_residuals": heading_res_norm * (math.pi / nh),
            "size_scores": size_scores,
            "size_residuals_normalized": size_res_norm,
            "size_residuals": size_res_norm * mean_size_arr.view(1, 1, ns, 3),
            "sem_cls_scores": sem_cls_scores,
            "aggregated_vote_pos": base_pos,
        }


class VoteNetDetection(DetectionModel):
    r"""VoteNet 3D object detector (packed point format).

    Reference: :arxiv: [Qi et al., 2019](https://arxiv.org/abs/1904.09664).
    Reference implementation: :github:
    [facebookresearch/votenet](https://github.com/facebookresearch/votenet).

    A PointNet++ backbone extracts seed points; a voting module shifts each seed toward its object
    center; a proposal module clusters the votes and decodes a fixed number of oriented (or
    axis-aligned) box proposals per scene. The backbone, voting and proposal hyperparameters are all
    constructor arguments; the registered factories supply the dataset-specific values.

    Args:
        in_channels: Input feature channels per point excluding xyz (e.g. $1$ for a floor-relative
            height feature, $4$ for height + RGB).
        num_classes: Number of semantic classes.
        num_heading_bin: Number of heading-angle bins ($1$ for axis-aligned ScanNet boxes, $12$ for
            oriented SUN RGB-D boxes).
        num_size_cluster: Number of size templates (one per class here).
        mean_size_arr: Per-template mean box size, shape $(\text{num\_size\_cluster}, 3)$.
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

    mean_size_arr: Tensor  # registered as a non-persistent buffer in __init__

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        *,
        num_heading_bin: int,
        num_size_cluster: int,
        mean_size_arr: Union[Tensor, List[List[float]]],
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

        mean = torch.as_tensor(mean_size_arr, dtype=torch.float32)
        if mean.shape != (num_size_cluster, 3):
            raise ValueError(f"`mean_size_arr` must have shape ({num_size_cluster}, 3), got {tuple(mean.shape)}.")
        # Not part of the checkpoint (the reference rebuilds it on the fly); persistent=False
        # keeps it out of the state dict while still moving with `.to(device)`.
        self.register_buffer("mean_size_arr", mean, persistent=False)

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
            num_class=num_classes,
            num_heading_bin=num_heading_bin,
            num_size_cluster=num_size_cluster,
            num_proposal=num_proposal,
            sampling=sampling,
            seed_feat_dim=self.backbone.out_channels,
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
        self.proposal.num_class = num_classes
        out_dim = 2 + 3 + self.num_heading_bin * 2 + self.num_size_cluster * 4 + num_classes
        self.proposal.mlp.lins[-1] = nn.Linear(self.proposal.aggr_dim, out_dim)

    def forward_features(self, x: OptTensor, pos: Tensor, batch: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        assert x is not None, "VoteNet requires input features (got x=None)."
        return self.backbone(x, pos, batch)

    def forward_head(
        self,
        seed_x: Tensor,
        seed_pos: Tensor,
        seed_batch: Tensor,
        seed_inds: Tensor,
    ) -> VoteNetOutput:
        vote_pos, vote_x, vote_batch = self.vgen(seed_pos, seed_x, seed_batch)
        vote_x = vote_x / vote_x.norm(p=2, dim=1, keepdim=True)

        net, aggr_pos, aggr_batch = self.proposal(vote_pos, vote_x, vote_batch, seed_pos, seed_batch)
        decoded = self.proposal.decode(net, aggr_pos, aggr_batch, self.mean_size_arr)

        out: VoteNetOutput = {
            **decoded,  # type: ignore[typeddict-item]
            "seed_pos": seed_pos,
            "seed_batch": seed_batch,
            "seed_inds": seed_inds,
            "vote_pos": vote_pos,
            "vote_batch": vote_batch,
        }
        return out

    def forward(self, x: OptTensor, pos: Tensor, batch: Tensor) -> VoteNetOutput:
        seed_x, seed_pos, seed_batch, seed_inds = self.forward_features(x, pos, batch)
        return self.forward_head(seed_x, seed_pos, seed_batch, seed_inds)


_SCANNET_MEAN_SIZE_ARR: List[List[float]] = [
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

# SUN RGB-D per-class mean box sizes (full edge length), from `model_util_sunrgbd.py`.
_SUNRGBD_MEAN_SIZE_ARR: List[List[float]] = [
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

_VOTENET_BACKBONE_HPARAMS: Dict[str, Any] = dict(
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
)


_VOTENET_SCANNET_TRANSFORMS: Callable[..., Any] = T.Compose(
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
)

_VOTENET_SUNRGBD_TRANSFORMS: Callable[..., Any] = T.Compose(
    [
        T.AxisMinOffset(keys=DataKeys.POS, axis=2, quantile=0.0099, dst_keys="height"),
        T.RandomSample(keys=[DataKeys.POS, "height"], num_samples=20000),
        T.Cat(keys=["height"], dst_key=DataKeys.X, dim=1),
    ]
)


@register_model(
    "votenet-fair-base.scannet",
    task="detection",
    weights="hf://torch-pointcloud/votenet/votenet-fair-base.scannet.pt",
    transforms=_VOTENET_SCANNET_TRANSFORMS,
    hparams=dict(
        _VOTENET_BACKBONE_HPARAMS,
        in_channels=1,
        num_classes=18,
        num_heading_bin=1,
        num_size_cluster=18,
        mean_size_arr=_SCANNET_MEAN_SIZE_ARR,
        sampling="vote_fps",
    ),
)
def votenet_fair_base_scannet(**hparams: Any) -> VoteNetDetection:
    return VoteNetDetection(**hparams)


@register_model(
    "votenet-fair-base.sunrgbd",
    task="detection",
    weights="hf://torch-pointcloud/votenet/votenet-fair-base.sunrgbd.pt",
    transforms=_VOTENET_SUNRGBD_TRANSFORMS,
    hparams=dict(
        _VOTENET_BACKBONE_HPARAMS,
        in_channels=1,
        num_classes=10,
        num_heading_bin=12,
        num_size_cluster=10,
        mean_size_arr=_SUNRGBD_MEAN_SIZE_ARR,
        sampling="seed_fps",
    ),
)
def votenet_fair_base_sunrgbd(**hparams: Any) -> VoteNetDetection:
    return VoteNetDetection(**hparams)
