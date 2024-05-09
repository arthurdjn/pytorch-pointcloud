import math
from typing import Any, Literal, Optional, Tuple, TypedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torch_pointcloud.loss.huber_loss import huber_loss
from torch_pointcloud.ops import fps
from torch_pointcloud.ops.metrics import sided_distance
from torch_pointcloud.utils import to_tensor

from .pointnet2 import PointNetFP, PointNetSA


class VoteNetResult(TypedDict):
    seed_xyz: torch.Tensor
    # seed_features: torch.Tensor
    seed_idxs: torch.Tensor
    vote_xyz: torch.Tensor
    # vote_features: torch.Tensor
    # vote_idxs: torch.Tensor  # Does not exists
    aggregated_vote_xyz: torch.Tensor
    # aggregated_vote_features: torch.Tensor
    aggregated_vote_idxs: torch.Tensor  # TODO:  Not necessary
    objectness_scores: torch.Tensor
    center: torch.Tensor
    heading_scores: torch.Tensor
    heading_residuals_normalized: torch.Tensor
    heading_residuals: torch.Tensor
    size_scores: torch.Tensor
    size_residuals_normalized: torch.Tensor
    size_residuals: torch.Tensor
    sem_cls_scores: torch.Tensor


class VoteNetTarget(TypedDict):
    vote_xyz: torch.Tensor
    vote_label_mask: torch.Tensor
    center_label: Tensor  # List[torch.Tensor]
    heading_class_label: torch.Tensor
    heading_residual_label: torch.Tensor
    size_class_label: torch.Tensor
    size_residual_label: torch.Tensor
    sem_cls_label: torch.Tensor
    box_label_mask: torch.Tensor


class VoteNetLossResult(TypedDict):
    loss: Tensor
    box_loss: Tensor
    vote_loss: Tensor
    objectness_loss: Tensor
    center_loss: Tensor
    heading_cls_loss: Tensor
    heading_reg_loss: Tensor
    size_cls_loss: Tensor
    size_reg_loss: Tensor
    sem_cls_loss: Tensor


class VoteNetBackbone(nn.Module):
    def __init__(self, input_feature_dim: int = 0):
        super().__init__()

        self.sa1 = PointNetSA(
            num_points=2048,
            radius_list=[0.2],
            samples_list=[64],
            channels=[[input_feature_dim + 3, 64, 64, 128]],
            use_pos=True,
            normalize_pos=True,
        )

        self.sa2 = PointNetSA(
            num_points=1024,
            radius_list=[0.4],
            samples_list=[32],
            channels=[[128 + 3, 128, 128, 256]],
            use_pos=True,
            normalize_pos=True,
        )

        self.sa3 = PointNetSA(
            num_points=512,
            radius_list=[0.8],
            samples_list=[16],
            channels=[[256 + 3, 128, 128, 256]],
            use_pos=True,
            normalize_pos=True,
        )

        self.sa4 = PointNetSA(
            num_points=256,
            radius_list=[1.2],
            samples_list=[16],
            channels=[[256 + 3, 128, 128, 256]],
            use_pos=True,
            normalize_pos=True,
        )

        self.fp4 = PointNetFP(channels=[256 + 256, 256, 256])
        self.fp3 = PointNetFP(channels=[256 + 256, 256, 256])

    def forward(self, pos: torch.Tensor, features: Optional[torch.Tensor] = None) -> Any:
        pos1, feats1, sa1_idxs = self.sa1(pos, features)
        pos2, feats2, _ = self.sa2(pos1, feats1)
        pos3, feats3, _ = self.sa3(pos2, feats2)
        pos4, feats4, _ = self.sa4(pos3, feats3)

        _, feats3 = self.fp4(pos4, feats4, pos3, feats3)
        _, feats2 = self.fp3(pos3, feats3, pos2, feats2)

        num_seed = feats2.shape[1]
        return pos2, feats2, sa1_idxs[:, 0:num_seed]


class VotingModule(nn.Module):
    def __init__(self, vote_factor: int, seed_feature_dim: int):
        super().__init__()
        self.vote_factor = vote_factor
        self.in_dim = seed_feature_dim
        self.out_dim = seed_feature_dim

        self.nn = nn.Sequential(
            nn.Sequential(
                nn.Conv1d(self.in_dim, self.in_dim, 1),
                nn.BatchNorm1d(self.in_dim),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Conv1d(self.in_dim, self.in_dim, 1),
                nn.BatchNorm1d(self.in_dim),
                nn.ReLU(),
            ),
            nn.Conv1d(self.in_dim, (3 + self.out_dim) * self.vote_factor, 1),
        )

    def forward(self, seed_xyz: torch.Tensor, seed_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, num_seed, _ = seed_xyz.shape
        num_vote = num_seed * self.vote_factor
        net = self.nn(seed_features)  # (B, (3+out_dim)*vote_factor, num_seed)

        net = net.transpose(2, 1).view(B, num_seed, self.vote_factor, 3 + self.out_dim)
        offset = net[:, :, :, 0:3]
        vote_xyz = seed_xyz.unsqueeze(2) + offset
        vote_xyz = vote_xyz.contiguous().view(B, num_vote, 3)

        residual_features = net[:, :, :, 3:]  # (B, num_seed, vote_factor, out_dim)
        vote_features = seed_features.transpose(2, 1).unsqueeze(2) + residual_features
        vote_features = vote_features.contiguous().view(B, num_vote, self.out_dim)
        vote_features = vote_features.transpose(2, 1).contiguous()

        return vote_xyz, vote_features


class ProposalModule(nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_heading_bin: int,
        num_size_cluster: int,
        num_proposal: int,
        sampling: Literal["vote_fps", "seed_fps", "random"] = "seed_fps",
        seed_feat_dim: int = 256,
    ):
        super().__init__()
        if sampling not in ["vote_fps", "seed_fps", "random"]:
            raise ValueError(
                f"Unknown sampling strategy. Got {sampling!r}, "
                "but expected one of 'vote_fps', 'seed_fps', or 'random'."
            )

        self.num_classes = num_classes
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.num_proposal = num_proposal
        self.sampling = sampling
        self.seed_feat_dim = seed_feat_dim

        self.sa = PointNetSA(
            num_points=self.num_proposal,
            radius_list=[0.3],
            samples_list=[16],
            channels=[[self.seed_feat_dim + 3, 128, 128, 128]],
            use_pos=True,
            normalize_pos=True,
        )

        self.shared_mlp = nn.Sequential(
            nn.Sequential(
                nn.Conv1d(128, 128, 1),
                nn.BatchNorm1d(128),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Conv1d(128, 128, 1),
                nn.BatchNorm1d(128),
                nn.ReLU(),
            ),
            nn.Conv1d(128, 2 + 3 + num_heading_bin * 2 + num_size_cluster * 4 + num_classes, 1),
        )

    def forward(
        self, vote_xyz: torch.Tensor, vote_features: torch.Tensor, seed_xyz: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.sampling == "vote_fps":
            xyz, features, idxs = self.sa(vote_xyz, vote_features)
        elif self.sampling == "seed_fps":
            assert seed_xyz is not None, "seed_xyz must be provided when using seed_fps sampling."
            idxs = fps(seed_xyz, num_samples=self.num_proposal)
            xyz, features, idxs = self.sa(vote_xyz, vote_features, idxs)
        else:
            assert seed_xyz is not None, "seed_xyz must be provided when using random sampling."
            B, num_seed, *_ = seed_xyz.shape
            idxs = torch.randint(0, num_seed, (B, self.num_proposal), dtype=torch.int).to(vote_xyz.device)
            features, xyz, _ = self.sa(vote_xyz, vote_features, idxs)

        return xyz, features, idxs


class VoteNet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_heading_bin: int,
        num_size_cluster: int,
        mean_size_arr: torch.Tensor,
        input_feature_dim: int = 0,
        num_proposal: int = 128,
        vote_factor: int = 1,
        sampling: Literal["vote_fps", "seed_fps", "random"] = "vote_fps",
    ):
        super().__init__()
        if mean_size_arr.shape[0] != num_size_cluster:
            raise ValueError(
                f"Expected mean_size_arr to have shape (num_classes, 3), but got {mean_size_arr.shape!r} instead."
            )

        self.num_classes = num_classes
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        self.input_feature_dim = input_feature_dim
        self.num_proposal = num_proposal
        self.vote_factor = vote_factor
        self.sampling = sampling

        self.backbone = VoteNetBackbone(input_feature_dim=self.input_feature_dim)
        self.vnet = VotingModule(vote_factor=self.vote_factor, seed_feature_dim=256)
        self.pnet = ProposalModule(
            num_classes=num_classes,
            num_heading_bin=num_heading_bin,
            num_size_cluster=num_size_cluster,
            num_proposal=num_proposal,
            sampling=sampling,
        )
        self.head = nn.Sequential(
            nn.Sequential(
                nn.Conv1d(128, 128, 1),
                nn.BatchNorm1d(128),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Conv1d(128, 128, 1),
                nn.BatchNorm1d(128),
                nn.ReLU(),
            ),
            nn.Conv1d(128, 2 + 3 + num_heading_bin * 2 + num_size_cluster * 4 + num_classes, 1),
        )

    def forward(self, pos: torch.Tensor, features: Optional[torch.Tensor] = None) -> VoteNetResult:
        seed_xyz, seed_features, seed_idxs = self.backbone(pos, features)

        vote_xyz, vote_features = self.vnet(seed_xyz, seed_features)
        assert vote_features is not None  # for type checking

        vote_features_norm = torch.norm(vote_features, p=2, dim=1)
        vote_features = vote_features / vote_features_norm.unsqueeze(1)
        aggregated_vote_xyz, aggregated_vote_features, aggregated_vote_idxs = self.pnet(vote_xyz, vote_features)
        preds = self.head(vote_features)

        # process the results
        # fmt: off
        preds = preds.transpose(2, 1)
        objectness_scores = preds[:, :, :2]
        center = preds[:, :, 2:5]
        heading_scores = preds[:, :, 5 : 5 + self.num_heading_bin]
        heading_residuals_normalized = preds[:, :, 5 + self.num_heading_bin : 5 + self.num_heading_bin * 2]
        heading_residuals = heading_residuals_normalized * math.pi / self.num_heading_bin
        size_scores = preds[:, :, 5 + self.num_heading_bin * 2 : 5 + self.num_heading_bin * 2 + self.num_size_cluster]
        size_residuals_normalized = preds[:, :, 5 + self.num_heading_bin * 2 + self.num_size_cluster:5 + self.num_heading_bin * 2 + self.num_size_cluster * 4].view(-1, self.num_proposal, self.num_size_cluster, 3)
        size_residuals = size_residuals_normalized * self.mean_size_arr.unsqueeze(0).unsqueeze(0)
        sem_cls_scores = preds[:, :, 5 + self.num_heading_bin * 2 + self.num_size_cluster * 4 :]
        # fmt: on

        # ? Is it necessary to return the intermediate features? They are not used in the loss computation
        # ? Maybe take inspiration from timm, with something like `forward_features` and `forward_head`
        return VoteNetResult(
            seed_xyz=seed_xyz,
            seed_idxs=seed_idxs,
            vote_xyz=vote_xyz,
            aggregated_vote_xyz=aggregated_vote_xyz,
            aggregated_vote_idxs=aggregated_vote_idxs,
            objectness_scores=objectness_scores,
            center=center,
            heading_scores=heading_scores,
            heading_residuals_normalized=heading_residuals_normalized,
            heading_residuals=heading_residuals,
            size_scores=size_scores,
            size_residuals_normalized=size_residuals_normalized,
            size_residuals=size_residuals,
            sem_cls_scores=sem_cls_scores,
        )


class VoteNetLoss(nn.Module):
    def __init__(
        self,
        num_heading_bin: int,
        num_size_cluster: int,
        mean_size_arr: torch.Tensor,
        vote_factor: int = 3,
        objectness_weights: Optional[torch.Tensor] = None,
        threshold_near: float = 0.3,
        threshold_far: float = 0.6,
    ) -> None:
        super().__init__()
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = to_tensor(mean_size_arr)
        self.vote_factor = vote_factor
        self.objectness_weights = objectness_weights if objectness_weights is not None else torch.tensor([0.2, 0.8])
        self.threshold_near = threshold_near
        self.threshold_far = threshold_far
        self.mean_sizes = [0]

    def forward(self, result: VoteNetResult, target: VoteNetTarget) -> VoteNetLossResult:
        # TODO: Maybe unpack the result and target there, and compute heading and size residuals there

        vote_loss = self._compute_vote_loss(
            seed_xyz=result["seed_xyz"],
            seed_idxs=result["seed_idxs"],
            vote_xyz=result["vote_xyz"],
            vote_xyz_target=target["vote_xyz"],
            vote_label_mask=target["vote_label_mask"],
        )

        objectness_loss, objectness_target, objectness_mask, object_assignment = self._compute_objectness_loss(
            aggregated_vote_xyz=result["aggregated_vote_xyz"],
            center_label=target["center_label"],
            objectness_scores=result["objectness_scores"],
        )

        center_loss = self._compute_center_loss(
            center=result["center"],
            center_label=target["center_label"],
            objectness_target=objectness_target,
            box_label_mask=target["box_label_mask"],
        )

        heading_cls_loss = self._compute_heading_cls_loss(
            object_assignment=object_assignment,
            objectness_target=objectness_target,
            heading_scores=result["heading_scores"],
            heading_class_label=target["heading_class_label"],
        )

        heading_reg_loss = self._compute_heading_reg_loss(
            object_assignment=object_assignment,
            objectness_label=objectness_target,
            heading_residuals_normalized=result["heading_residuals_normalized"],
            heading_class_label=target["heading_class_label"],
            heading_residual_label=target["heading_residual_label"],
        )

        size_cls_loss = self._compute_size_cls_loss(
            object_assignment=object_assignment,
            objectness_label=objectness_target,
            size_scores=result["size_scores"],
            size_class_label=target["size_class_label"],
        )

        size_reg_loss = self._compute_size_reg_loss(
            object_assignment=object_assignment,
            objectness_label=objectness_target,
            size_class_label=target["size_class_label"],
            size_residuals_normalized=result["size_residuals_normalized"],
            size_residual_label=target["size_residual_label"],
        )

        sem_cls_loss = self._compute_sem_cls_loss(
            object_assignment=object_assignment,
            objectness_label=objectness_target,
            sem_cls_scores=result["sem_cls_scores"],
            sem_cls_label=target["sem_cls_label"],
        )

        box_loss = center_loss + 0.1 * heading_cls_loss + heading_reg_loss + 0.1 * size_cls_loss + size_reg_loss
        loss = vote_loss + 0.5 * objectness_loss + box_loss + 0.1 * sem_cls_loss

        return {
            "loss": loss,
            "box_loss": box_loss,
            "vote_loss": vote_loss,
            "objectness_loss": objectness_loss,
            "center_loss": center_loss,
            "heading_cls_loss": heading_cls_loss,
            "heading_reg_loss": heading_reg_loss,
            "size_cls_loss": size_cls_loss,
            "size_reg_loss": size_reg_loss,
            "sem_cls_loss": sem_cls_loss,
        }

    # NOTE: There are differences between the original implementation and this one
    # This refactor uses L2 distance instead of L1 distance
    # TODO: Support L1 distance as an option (implement it in CUDA)
    def _compute_vote_loss(
        self,
        seed_xyz: Tensor,
        seed_idxs: Tensor,
        vote_xyz: Tensor,
        vote_xyz_target: Tensor,
        vote_label_mask: Tensor,
    ) -> torch.Tensor:
        B, num_seed, _ = seed_xyz.shape

        # Get associated ground truth votes positions associated to the seed points
        seed_mask = torch.gather(vote_label_mask, 1, seed_idxs[:, :, 0])  # (B, num_seed)
        seed_idxs_expand = seed_idxs.view(B, num_seed, -1).repeat(1, 1, self.vote_factor)

        seed_vote_xyz_target = torch.gather(vote_xyz_target, 1, seed_idxs_expand)
        seed_vote_xyz_target += seed_xyz.repeat(1, 1, self.vote_factor)

        # Compute distances between each seed vote and all GT votes
        # NOTE: vote_xyz: (B, num_seed, 3)
        #       seed_vote_xyz_target: (B, num_seed, 3 * vote_factor)
        #       To be able to compute the distance, they must have a similar channel dimension:
        #       -> vote_xyz: (B * num_seed, 1, 3)
        #       -> seed_vote_xyz_target: (B * num_seed, vote_factor, 3)
        vote_xyz = vote_xyz.view(B * num_seed, -1, 3)
        seed_vote_xyz_target = seed_vote_xyz_target.view(B * num_seed, self.vote_factor, 3)

        # A predicted vote to no is not penalized as long as there is a good vote near the GT vote.
        dists, _ = sided_distance(vote_xyz, seed_vote_xyz_target)
        dists = torch.sqrt(dists)  # (B * num_seed, vote_factor)
        vote_dists, _ = torch.min(dists, dim=1)  # (B * num_seed,)
        vote_dists = vote_dists.view(B, num_seed)  # (B, num_seed)
        return torch.sum(vote_dists * seed_mask) / (seed_mask.sum() + 1e-6)

    def _compute_objectness_loss(
        self,
        aggregated_vote_xyz: Tensor,
        center_label: Tensor,
        objectness_scores: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        device = aggregated_vote_xyz.device
        B, num_proposal, _ = aggregated_vote_xyz.shape

        # Associate proposal and GT objects by point-to-point distances
        dists, idxs = sided_distance(aggregated_vote_xyz, center_label[:, :, :3])  # (B, num_proposal)
        dists = torch.sqrt(dists)

        # Generate objectness label and mask
        # objectness_label: 1 if pred object center is within `threshold_near` of any GT object
        # objectness_mask: 0 if pred object center is in gray zone (i.e. do not care), 1 otherwise
        objectness_target = torch.zeros((B, num_proposal), dtype=torch.long).to(device)
        objectness_mask = torch.zeros((B, num_proposal)).to(device)
        objectness_target[dists < self.threshold_near] = 1
        objectness_mask[dists < self.threshold_near] = 1
        objectness_mask[dists > self.threshold_far] = 1

        # Compute objectness loss
        objectness_loss = F.cross_entropy(
            objectness_scores.transpose(2, 1),  # (B, 2, num_proposal)
            objectness_target,  # (B, num_proposal)
            weight=self.objectness_weights.to(device),
            reduction="none",
        )

        # Compute the average objectness loss, only consider actual objects
        objectness_loss = torch.sum(objectness_loss * objectness_mask) / (torch.sum(objectness_mask) + 1e-6)
        return objectness_loss, objectness_target, objectness_mask, idxs

    # ! TODO: For testing purposes, remove the padding implementation and use the original implementation
    # TODO: Small difference in centroid_reg_loss1 from original implementation
    # TODO: Add utility function to pad, mask and get lengths of a list of tensors instead of `torch.nn.utils.rnn.pad_sequence`
    def _compute_center_loss(
        self,
        center: Tensor,
        center_label: Tensor,
        objectness_target: Tensor,
        box_label_mask: Tensor,  # Optional if not using padding (center_label = list of tensors)
    ) -> torch.Tensor:
        center_label = center_label[:, :, :3]

        # TODO: Create utility to pad & mask tensors
        # center_label_padded = torch.nn.utils.rnn.pad_sequence(center_label, batch_first=True, padding_value=0)
        # center_target_lengths = torch.tensor([len(c) for c in center_label]).to(center_label_padded.device)
        # center_target_mask = torch.zeros_like(center_label_padded)
        # for b, length in enumerate(center_target_lengths):
        #     center_target_mask[b, :length] = 1

        # center_target_lengths = box_label_mask.sum(dim=1)
        dists1, _ = sided_distance(center, center_label)
        dists2, _ = sided_distance(center_label, center)

        centroid_reg_loss1 = torch.sum(dists1 * objectness_target) / (objectness_target.sum() + 1e-6)
        centroid_reg_loss2 = torch.sum(dists2 * box_label_mask) / (box_label_mask.sum() + 1e-6)
        return centroid_reg_loss1 + centroid_reg_loss2

    def _compute_heading_cls_loss(
        self,
        object_assignment: Tensor,
        objectness_target: Tensor,
        heading_scores: Tensor,
        heading_class_label: Tensor,
    ) -> torch.Tensor:
        heading_class_label = torch.gather(heading_class_label, 1, object_assignment)
        loss = F.cross_entropy(heading_scores.transpose(2, 1), heading_class_label, reduction="none")
        return torch.sum(loss * objectness_target) / (torch.sum(objectness_target) + 1e-6)

    def _compute_heading_reg_loss(
        self,
        object_assignment: Tensor,
        objectness_label: Tensor,
        heading_residuals_normalized: Tensor,
        heading_class_label: Tensor,
        heading_residual_label: Tensor,
    ) -> torch.Tensor:
        B, K, *_ = object_assignment.shape

        heading_class_label = torch.gather(heading_class_label, 1, object_assignment)
        heading_residual_label = torch.gather(heading_residual_label, 1, object_assignment)
        heading_residual_normalized_label = heading_residual_label / (math.pi / self.num_heading_bin)

        heading_label_one_hot = torch.zeros((B, K, self.num_heading_bin)).to(heading_class_label.device)
        heading_label_one_hot.scatter_(2, heading_class_label.unsqueeze(-1), 1)

        errors = torch.sum(heading_residuals_normalized * heading_label_one_hot, -1) - heading_residual_normalized_label
        heading_residual_normalized_loss = huber_loss(errors, delta=1.0)
        return torch.sum(heading_residual_normalized_loss * objectness_label) / (objectness_label.sum() + 1e-6)

    def _compute_size_cls_loss(
        self,
        object_assignment: Tensor,
        objectness_label: Tensor,
        size_scores: Tensor,
        size_class_label: Tensor,
    ) -> torch.Tensor:
        size_class_label = torch.gather(size_class_label, 1, object_assignment)
        size_class_loss = F.cross_entropy(size_scores.transpose(2, 1), size_class_label.long(), reduction="none")
        return torch.sum(size_class_loss * objectness_label) / (torch.sum(objectness_label) + 1e-6)

    def _compute_size_reg_loss(
        self,
        object_assignment: Tensor,
        objectness_label: Tensor,
        size_class_label: Tensor,
        size_residuals_normalized: Tensor,
        size_residual_label: Tensor,
    ) -> torch.Tensor:
        B, K, *_ = object_assignment.shape

        size_class_label = torch.gather(size_class_label, 1, object_assignment)  # select (B, K1) from (B, K2)
        size_residual_label = torch.gather(size_residual_label, 1, object_assignment.unsqueeze(-1).repeat(1, 1, 3))

        size_label_one_hot = torch.zeros((B, K, self.num_size_cluster)).to(size_class_label.device)
        size_label_one_hot.scatter_(2, size_class_label.unsqueeze(-1).long(), 1)

        size_label_one_hot_tiled = size_label_one_hot.unsqueeze(-1).repeat(1, 1, 1, 3)  # (B, K, num_size_cluster, 3)
        predicted_size_residual_normalized = torch.sum(size_residuals_normalized * size_label_one_hot_tiled, dim=2)

        mean_size_arr_expanded = self.mean_size_arr.unsqueeze(0).unsqueeze(0).to(size_label_one_hot_tiled.device)
        mean_size_label = torch.sum(size_label_one_hot_tiled * mean_size_arr_expanded, 2)  # (B, K, 3)
        size_residual_label_normalized = size_residual_label / mean_size_label  # (B, K, 3)
        size_residual_normalized_loss = torch.mean(
            huber_loss(predicted_size_residual_normalized - size_residual_label_normalized, delta=1.0), -1
        )  # (B, K, 3) -> (B, K)
        return torch.sum(size_residual_normalized_loss * objectness_label) / (torch.sum(objectness_label) + 1e-6)

    def _compute_sem_cls_loss(
        self,
        object_assignment: Tensor,
        objectness_label: Tensor,
        sem_cls_scores: Tensor,
        sem_cls_label: Tensor,
    ) -> torch.Tensor:
        sem_cls_label = torch.gather(sem_cls_label, 1, object_assignment)
        sem_cls_loss = F.cross_entropy(sem_cls_scores.transpose(2, 1), sem_cls_label.long(), reduction="none")  # (B, K)
        return torch.sum(sem_cls_loss * objectness_label) / (torch.sum(objectness_label) + 1e-6)
