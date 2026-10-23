r"""GeoTransformer registration loss: an overlap-aware circle loss on the superpoint features and a negative
log-likelihood on the patch optimal-transport assignments."""

from typing import TYPE_CHECKING, Any, Dict

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from torch_pointcloud.ops.geometry import transform_points
from torch_pointcloud.ops.registration import pairwise_squared_distance
from torch_pointcloud.utils.data import DataKeys

if TYPE_CHECKING:
    from torch_pointcloud.models.geotransformer import GeoTransformerOutput


def weighted_circle_loss(
    pos_masks: Tensor,
    neg_masks: Tensor,
    feat_dists: Tensor,
    pos_scales: Tensor,
    positive_margin: float = 0.1,
    negative_margin: float = 1.4,
    positive_optimal: float = 0.1,
    negative_optimal: float = 1.4,
    log_scale: float = 24.0,
) -> Tensor:
    r"""Circle loss over a distance matrix, with per-pair weights on the positive terms.

    The positives of a row are pushed under `positive_margin` and the negatives beyond `negative_margin`, with
    the self-paced weights of the circle loss; the row and column losses are averaged over the rows and columns
    that have both a positive and a negative.

    Args:
        pos_masks: Positive pairs, shape $(N, M)$.
        neg_masks: Negative pairs, shape $(N, M)$.
        feat_dists: Feature distances, shape $(N, M)$.
        pos_scales: Weight of every positive pair, shape $(N, M)$.
        positive_margin: Distance under which a positive pair stops being penalized.
        negative_margin: Distance beyond which a negative pair stops being penalized.
        positive_optimal: Target distance of the positives in the self-paced weights.
        negative_optimal: Target distance of the negatives in the self-paced weights.
        log_scale: Temperature of the log-sum-exp.

    Returns:
        The scalar loss.

    Example:
        ```pycon
        >>> import torch
        >>> from torch_pointcloud.losses import weighted_circle_loss
        >>> dists = torch.tensor([[0.2, 1.0], [1.2, 0.3]])
        >>> pos = torch.eye(2, dtype=torch.bool)
        >>> loss = weighted_circle_loss(pos, ~pos, dists, pos.float())
        >>> loss.ndim
        0

        ```
    """
    pos_weights = feat_dists - 1e5 * (~pos_masks).float()  # mask the non-positives
    pos_weights = pos_weights - positive_optimal
    pos_weights = torch.maximum(torch.zeros_like(pos_weights), pos_weights)
    pos_weights = pos_scales * pos_weights.detach()

    neg_weights = feat_dists + 1e5 * (~neg_masks).float()  # mask the non-negatives
    neg_weights = negative_optimal - neg_weights
    neg_weights = torch.maximum(torch.zeros_like(neg_weights), neg_weights).detach()

    loss_pos_row = torch.logsumexp(log_scale * (feat_dists - positive_margin) * pos_weights, dim=-1)
    loss_pos_col = torch.logsumexp(log_scale * (feat_dists - positive_margin) * pos_weights, dim=-2)
    loss_neg_row = torch.logsumexp(log_scale * (negative_margin - feat_dists) * neg_weights, dim=-1)
    loss_neg_col = torch.logsumexp(log_scale * (negative_margin - feat_dists) * neg_weights, dim=-2)

    loss_row = F.softplus(loss_pos_row + loss_neg_row) / log_scale
    loss_col = F.softplus(loss_pos_col + loss_neg_col) / log_scale

    row_masks = pos_masks.any(dim=-1) & neg_masks.any(dim=-1)
    col_masks = pos_masks.any(dim=-2) & neg_masks.any(dim=-2)
    zero = feat_dists.sum() * 0.0
    row_loss = loss_row[row_masks].mean() if bool(row_masks.any()) else zero
    col_loss = loss_col[col_masks].mean() if bool(col_masks.any()) else zero
    return (row_loss + col_loss) / 2


class GeoTransformerLoss(nn.Module):
    r"""Training criterion of `GeoTransformerRegistration`.

    - **Coarse matching:** the `weighted_circle_loss` between the unit-norm superpoint features of the two clouds,
      where a superpoint pair is positive when its ground-truth patch overlap exceeds `positive_overlap`, weighted
      by the square root of that overlap, and negative when the patches do not overlap at all.
    - **Fine matching:** the negative mean log assignment of the optimal-transport scores at the ground-truth
      entries of every patch pair: point pairs closer than `positive_radius` under the ground-truth transform, and
      the dustbin for the points of a patch without a match.

    The forward reads the ground-truth superpoint correspondences of the model output, so the model must have
    received the ground-truth transform, and the transform itself from `data[DataKeys.TRANSFORM]`, shape
    $(B, 4, 4)$.

    Args:
        positive_margin: Circle loss positive margin.
        negative_margin: Circle loss negative margin.
        positive_optimal: Circle loss positive target distance.
        negative_optimal: Circle loss negative target distance.
        log_scale: Circle loss temperature.
        positive_overlap: Minimum ground-truth patch overlap of a positive superpoint pair.
        positive_radius: Distance under which two patch points are a ground-truth match.
        weight_coarse: Weight of the coarse matching loss.
        weight_fine: Weight of the fine matching loss.
    """

    def __init__(
        self,
        positive_margin: float = 0.1,
        negative_margin: float = 1.4,
        positive_optimal: float = 0.1,
        negative_optimal: float = 1.4,
        log_scale: float = 24.0,
        positive_overlap: float = 0.1,
        positive_radius: float = 0.05,
        weight_coarse: float = 1.0,
        weight_fine: float = 1.0,
    ) -> None:
        super().__init__()
        self.positive_margin = positive_margin
        self.negative_margin = negative_margin
        self.positive_optimal = positive_optimal
        self.negative_optimal = negative_optimal
        self.log_scale = log_scale
        self.positive_overlap = positive_overlap
        self.positive_radius = positive_radius
        self.weight_coarse = weight_coarse
        self.weight_fine = weight_fine

    def coarse_loss(self, output: "GeoTransformerOutput") -> Tensor:
        gt_batch = output["gt_node_corr_batch"]
        batch_size = int(output["ref_batch_c"].max()) + 1
        losses = []
        for b in range(batch_size):
            ref_feats = output["ref_feats_c"][output["ref_batch_c"] == b]
            src_feats = output["src_feats_c"][output["src_batch_c"] == b]
            gt_indices = output["gt_node_corr_indices"][gt_batch == b]
            gt_overlaps = output["gt_node_corr_overlaps"][gt_batch == b]

            feat_dists = torch.sqrt(pairwise_squared_distance(ref_feats, src_feats, normalized=True))
            overlaps = torch.zeros_like(feat_dists)
            overlaps[gt_indices[:, 0], gt_indices[:, 1]] = gt_overlaps
            pos_masks = overlaps > self.positive_overlap
            neg_masks = overlaps == 0
            pos_scales = torch.sqrt(overlaps * pos_masks.float())
            losses.append(
                weighted_circle_loss(
                    pos_masks,
                    neg_masks,
                    feat_dists,
                    pos_scales,
                    positive_margin=self.positive_margin,
                    negative_margin=self.negative_margin,
                    positive_optimal=self.positive_optimal,
                    negative_optimal=self.negative_optimal,
                    log_scale=self.log_scale,
                )
            )
        return torch.stack(losses).mean()

    def fine_loss(self, output: "GeoTransformerOutput", transform: Tensor) -> Tensor:
        ref_points = output["ref_patch_points"]
        src_points = transform_points(output["src_patch_points"], transform[output["node_corr_batch"]])
        ref_masks, src_masks = output["ref_patch_masks"], output["src_patch_masks"]
        matching_scores = output["matching_scores"]

        dists = pairwise_squared_distance(ref_points, src_points)  # (P, K, K)
        gt_masks = ref_masks.unsqueeze(2) & src_masks.unsqueeze(1)
        gt_corr_map = (dists < self.positive_radius**2) & gt_masks
        slack_row_labels = (gt_corr_map.sum(2) == 0) & ref_masks
        slack_col_labels = (gt_corr_map.sum(1) == 0) & src_masks

        labels = torch.zeros_like(matching_scores, dtype=torch.bool)
        labels[:, :-1, :-1] = gt_corr_map
        labels[:, :-1, -1] = slack_row_labels
        labels[:, -1, :-1] = slack_col_labels
        if not bool(labels.any()):
            return matching_scores.sum() * 0.0
        return -matching_scores[labels].mean()

    def forward(self, output: "GeoTransformerOutput", data: Dict[str, Any]) -> Dict[str, Tensor]:
        r"""Compute the loss of a batch.

        Args:
            output: The model output, with the ground-truth superpoint correspondences.
            data: The batch, holding the ground-truth transforms under `DataKeys.TRANSFORM`, shape $(B, 4, 4)$.

        Returns:
            A dict with the total `loss` and its detached `coarse_loss` and `fine_loss` terms.
        """
        coarse = self.coarse_loss(output)
        fine = self.fine_loss(output, data[DataKeys.TRANSFORM])
        loss = self.weight_coarse * coarse + self.weight_fine * fine
        return {"loss": loss, "coarse_loss": coarse.detach(), "fine_loss": fine.detach()}
