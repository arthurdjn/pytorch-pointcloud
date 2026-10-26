r"""GeoTransformer registration loss: an overlap-aware circle loss on the superpoint features and a negative
log-likelihood on the patch optimal-transport assignments."""

from typing import TYPE_CHECKING, Any, Dict, Optional

import torch
from torch import Tensor, nn

from torch_pointcloud.ops.geometry import transform_points
from torch_pointcloud.ops.registration import pairwise_squared_distance
from torch_pointcloud.utils.data import DataKeys

from .circle import weighted_circle_loss

if TYPE_CHECKING:
    from torch_pointcloud.models.geotransformer import GeoTransformerOutput


def assignment_negative_log_likelihood(
    log_assignment: Tensor,
    row_points: Tensor,
    col_points: Tensor,
    radius: float,
    row_mask: Optional[Tensor] = None,
    col_mask: Optional[Tensor] = None,
) -> Tensor:
    r"""Negative mean log-likelihood of the ground-truth matches in assignments with a dustbin row and column.

    A row point and a column point match when they are closer than `radius`; a valid point without any match is
    assigned to the dustbin. The points are given in a common frame, the column points moved by the ground truth.

    Args:
        log_assignment: Log assignments, the last row and column being the dustbins, shape $(B, N + 1, M + 1)$.
        row_points: Points of the rows, shape $(B, N, 3)$.
        col_points: Points of the columns, shape $(B, M, 3)$.
        radius: Distance under which two points match.
        row_mask: Valid rows, shape $(B, N)$; every row when `None`.
        col_mask: Valid columns, shape $(B, M)$; every column when `None`.

    Returns:
        The scalar loss, zero when no entry is labeled.

    Example:
        ```pycon
        >>> import torch
        >>> from torch_pointcloud.losses.geotransformer import assignment_negative_log_likelihood
        >>> points = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
        >>> log_assignment = torch.log(torch.full((1, 3, 3), 0.5))
        >>> round(assignment_negative_log_likelihood(log_assignment, points, points, radius=0.1).item(), 4)
        0.6931

        ```
    """
    if row_mask is None:
        row_mask = row_points.new_ones(row_points.shape[:2], dtype=torch.bool)
    if col_mask is None:
        col_mask = col_points.new_ones(col_points.shape[:2], dtype=torch.bool)

    distances = pairwise_squared_distance(row_points, col_points)  # (B, N, M)
    matches = (distances < radius**2) & row_mask.unsqueeze(2) & col_mask.unsqueeze(1)
    labels = torch.zeros_like(log_assignment, dtype=torch.bool)
    labels[:, :-1, :-1] = matches
    labels[:, :-1, -1] = (matches.sum(2) == 0) & row_mask
    labels[:, -1, :-1] = (matches.sum(1) == 0) & col_mask
    if not bool(labels.any()):
        return log_assignment.sum() * 0.0
    return -log_assignment[labels].mean()


class GeoTransformerLoss(nn.Module):
    r"""Training criterion of `GeoTransformerRegistration`.

    - **Coarse matching:** the `weighted_circle_loss` between the unit-norm superpoint features of the two clouds,
      where a superpoint pair is positive when its ground-truth patch overlap exceeds `positive_overlap`, weighted
      by the square root of that overlap, and negative when the patches do not overlap at all.
    - **Fine matching:** the `assignment_negative_log_likelihood` of the optimal-transport scores of every patch
      pair: point pairs closer than `positive_radius` under the ground-truth transform, and the dustbin for the
      points of a patch without a match.

    The forward reads the ground-truth superpoint correspondences of the model output, so the model must have
    received the ground-truth transform, and the transform itself from `data[DataKeys.TRANSFORM]`, shape
    $(B, 4, 4)$.

    Args:
        positive_radius: Distance under which two patch points are a ground-truth match, in the unit of the
            coordinates.
        positive_margin: Circle loss positive margin.
        negative_margin: Circle loss negative margin.
        positive_optimal: Circle loss positive target distance.
        negative_optimal: Circle loss negative target distance.
        log_scale: Circle loss temperature.
        positive_overlap: Minimum ground-truth patch overlap of a positive superpoint pair.
        weight_coarse: Weight of the coarse matching loss.
        weight_fine: Weight of the fine matching loss.
    """

    def __init__(
        self,
        positive_radius: float,
        positive_margin: float = 0.1,
        negative_margin: float = 1.4,
        positive_optimal: float = 0.1,
        negative_optimal: float = 1.4,
        log_scale: float = 24.0,
        positive_overlap: float = 0.1,
        weight_coarse: float = 1.0,
        weight_fine: float = 1.0,
    ) -> None:
        super().__init__()
        self.positive_radius = positive_radius
        self.positive_margin = positive_margin
        self.negative_margin = negative_margin
        self.positive_optimal = positive_optimal
        self.negative_optimal = negative_optimal
        self.log_scale = log_scale
        self.positive_overlap = positive_overlap
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

            distances = torch.sqrt(pairwise_squared_distance(ref_feats, src_feats, normalized=True))
            overlaps = torch.zeros_like(distances)
            overlaps[gt_indices[:, 1], gt_indices[:, 0]] = gt_overlaps
            positive_mask = overlaps > self.positive_overlap
            losses.append(
                weighted_circle_loss(
                    distances,
                    positive_mask,
                    overlaps == 0,
                    torch.sqrt(overlaps * positive_mask.float()),
                    self.positive_margin,
                    self.negative_margin,
                    self.positive_optimal,
                    self.negative_optimal,
                    self.log_scale,
                )
            )
        return torch.stack(losses).mean()

    def fine_loss(self, output: "GeoTransformerOutput", transform: Tensor) -> Tensor:
        src_points = transform_points(output["src_patch_points"], transform[output["node_corr_batch"]])
        return assignment_negative_log_likelihood(
            output["matching_scores"],
            output["ref_patch_points"],
            src_points,
            self.positive_radius,
            row_mask=output["ref_patch_masks"],
            col_mask=output["src_patch_masks"],
        )

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
