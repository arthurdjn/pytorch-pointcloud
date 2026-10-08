r"""Loss of the query-based `TransFusionHead` (LION): Hungarian-matched query targets and a dense center heatmap."""

from typing import TYPE_CHECKING, Any, Dict, List, Sequence, Tuple

import torch
from torch import Tensor, nn

from torch_pointcloud.losses._utils import unbatch_boxes
from torch_pointcloud.losses.focal import gaussian_focal_loss, sigmoid_focal_loss
from torch_pointcloud.losses.matching import hungarian_match_batched
from torch_pointcloud.ops.box3d import boxes_iou3d
from torch_pointcloud.ops.heatmap import draw_heatmap_targets

if TYPE_CHECKING:
    from torch_pointcloud.models.lion import TransFusionHeadOutput

_LOG_EPS = 1e-12


class TransFusionHeadLoss(nn.Module):
    r"""Loss of the `TransFusionHead` (dense heatmap, matched classification, box, IoU rescore).

    Reference: :arxiv: [TransFusion: Robust LiDAR-Camera Fusion for 3D Object Detection with Transformers](https://arxiv.org/abs/2203.11496) (Bai et al., 2022).

    The head predicts a dense per-class BEV heatmap plus a fixed set of object queries, each carrying a
    class logit vector and a box code. Four terms are summed:

    - **Heatmap:** the ground-truth box centers are splatted onto a per-class BEV Gaussian map and the
      dense heatmap is supervised by the penalty-reduced center focal loss.
    - **Classification:** every scene's queries are decoded to boxes and matched to the ground truth by a
      per-scene Hungarian assignment (cost: focal classification + normalized center $L_1$ + 3D IoU). The
      per-query class logits are then trained by sigmoid focal loss over one-hot targets (background for
      unmatched queries), normalized by the positive count.
    - **Box regression:** code-weighted $L_1$ over the $10$-dim box code
      $(x, y, z + d_z / 2, \log d_x, \log d_y, \log d_z, \sin\theta, \cos\theta, v_x, v_y)$ at the
      matched queries.
    - **IoU rescore:** an $L_1$ term regressing the per-query `iou` branch toward $2 \cdot \text{IoU}_{3D} - 1$
      between each matched query's decoded box and its ground-truth box.

    The loss holds no reference to the model: the grid geometry is rebuilt from the constructor params.

    !!! note
        nuScenes ground-truth boxes carry no velocity ($(K, 7)$), so the velocity targets are zero. The
        default `code_weights` zeroes the last two (velocity) codes to leave that branch unsupervised.

    Args:
        num_classes: Number of foreground classes (heatmap channels and query logits).
        point_cloud_range: Range $(x_\min, y_\min, z_\min, x_\max, y_\max, z_\max)$.
        voxel_size: Voxel size $(v_x, v_y, v_z)$.
        feature_map_stride: Stride from the voxel grid to the BEV feature map.
        num_proposals: Number of object queries per scene.
        gaussian_overlap: Min-overlap passed to the Gaussian-radius solver.
        min_radius: Lower clamp on the integer splat radius.
        matcher_cls_cost: Weight of the focal classification term in the matching cost.
        matcher_reg_cost: Weight of the normalized center-$L_1$ term in the matching cost.
        matcher_iou_cost: Weight of the 3D-IoU term in the matching cost.
        code_weights: Per-code regression weight, length $10$; the last two (velocity) default to $0$.
        cls_weight: Multiplier on the classification term.
        loc_weight: Multiplier on the box-regression term.
        heatmap_weight: Multiplier on the heatmap term.
        iou_weight: Multiplier on the IoU-rescore term.
        focal_alpha: Focal positive/negative balance (classification loss and matching cost).
        focal_gamma: Focal focusing exponent (classification loss and matching cost).
    """

    code_weights: Tensor

    def __init__(
        self,
        num_classes: int,
        *,
        point_cloud_range: Sequence[float],
        voxel_size: Sequence[float],
        feature_map_stride: int,
        num_proposals: int = 200,
        gaussian_overlap: float = 0.1,
        min_radius: int = 2,
        matcher_cls_cost: float = 0.15,
        matcher_reg_cost: float = 0.25,
        matcher_iou_cost: float = 0.25,
        code_weights: Sequence[float] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0),
        cls_weight: float = 1.0,
        loc_weight: float = 0.25,
        heatmap_weight: float = 1.0,
        iou_weight: float = 0.5,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ) -> None:
        super().__init__()
        if len(code_weights) != 10:
            raise ValueError(f"`code_weights` must have length 10, got {len(code_weights)}.")

        self.num_classes = num_classes
        self.point_cloud_range = tuple(float(p) for p in point_cloud_range)
        self.voxel_size = tuple(float(v) for v in voxel_size)
        self.feature_map_stride = feature_map_stride
        self.num_proposals = num_proposals
        self.gaussian_overlap = gaussian_overlap
        self.min_radius = min_radius
        self.matcher_cls_cost = matcher_cls_cost
        self.matcher_reg_cost = matcher_reg_cost
        self.matcher_iou_cost = matcher_iou_cost
        self.cls_weight = cls_weight
        self.loc_weight = loc_weight
        self.heatmap_weight = heatmap_weight
        self.iou_weight = iou_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

        pcr, vs = self.point_cloud_range, self.voxel_size
        grid = [int(round((pcr[i + 3] - pcr[i]) / vs[i])) for i in range(3)]
        self.feature_map_size = (grid[0] // feature_map_stride, grid[1] // feature_map_stride)

        self.register_buffer("code_weights", torch.tensor(list(code_weights), dtype=torch.float32))

    def forward(self, output: "TransFusionHeadOutput", data: Dict[str, Any]) -> Dict[str, Tensor]:
        r"""Compute the TransFusion loss and its terms.

        Args:
            output: The head's raw output: per-query `center` $(B, 2, Q)$, `height` $(B, 1, Q)$,
                `dim` $(B, 3, Q)$, `rot` $(B, 2, Q)$, `vel` $(B, 2, Q)$, `iou` $(B, 1, Q)$ and `heatmap`
                $(B, C, Q)$ class logits, plus the dense `dense_heatmap` $(B, C, H, W)$.
            data: Packed ground truth (`DataKeys.BOX` $(K, 7)$ full-extent, `DataKeys.LABEL` $(K,)$
                $0$-based, `DataKeys.BATCH_BOX` $(K,)$ per-box scene index).

        Returns:
            A dict with the scalar `loss` and detached `heatmap_loss`, `cls_loss`, `box_loss`, `iou_loss`.
        """
        center = output["center"]
        batch_size, _, num_queries = center.shape

        boxes_per_scene, labels_per_scene = self._densify_gt(data, batch_size, center.device)

        hm_target = self._heatmap_targets(boxes_per_scene, labels_per_scene)
        heatmap_loss = gaussian_focal_loss(output["dense_heatmap"].sigmoid(), hm_target) * self.heatmap_weight

        decoded = self._decode_queries(output)
        assignment, ious = self._match(decoded, output["heatmap"], boxes_per_scene, labels_per_scene)
        counts = [boxes.shape[0] for boxes in boxes_per_scene]
        num_pos = sum(min(num_queries, count) for count in counts)

        # Targets of the matched queries, gathered from the ground truth of the whole batch at once.
        labels = center.new_full((batch_size, num_queries), self.num_classes, dtype=torch.long)
        bbox_targets = center.new_zeros((batch_size, num_queries, 10))
        bbox_weights = center.new_zeros((batch_size, num_queries, 10))
        iou_sum = center.new_zeros(())
        if num_pos > 0:
            matched = assignment >= 0
            starts = center.new_tensor([sum(counts[:b]) for b in range(batch_size)], dtype=torch.long)
            gt_index = assignment.clamp(min=0) + starts.unsqueeze(1)
            all_boxes = torch.cat(boxes_per_scene)
            all_labels = torch.cat(labels_per_scene)

            labels = torch.where(matched, all_labels[gt_index], labels)
            encoded = self._encode(all_boxes[gt_index.view(-1)]).view(batch_size, num_queries, 10)
            bbox_targets = torch.where(matched.unsqueeze(-1), encoded, bbox_targets)
            bbox_weights = matched.to(center.dtype).unsqueeze(-1).expand(batch_size, num_queries, 10)

            # IoU-rescore target: the 3D IoU of each matched query's decoded box with its ground-truth box.
            for b in range(batch_size):
                if counts[b] == 0:
                    continue

                iou = ious[b].gather(1, assignment[b].clamp(min=0).unsqueeze(1)).squeeze(1)
                iou_sum = iou_sum + ((output["iou"][b, 0] - (iou * 2 - 1)).abs() * matched[b]).sum()

        cls_loss = self._cls_loss(output["heatmap"], labels, num_pos) * self.cls_weight
        box_loss = self._box_loss(output, bbox_targets, bbox_weights, num_pos) * self.loc_weight
        iou_loss = iou_sum / max(num_pos, 1) * self.iou_weight

        return {
            "loss": heatmap_loss + cls_loss + box_loss + iou_loss,
            "heatmap_loss": heatmap_loss.detach(),
            "cls_loss": cls_loss.detach(),
            "box_loss": box_loss.detach(),
            "iou_loss": iou_loss.detach(),
        }

    def _heatmap_targets(self, boxes_per_scene: List[Tensor], labels_per_scene: List[Tensor]) -> Tensor:
        r"""Per-class Gaussian center heatmaps of every scene, stacked to $(B, C, H, W)$."""
        heatmaps = []
        for boxes, labels in zip(boxes_per_scene, labels_per_scene):
            heatmap, _, _, _ = draw_heatmap_targets(
                boxes,
                labels,
                self.num_classes,
                self.feature_map_size,
                self.voxel_size,
                self.point_cloud_range,
                self.feature_map_stride,
                gaussian_overlap=self.gaussian_overlap,
                min_radius=self.min_radius,
            )
            heatmaps.append(heatmap)

        return torch.stack(heatmaps)

    @staticmethod
    def _densify_gt(data: Dict[str, Any], batch_size: int, device: torch.device) -> Tuple[List[Tensor], List[Tensor]]:
        r"""Split the packed ground truth into per-scene box / zero-based-label lists, dropping empty boxes."""
        boxes_per_scene, labels_per_scene = unbatch_boxes(data, batch_size, device)
        kept_boxes: List[Tensor] = []
        kept_labels: List[Tensor] = []
        for boxes, labels in zip(boxes_per_scene, labels_per_scene):
            keep = (boxes[:, 3] > 0) & (boxes[:, 4] > 0)
            kept_boxes.append(boxes[keep])
            kept_labels.append(labels[keep])
        return kept_boxes, kept_labels

    def _decode_queries(self, output: "TransFusionHeadOutput") -> Tensor:
        r"""Decode the per-query predictions of every scene to oriented boxes $(B, Q, 7)$ in metric coordinates."""
        vs, pcr, stride = self.voxel_size, self.point_cloud_range, self.feature_map_stride
        center = output["center"]
        dim = output["dim"].exp()
        rot = output["rot"]

        x = center[:, 0] * stride * vs[0] + pcr[0]
        y = center[:, 1] * stride * vs[1] + pcr[1]
        z = output["height"][:, 0] - dim[:, 2] * 0.5
        angle = torch.atan2(rot[:, 0], rot[:, 1])
        return torch.stack([x, y, z, dim[:, 0], dim[:, 1], dim[:, 2], angle], dim=-1)

    @torch.no_grad()
    def _match(
        self,
        decoded: Tensor,
        cls_logits: Tensor,
        boxes_per_scene: List[Tensor],
        labels_per_scene: List[Tensor],
    ) -> Tuple[Tensor, List[Tensor]]:
        r"""Hungarian match of every scene's queries to its ground truth (focal cls + center $L_1$ + 3D-IoU cost).

        The costs of all scenes are solved after a single device transfer. Returns the matched box index of
        every query, $(B, Q)$ long with $-1$ when unmatched, and the per-scene query-to-box 3D IoU matrices
        $(Q, K_b)$ the IoU-rescore term reuses.
        """
        batch_size, num_queries = decoded.shape[:2]
        counts = [boxes.shape[0] for boxes in boxes_per_scene]

        prob = cls_logits.sigmoid().permute(0, 2, 1)  # (B, Q, C)
        neg = -(1 - prob + _LOG_EPS).log() * (1 - self.focal_alpha) * prob.pow(self.focal_gamma)
        pos = -(prob + _LOG_EPS).log() * self.focal_alpha * (1 - prob).pow(self.focal_gamma)
        pc_start = decoded.new_tensor(self.point_cloud_range[0:2])
        pc_range = decoded.new_tensor(self.point_cloud_range[3:5]) - pc_start

        cost = decoded.new_zeros(batch_size, num_queries, max(counts, default=0))
        ious: List[Tensor] = []
        for b, (gt_boxes, gt_labels) in enumerate(zip(boxes_per_scene, labels_per_scene)):
            if counts[b] == 0:
                ious.append(decoded.new_zeros(num_queries, 0))
                continue

            cls_cost = (pos[b][:, gt_labels] - neg[b][:, gt_labels]) * self.matcher_cls_cost
            reg_cost = torch.cdist(
                (decoded[b, :, :2] - pc_start) / pc_range, (gt_boxes[:, :2] - pc_start) / pc_range, p=1
            )
            iou = boxes_iou3d(decoded[b], gt_boxes[:, :7])
            ious.append(iou)
            cost[b, :, : counts[b]] = cls_cost + reg_cost * self.matcher_reg_cost - iou * self.matcher_iou_cost

        return hungarian_match_batched(cost, counts), ious

    def _encode(self, boxes: Tensor) -> Tensor:
        r"""Encode matched ground-truth boxes into the $10$-dim query box code (grid center, box-top $z$, log size, sincos).

        The height code is $z + d_z / 2$: the head's decode recovers the gravity center as
        $\text{height} - d_z / 2$ after exponentiating the size code.
        """
        vs, pcr, stride = self.voxel_size, self.point_cloud_range, self.feature_map_stride
        targets = boxes.new_zeros((boxes.shape[0], 10))
        targets[:, 0] = (boxes[:, 0] - pcr[0]) / (stride * vs[0])
        targets[:, 1] = (boxes[:, 1] - pcr[1]) / (stride * vs[1])
        targets[:, 2] = boxes[:, 2] + 0.5 * boxes[:, 5]
        targets[:, 3:6] = boxes[:, 3:6].log()
        targets[:, 6] = torch.sin(boxes[:, 6])
        targets[:, 7] = torch.cos(boxes[:, 6])
        if boxes.shape[1] >= 9:
            targets[:, 8:10] = boxes[:, 7:9]
        return targets

    def _cls_loss(self, cls_logits: Tensor, labels: Tensor, num_pos: int) -> Tensor:
        r"""Sigmoid focal classification over one-hot query targets (background maps to an all-zero row)."""
        batch_size, num_queries = labels.shape
        cls_score = cls_logits.permute(0, 2, 1)
        one_hot = cls_score.new_zeros((batch_size, num_queries, self.num_classes + 1))
        one_hot.scatter_(-1, labels.unsqueeze(-1), 1.0)
        one_hot = one_hot[..., : self.num_classes]
        loss = sigmoid_focal_loss(cls_score, one_hot, alpha=self.focal_alpha, gamma=self.focal_gamma)
        return loss.sum() / max(num_pos, 1)

    def _box_loss(
        self,
        output: "TransFusionHeadOutput",
        bbox_targets: Tensor,
        bbox_weights: Tensor,
        num_pos: int,
    ) -> Tensor:
        r"""Code-weighted $L_1$ box regression over the matched queries."""
        preds = torch.cat(
            [output["center"], output["height"], output["dim"], output["rot"], output["vel"]], dim=1
        ).permute(0, 2, 1)
        reg_weights = bbox_weights * self.code_weights
        loss = (preds - bbox_targets).abs() * reg_weights
        return loss.sum() / max(num_pos, 1)
