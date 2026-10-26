r"""Loss of the RoI refinement heads of the two-stage detectors.

The RoIs are sampled and matched to the ground truth by [`sample_rois`][torch_pointcloud.ops.roi.sample_rois]
during the training forward; [`roi_head_loss`][torch_pointcloud.losses.roi_head.roi_head_loss] then supervises
the confidence logit and the box residuals of every sampled RoI:

```python
import torch
from torch_pointcloud.losses.functional import roi_head_loss

rois = torch.tensor([[0.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0], [20.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0]])
target_boxes = torch.tensor([[0.2, 0.0, 0.0, 4.0, 2.0, 1.5, 0.1], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
cls_loss, box_loss, corner = roi_head_loss(
    torch.zeros(2),
    torch.zeros(2, 7),
    rois,
    target_boxes,
    torch.tensor([0.9, 0.0]),
    cls_foreground_threshold=0.75,
    cls_background_threshold=0.25,
    reg_foreground_threshold=0.55,
)
loss = cls_loss + box_loss + corner
```
"""

from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from torch_pointcloud.losses.corner import corner_loss
from torch_pointcloud.ops.roi import decode_roi_residuals, encode_roi_residuals
from torch_pointcloud.utils.data import DataKeys

if TYPE_CHECKING:
    from torch_pointcloud.models.voxel_rcnn import VoxelRCNNTrainOutput


def roi_head_loss(
    cls_logits: Tensor,
    box_codes: Tensor,
    rois: Tensor,
    target_boxes: Tensor,
    iou: Tensor,
    *,
    cls_foreground_threshold: float,
    cls_background_threshold: float,
    reg_foreground_threshold: float,
    code_weights: Optional[Tensor] = None,
    smooth_l1_beta: float = 1.0 / 9.0,
) -> Tuple[Tensor, Tensor, Tensor]:
    r"""Confidence, box and corner losses of the RoIs of a refinement head.

    The confidence target of a RoI is its IoU with its matched box, mapped linearly from
    `[cls_background_threshold, cls_foreground_threshold]` onto $[0, 1]$ and clamped; the binary cross-entropy is
    averaged over the RoIs. The RoIs whose IoU exceeds `reg_foreground_threshold` regress their matched box: the
    code-weighted smooth-$L_1$ of the residuals against
    [`encode_roi_residuals`][torch_pointcloud.ops.roi.encode_roi_residuals] is summed over the codes and averaged
    over these RoIs, and the [`corner_loss`][torch_pointcloud.losses.corner.corner_loss] of the decoded boxes is
    averaged over them.

    Args:
        cls_logits: Confidence logit of every RoI, shape $(R,)$.
        box_codes: Predicted residuals of every RoI, shape $(R, 7)$.
        rois: RoIs $(c_x, c_y, c_z, d_x, d_y, d_z, \theta)$, shape $(R, 7)$.
        target_boxes: Matched ground-truth box of every RoI, shape $(R, 7)$; read for the regressed RoIs only.
        iou: 3D IoU of every RoI with its matched box, shape $(R,)$.
        cls_foreground_threshold: IoU from which the confidence target is one.
        cls_background_threshold: IoU up to which the confidence target is zero.
        reg_foreground_threshold: IoU above which a RoI regresses its matched box.
        code_weights: Per-code weight of the residual loss, shape $(7,)$; one for every code when `None`.
        smooth_l1_beta: Smooth-$L_1$ transition point $\beta$ of the residual loss.

    Returns:
        The scalar confidence, box and corner losses.

    Shape:
        - cls_logits, iou: $(R,)$
        - box_codes, rois, target_boxes: $(R, 7)$
        - output: three scalars

    Example:
        ```pycon
        >>> rois = torch.tensor([[0.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0]])
        >>> cls_loss, box_loss, corner = roi_head_loss(
        ...     torch.zeros(1),
        ...     torch.zeros(1, 7),
        ...     rois,
        ...     rois,
        ...     torch.tensor([1.0]),
        ...     cls_foreground_threshold=0.75,
        ...     cls_background_threshold=0.25,
        ...     reg_foreground_threshold=0.55,
        ... )
        >>> round(cls_loss.item(), 4), box_loss.item(), corner.item()
        (0.6931, 0.0, 0.0)

        ```
    """
    cls_target = ((iou - cls_background_threshold) / (cls_foreground_threshold - cls_background_threshold)).clamp(0, 1)
    cls_loss = F.binary_cross_entropy_with_logits(cls_logits, cls_target)

    foreground = iou > reg_foreground_threshold
    num_foreground = foreground.sum().clamp_min(1)
    residuals = box_codes[foreground] - encode_roi_residuals(target_boxes[foreground], rois[foreground])
    if code_weights is not None:
        residuals = residuals * code_weights
    box_loss = F.smooth_l1_loss(residuals, torch.zeros_like(residuals), beta=smooth_l1_beta, reduction="sum")
    box_loss = box_loss / num_foreground

    decoded = decode_roi_residuals(box_codes[foreground], rois[foreground])
    corner = corner_loss(decoded, target_boxes[foreground]).sum() / num_foreground
    return cls_loss, box_loss, corner


class RoIHeadLoss(nn.Module):
    r"""Loss of a RoI refinement head, on the RoIs sampled by the training forward of its detector.

    Reads the confidence logits `rcnn_cls`, the residuals `rcnn_reg`, the sampled `rois`, their `roi_iou` and
    `roi_target_index` from the training output, gathers the matched boxes from the packed `DataKeys.BOX` ground
    truth, and weights the three terms of [`roi_head_loss`][torch_pointcloud.losses.roi_head.roi_head_loss].

    Args:
        cls_foreground_threshold: IoU from which the confidence target is one.
        cls_background_threshold: IoU up to which the confidence target is zero.
        reg_foreground_threshold: IoU above which a RoI regresses its matched box.
        code_weights: Per-code weight of the residual loss, shape $(7,)$.
        cls_weight: Weight of the confidence term in the total.
        reg_weight: Weight of the residual term in the total.
        corner_weight: Weight of the corner term in the total.
        smooth_l1_beta: Smooth-$L_1$ transition point $\beta$ of the residual loss.
    """

    code_weights: Tensor

    def __init__(
        self,
        *,
        cls_foreground_threshold: float,
        cls_background_threshold: float,
        reg_foreground_threshold: float,
        code_weights: Sequence[float] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        cls_weight: float = 1.0,
        reg_weight: float = 1.0,
        corner_weight: float = 1.0,
        smooth_l1_beta: float = 1.0 / 9.0,
    ) -> None:
        if not cls_background_threshold < cls_foreground_threshold:
            raise ValueError("`cls_background_threshold` must be below `cls_foreground_threshold`.")

        super().__init__()
        self.cls_foreground_threshold = cls_foreground_threshold
        self.cls_background_threshold = cls_background_threshold
        self.reg_foreground_threshold = reg_foreground_threshold
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight
        self.corner_weight = corner_weight
        self.smooth_l1_beta = smooth_l1_beta
        self.register_buffer("code_weights", torch.tensor(list(code_weights), dtype=torch.float32), persistent=False)

    def forward(self, output: "VoxelRCNNTrainOutput", data: Dict[str, Any]) -> Dict[str, Tensor]:
        r"""Compute the refinement loss and its terms.

        Args:
            output: The training output of the model: `rcnn_cls` $(R, 1)$, `rcnn_reg` $(R, 7)$, `rois` $(R, 7)$,
                `roi_iou` $(R,)$ and `roi_target_index` $(R,)$ into the packed ground-truth boxes.
            data: Packed ground truth with `DataKeys.BOX` $(K, 7)$.

        Returns:
            A dict with the scalar `loss` and the detached `rcnn_cls_loss`, `rcnn_box_loss` and `rcnn_corner_loss`.
        """
        rois, boxes = output["rois"], data[DataKeys.BOX]
        target_boxes = boxes[output["roi_target_index"].clamp_min(0)] if boxes.numel() > 0 else torch.zeros_like(rois)
        cls_loss, box_loss, corner = roi_head_loss(
            output["rcnn_cls"].view(-1),
            output["rcnn_reg"],
            rois,
            target_boxes,
            output["roi_iou"],
            cls_foreground_threshold=self.cls_foreground_threshold,
            cls_background_threshold=self.cls_background_threshold,
            reg_foreground_threshold=self.reg_foreground_threshold,
            code_weights=self.code_weights,
            smooth_l1_beta=self.smooth_l1_beta,
        )
        loss = self.cls_weight * cls_loss + self.reg_weight * box_loss + self.corner_weight * corner
        return {
            "loss": loss,
            "rcnn_cls_loss": cls_loss.detach(),
            "rcnn_box_loss": box_loss.detach(),
            "rcnn_corner_loss": corner.detach(),
        }

    def extra_repr(self) -> str:
        return (
            f"cls_foreground_threshold={self.cls_foreground_threshold}, "
            f"cls_background_threshold={self.cls_background_threshold}, "
            f"reg_foreground_threshold={self.reg_foreground_threshold}"
        )
