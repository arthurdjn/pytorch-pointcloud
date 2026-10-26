r"""SPFormer set-prediction loss: Hungarian query-to-instance matching over superpoint masks, with per-layer terms."""

from typing import TYPE_CHECKING, Any, Dict, List

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from torch_pointcloud.losses.mask import dice_loss, pairwise_binary_cross_entropy_with_logits, pairwise_dice_loss
from torch_pointcloud.losses.matching import hungarian_match
from torch_pointcloud.ops.superpoint import superpoint_instance_masks
from torch_pointcloud.utils.data import DataKeys

if TYPE_CHECKING:
    from torch_pointcloud.models.spformer import SPFormerOutput, SPFormerPrediction

__all__ = ["SPFormerLoss"]


class SPFormerLoss(nn.Module):
    r"""SPFormer set-prediction instance segmentation loss.

    Reference: :arxiv: [Superpoint Transformer for 3D Scene Instance Segmentation](https://arxiv.org/abs/2211.15766)
    (Sun et al., 2023).

    The ground-truth instances of each scene become superpoint masks (`superpoint_instance_masks`). Every decoder
    layer's queries are matched one-to-one to them by a Hungarian assignment whose cost sums the negative class
    probability, the mask binary cross-entropy and the mask dice loss, each weighted. Each layer is then supervised
    with:

    - **Class:** weighted cross-entropy over the $C + 1$ class logits of every query, the unmatched ones assigned the
      no-object slot, down-weighted by `no_object_weight`.
    - **Mask:** binary cross-entropy and dice loss between the matched queries' mask logits and their instances.
    - **Score:** mean squared error between the matched queries' scores and the IoU of their binarized masks with
      their instances, over the pairs whose IoU exceeds `score_iou_threshold`.

    The mask and score terms are averaged over the scenes of the batch, except the last layer's dice term, which is
    summed over them. The layer losses are summed. Ground truth is read from the batch at the resolution of
    `output["superpoint"]`: the instance id and the class of every point (`ignore_index` for no instance class).

    Args:
        num_classes: Number of instance classes.
        class_weight: Weight of the class term.
        mask_weight: Weight of the mask binary cross-entropy term.
        dice_weight: Weight of the mask dice term.
        score_weight: Weight of the score term.
        class_cost: Weight of the class probability in the matching cost.
        mask_cost: Weight of the mask binary cross-entropy in the matching cost.
        dice_cost: Weight of the mask dice loss in the matching cost.
        no_object_weight: Cross-entropy weight of the no-object slot.
        score_iou_threshold: IoU a matched pair must exceed to supervise the score.
        ignore_index: Class of the points that belong to no instance class.
        instance_key: Batch key of the per-point instance ids.
        label_key: Batch key of the per-point classes.
    """

    class_weights: Tensor

    def __init__(
        self,
        num_classes: int,
        *,
        class_weight: float = 0.5,
        mask_weight: float = 1.0,
        dice_weight: float = 1.0,
        score_weight: float = 0.5,
        class_cost: float = 0.5,
        mask_cost: float = 1.0,
        dice_cost: float = 1.0,
        no_object_weight: float = 0.1,
        score_iou_threshold: float = 0.5,
        ignore_index: int = -1,
        instance_key: str = DataKeys.INSTANCE,
        label_key: str = DataKeys.SEGMENT,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.class_weight = class_weight
        self.mask_weight = mask_weight
        self.dice_weight = dice_weight
        self.score_weight = score_weight
        self.class_cost = class_cost
        self.mask_cost = mask_cost
        self.dice_cost = dice_cost
        self.score_iou_threshold = score_iou_threshold
        self.ignore_index = ignore_index
        self.instance_key = instance_key
        self.label_key = label_key
        class_weights = torch.ones(num_classes + 1)
        class_weights[-1] = no_object_weight
        self.register_buffer("class_weights", class_weights)

    def forward(self, output: "SPFormerOutput", data: Dict[str, Any]) -> Dict[str, Tensor]:
        masks, labels, batch = superpoint_instance_masks(
            data[self.instance_key],
            data[self.label_key],
            output["superpoint"],
            output["superpoint_batch"],
            ignore_index=self.ignore_index,
        )
        layers: List["SPFormerPrediction"] = [*output["aux_outputs"], output]
        total = output["class_logits"].new_zeros(())
        for index, layer in enumerate(layers):
            terms = self.layer_loss(
                layer["class_logits"],
                layer["scores"],
                layer["mask_logits"],
                output["superpoint_batch"],
                masks,
                labels,
                batch,
                sum_dice=index == len(layers) - 1,
            )
            for value in terms.values():
                total = total + value
        result = {"loss": total}
        result.update({name: value.detach() for name, value in terms.items()})
        return result

    def layer_loss(
        self,
        class_logits: Tensor,
        scores: Tensor,
        mask_logits: Tensor,
        superpoint_batch: Tensor,
        masks: Tensor,
        labels: Tensor,
        batch: Tensor,
        *,
        sum_dice: bool = False,
    ) -> Dict[str, Tensor]:
        r"""Weighted loss terms of one decoder layer.

        Args:
            class_logits: Per-query class logits, shape $(B, Q, C + 1)$.
            scores: Per-query mask quality scores, shape $(B, Q)$.
            mask_logits: Per-query mask logits over the superpoints of its scene, shape $(Q, S)$.
            superpoint_batch: Scene index of every superpoint, shape $(S,)$, sorted.
            masks: Ground-truth instance masks over the superpoints, shape $(I, S)$.
            labels: Ground-truth instance classes, shape $(I,)$.
            batch: Scene index of every instance, shape $(I,)$.
            sum_dice: Sum the dice term over the scenes instead of averaging it.

        Returns:
            The weighted `class_loss`, `mask_loss`, `dice_loss` and `score_loss` terms.
        """
        num_scenes = class_logits.shape[0]
        sizes = torch.bincount(superpoint_batch, minlength=num_scenes).tolist()
        instance_counts = torch.bincount(batch, minlength=num_scenes).tolist()
        class_target = torch.full(class_logits.shape[:2], self.num_classes, device=class_logits.device)
        mask_loss = class_logits.new_zeros(())
        dice = class_logits.new_zeros(())
        score_loss = class_logits.new_zeros(())
        start, first = 0, 0
        for scene, (size, count) in enumerate(zip(sizes, instance_counts)):
            if count > 0:
                logits = mask_logits[:, start : start + size]
                target = masks[first : first + count, start : start + size].float()
                target_labels = labels[first : first + count]
                with torch.no_grad():
                    cost = (
                        -self.class_cost * class_logits[scene].softmax(dim=-1)[:, target_labels]
                        + self.mask_cost * pairwise_binary_cross_entropy_with_logits(logits, target)
                        + self.dice_cost * pairwise_dice_loss(logits, target)
                    )
                query, instance = hungarian_match(cost)
                class_target[scene, query] = target_labels[instance].long()
                matched, target = logits[query], target[instance]
                mask_loss = mask_loss + F.binary_cross_entropy_with_logits(matched, target)
                dice = dice + dice_loss(matched, target).mean()
                with torch.no_grad():
                    binary = (matched.sigmoid() >= 0.5).float()
                    intersection = (binary * target).sum(dim=-1)
                    iou = intersection / (binary.sum(dim=-1) + target.sum(dim=-1) - intersection + 1e-6)
                keep = iou > self.score_iou_threshold
                if keep.any():
                    score_loss = score_loss + F.mse_loss(scores[scene, query][keep], iou[keep])
            start += size
            first += count

        class_loss = F.cross_entropy(class_logits.transpose(1, 2), class_target, self.class_weights)
        return {
            "class_loss": self.class_weight * class_loss,
            "mask_loss": self.mask_weight * mask_loss / num_scenes,
            "dice_loss": self.dice_weight * (dice if sum_dice else dice / num_scenes),
            "score_loss": self.score_weight * score_loss / num_scenes,
        }
