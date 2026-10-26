r"""SPFormer set-prediction loss: Hungarian query-to-instance matching over superpoint masks, with per-layer terms."""

from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from torch_pointcloud.losses.mask import dice_loss, mask_matching_cost
from torch_pointcloud.losses.matching import hungarian_match_batched
from torch_pointcloud.ops.superpoint import superpoint_instance_masks
from torch_pointcloud.utils.data import DataKeys

if TYPE_CHECKING:
    from torch_pointcloud.models.spformer import SPFormerOutput, SPFormerPrediction

__all__ = ["SPFormerLoss"]


class _SceneTarget(NamedTuple):
    scene: int
    superpoints: slice
    masks: Tensor  # (I_b, S_b) float
    labels: Tensor  # (I_b,)


class SPFormerLoss(nn.Module):
    r"""SPFormer set-prediction instance segmentation loss.

    Reference: :arxiv: [Superpoint Transformer for 3D Scene Instance Segmentation](https://arxiv.org/abs/2211.15766)
    (Sun et al., 2023).

    The ground-truth instances of each scene become superpoint masks (`superpoint_instance_masks`). Every decoder
    layer's queries are matched one-to-one to them by a Hungarian assignment whose cost (`mask_matching_cost`) sums
    the negative class probability, the mask binary cross-entropy and the mask dice loss, each weighted. The
    assignments of every layer and scene are solved after a single device transfer. Each layer is then supervised
    with:

    - **Class:** weighted cross-entropy over the $C + 1$ class logits of every query, the unmatched ones assigned the
      no-object slot, down-weighted by `no_object_weight`.
    - **Mask:** binary cross-entropy and dice loss between the matched queries' mask logits and their instances.
    - **Score:** mean squared error between the matched queries' scores and the IoU of their binarized masks with
      their instances, over the pairs whose IoU exceeds `score_iou_threshold`.

    The mask and score terms are averaged over the scenes of the batch, except the last layer's dice term, which is
    summed over them. `loss` sums the terms of every layer, while the returned `class_loss`, `mask_loss`, `dice_loss`
    and `score_loss` are the last layer's (detached). Ground truth is read from the batch at the resolution of
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
        num_scenes = output["class_logits"].shape[0]
        sizes = torch.bincount(output["superpoint_batch"], minlength=num_scenes).tolist()
        counts = torch.bincount(batch, minlength=num_scenes).tolist()
        targets = self._scene_targets(masks, labels, sizes, counts)
        layers: List["SPFormerPrediction"] = [*output["aux_outputs"], output]
        matches = self._match(layers, targets)

        total = output["class_logits"].new_zeros(())
        for index, (layer, layer_matches) in enumerate(zip(layers, matches)):
            terms = self._layer_loss(layer, targets, layer_matches, sum_dice=index == len(layers) - 1)
            for value in terms.values():
                total = total + value
        result = {"loss": total}
        result.update({name: value.detach() for name, value in terms.items()})
        return result

    @staticmethod
    def _scene_targets(masks: Tensor, labels: Tensor, sizes: List[int], counts: List[int]) -> List[_SceneTarget]:
        r"""Split the instance masks $(I, S)$ and classes $(I,)$ by scene, keeping the scenes with an instance."""
        targets = []
        start, first = 0, 0
        for scene, (size, count) in enumerate(zip(sizes, counts)):
            if count > 0:
                target = _SceneTarget(
                    scene,
                    slice(start, start + size),
                    masks[first : first + count, start : start + size].float(),
                    labels[first : first + count],
                )
                targets.append(target)
            start += size
            first += count
        return targets

    @torch.no_grad()
    def _match(
        self,
        layers: List["SPFormerPrediction"],
        targets: List[_SceneTarget],
    ) -> List[List[Tuple[Tensor, Tensor]]]:
        r"""Hungarian assignment of the queries of every decoder layer and scene to the ground-truth instances.

        The cost matrices of all layers and scenes are padded and solved after one device transfer, and the matched
        pairs come back in one. Returns, per layer and per scene of `targets`, the matched query indices (ascending)
        and their instance indices.
        """
        if not targets:
            return [[] for _ in layers]

        num_queries = layers[0]["class_logits"].shape[1]
        num_instances = [target.labels.shape[0] for target in targets]
        cost = layers[0]["class_logits"].new_zeros(len(layers), len(targets), num_queries, max(num_instances))
        for index, layer in enumerate(layers):
            for position, target in enumerate(targets):
                cost[index, position, :, : num_instances[position]] = mask_matching_cost(
                    layer["class_logits"][target.scene].softmax(dim=-1),
                    layer["mask_logits"][:, target.superpoints],
                    target.masks,
                    target.labels,
                    class_weight=self.class_cost,
                    mask_weight=self.mask_cost,
                    dice_weight=self.dice_cost,
                )

        assignment = hungarian_match_batched(cost.flatten(0, 1).cpu(), num_instances * len(layers))
        matched = assignment >= 0
        pairs = torch.stack([matched.nonzero(as_tuple=True)[1], assignment[matched]]).to(cost.device)
        scene_pairs = [(query, instance) for query, instance in pairs.split(matched.sum(dim=1).tolist(), dim=1)]
        return [scene_pairs[index * len(targets) : (index + 1) * len(targets)] for index in range(len(layers))]

    def _layer_loss(
        self,
        layer: "SPFormerPrediction",
        targets: List[_SceneTarget],
        matches: List[Tuple[Tensor, Tensor]],
        *,
        sum_dice: bool,
    ) -> Dict[str, Tensor]:
        r"""Weighted `class_loss`, `mask_loss`, `dice_loss` and `score_loss` of one decoder layer."""
        class_logits, scores, mask_logits = layer["class_logits"], layer["scores"], layer["mask_logits"]
        num_scenes = class_logits.shape[0]
        class_target = torch.full(class_logits.shape[:2], self.num_classes, device=class_logits.device)
        mask_loss = class_logits.new_zeros(())
        dice = class_logits.new_zeros(())
        score_loss = class_logits.new_zeros(())
        for target, (query, instance) in zip(targets, matches):
            class_target[target.scene, query] = target.labels[instance].long()
            matched, matched_masks = mask_logits[:, target.superpoints][query], target.masks[instance]
            mask_loss = mask_loss + F.binary_cross_entropy_with_logits(matched, matched_masks)
            dice = dice + dice_loss(matched, matched_masks).mean()
            with torch.no_grad():
                binary = (matched.sigmoid() >= 0.5).float()
                intersection = (binary * matched_masks).sum(dim=-1)
                iou = intersection / (binary.sum(dim=-1) + matched_masks.sum(dim=-1) - intersection + 1e-6)
            keep = (iou > self.score_iou_threshold).float()
            squared_error = (scores[target.scene, query] - iou).square()
            score_loss = score_loss + (squared_error * keep).sum() / keep.sum().clamp(min=1)

        class_loss = F.cross_entropy(class_logits.transpose(1, 2), class_target, self.class_weights)
        return {
            "class_loss": self.class_weight * class_loss,
            "mask_loss": self.mask_weight * mask_loss / num_scenes,
            "dice_loss": self.dice_weight * (dice if sum_dice else dice / num_scenes),
            "score_loss": self.score_weight * score_loss / num_scenes,
        }
