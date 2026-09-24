"""Transforms that build and encode 3D box targets."""

from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor

from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.types import KeyCollection

from . import functional as F
from .base import DictTransform

__all__ = [
    "EncodeVoteNetTargets",
    "GenerateVoteLabels",
    "InstanceToBox",
    "RelabelBoxes",
]


class RelabelBoxes(DictTransform):
    r"""Map raw box labels to a detection class set and flag don't-care boxes for the AP metric.

    A detection dataset (e.g. `KITTI`) returns the **raw** annotated boxes: every labeled class plus
    per-box attributes such as occlusion / truncation. This transform turns those into the inputs the
    3D AP metric expects, the way `Relabel` turns raw segmentation ids into a benchmark label set:

    - boxes whose raw label is a key of `mapping` are kept as ground truth, relabeled to `mapping[raw]`;
    - boxes whose raw label is a key of `ignore_mapping` (neighboring classes, e.g. KITTI `Van` for
      `Car`) are kept as **ignore regions** (`ignore_mask = True`), labeled `ignore_mapping[raw]`: the
      evaluated class they excuse. They suppress false positives of that class but are not scored;
    - a kept foreground box that falls outside any range in `ignore_fields` (e.g. KITTI's moderate rule:
      occlusion $\le 1$, truncation $\le 0.3$, 2D height $\ge 25$ px) is downgraded to an ignore region
      attributed to its mapped class;
    - every other box is dropped.

    All keys in `keys` (the box tensor and every per-box attribute, including those named in
    `ignore_fields`) are filtered together by the keep mask so they stay row-aligned. The output adds the
    boolean `ignore_mask_key` consumed by `box_matches`, which excuses an unmatched prediction only on ignore
    boxes labeled with its class.

    ![RelabelBoxes before / after](../../assets/transforms/relabel_boxes.png)

    Args:
        keys: Per-box tensors to filter together (e.g. `DataKeys.BOX`, `DataKeys.LABEL`,
            `DataKeys.TRUNCATION`, `DataKeys.OCCLUSION`). Must include `label_key` and every key
            referenced by `ignore_fields`.
        mapping: Raw-label to detection-label dict; raw labels absent from it (and from `ignore_mapping`)
            are dropped.
        label_key: Key holding the raw integer labels (must be one of `keys`).
        ignore_mapping: Raw labels kept as ignore regions rather than scored ground truth, mapped to the
            detection class they excuse (e.g. KITTI `Van` to the `Car` class index).
        ignore_fields: Per-attribute inclusive ranges `{key: (low, high)}` (use `None` for an open side);
            a foreground box outside any range becomes an ignore region.
        ignore_mask_key: Output key for the written boolean ignore mask.
        allow_missing_keys: If `True`, skip missing keys instead of raising.

    Example:
        ```python
        import torch_pointcloud.transforms as T

        # KITTI: raw 8-class boxes -> 3 detection classes, Van / Person_sitting as ignore regions
        # for Car / Pedestrian, moderate difficulty (occlusion <= 1, truncation <= 0.3,
        # height >= 25 px) as ignore.
        T.RelabelBoxes(
            keys=("box", "label", "truncation", "occlusion", "bbox_height"),
            mapping={0: 0, 3: 1, 5: 2},
            ignore_mapping={1: 0, 4: 1},
            ignore_fields={
                "occlusion": (None, 1),
                "truncation": (None, 0.3),
                "bbox_height": (25, None),
            },
        )
        ```
    """

    def __init__(
        self,
        keys: KeyCollection,
        mapping: Dict[int, int],
        *,
        label_key: str = DataKeys.LABEL,
        ignore_mapping: Optional[Dict[int, int]] = None,
        ignore_fields: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]] = None,
        ignore_mask_key: str = "ignore_mask",
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.mapping = {int(k): int(v) for k, v in mapping.items()}
        self.label_key = label_key
        self.ignore_mapping = {int(k): int(v) for k, v in (ignore_mapping or {}).items()}
        self.ignore_fields = dict(ignore_fields or {})
        self.ignore_mask_key = ignore_mask_key

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        labels = d[self.label_key].long()
        foreground = torch.isin(labels, torch.tensor(sorted(self.mapping), device=labels.device))
        is_ignore = (
            torch.isin(labels, torch.tensor(sorted(self.ignore_mapping), device=labels.device))
            if self.ignore_mapping
            else torch.zeros_like(labels, dtype=torch.bool)
        )

        hard = torch.zeros_like(labels, dtype=torch.bool)
        for field_key, (low, high) in self.ignore_fields.items():
            value = d[field_key]
            in_range = torch.ones_like(labels, dtype=torch.bool)
            if low is not None:
                in_range &= value >= low
            if high is not None:
                in_range &= value <= high
            hard |= ~in_range

        keep = foreground | is_ignore
        ignore = is_ignore | (foreground & hard)
        new_labels = F.relabel(labels, {**self.ignore_mapping, **self.mapping}, default=-1)

        for key in self.iter_keys(d):
            d[key] = d[key][keep]
        d[self.label_key] = new_labels[keep]
        d[self.ignore_mask_key] = ignore[keep]
        return d


class InstanceToBox(DictTransform):
    r"""Axis-aligned bounding boxes from per-point instance ids (e.g. ScanNet detection targets).

    Each distinct non-negative instance id in `instance_key` becomes one axis-aligned box covering its
    `pos_key` points: the center and full extents with heading $0$, written to `dst_box_key` as $(K, 7)$
    rows $[c_x, c_y, c_z, d_x, d_y, d_z, 0]$. The box class (the instance's most common `semantic_key`
    value) is written separately to `dst_class_key` as a $(K,)$ long tensor. Negative instance ids mark
    unlabeled points and never form a box. Instances whose class equals `ignore_index` are dropped, so
    mapping the stuff / non-target semantics to `ignore_index` with a `Relabel` upstream filters the boxes
    down to the detection classes.

    ![InstanceToBox before / after](../../assets/transforms/instance_to_box.png)

    Args:
        instance_key: Key of the $(N,)$ per-point instance ids.
        semantic_key: Key of the $(N,)$ per-point class labels the box class is read from.
        pos_key: Key of the $(N, 3)$ coordinates.
        dst_box_key: Key to write the $(K, 7)$ boxes to.
        dst_class_key: Key to write the $(K,)$ per-box classes to.
        ignore_index: Class value whose instances are dropped (e.g. unlabeled / stuff).
        allow_missing_keys: If `True`, return the data unchanged when an input key is missing instead of raising.
    """

    def __init__(
        self,
        instance_key: str = "instance",
        semantic_key: str = "segment",
        pos_key: str = "pos",
        dst_box_key: str = "box",
        dst_class_key: str = "label",
        ignore_index: int = -1,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__([instance_key, semantic_key, pos_key], allow_missing_keys)
        self.instance_key = instance_key
        self.semantic_key = semantic_key
        self.pos_key = pos_key
        self.dst_box_key = dst_box_key
        self.dst_class_key = dst_class_key
        self.ignore_index = ignore_index

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        for key in (self.instance_key, self.semantic_key, self.pos_key):
            if key not in d:
                if self.allow_missing_keys:
                    return d
                raise KeyError(f"Key {key!r} was missing in the data and `allow_missing_keys==False`.")

        boxes, classes = self._instance_boxes(d[self.pos_key], d[self.instance_key], d[self.semantic_key])
        d[self.dst_box_key] = boxes
        d[self.dst_class_key] = classes
        return d

    def _instance_boxes(self, pos: Tensor, instance: Tensor, segment: Tensor) -> Tuple[Tensor, Tensor]:
        boxes: list[Tensor] = []
        classes: list[Tensor] = []
        for inst in torch.unique(instance):
            if int(inst) < 0:
                continue
            mask = instance == inst
            cls = segment[mask].mode().values
            if int(cls) == self.ignore_index:
                continue
            lo, hi = pos[mask].amin(dim=0), pos[mask].amax(dim=0)
            boxes.append(torch.cat([(lo + hi) / 2, hi - lo, pos.new_zeros(1)]))
            classes.append(cls.long())

        if not boxes:
            return pos.new_zeros((0, 7)), torch.zeros(0, dtype=torch.long, device=pos.device)
        return torch.stack(boxes), torch.stack(classes)


class GenerateVoteLabels(DictTransform):
    r"""Generate per-point vote offsets and a vote mask from oriented GT boxes.

    Each point collects the offsets to the centers of the first `gt_vote_factor` boxes containing it, in box
    order, matching the VoteNet ScanNet and SUN RGB-D vote layout: a point inside fewer boxes repeats its
    first offset in the unfilled slots, so the min-over-votes loss can credit either center on overlapping
    objects. Points inside no box receive zero offsets and a zero mask. Boxes are $(K, 7)$ rows
    $[c_x, c_y, c_z, d_x, d_y, d_z, \theta]$ with full extents and heading in radians counterclockwise about
    $+z$. When `oriented` is `True` containment is yaw-aware, otherwise an axis-aligned test is used.

    See Also:
        `torch_pointcloud.transforms.functional.points_in_oriented_box`

    ![GenerateVoteLabels before / after](../../assets/transforms/generate_vote_labels.png)

    Args:
        pos_key: Key of the $(N, 3)$ coordinate tensor.
        box_key: Key of the $(K, 7)$ box tensor (full extents, counterclockwise heading).
        vote_key: Key to write the $(N, 3 G)$ vote offsets to.
        mask_key: Key to write the $(N,)$ vote mask to.
        oriented: If `True`, use yaw-aware containment, otherwise an axis-aligned test.
        gt_vote_factor: Number $G$ of vote slots per point.
        allow_missing_keys: If `True`, return the data unchanged when `pos_key` or `box_key` is missing
            instead of raising.
    """

    def __init__(
        self,
        pos_key: str = "pos",
        box_key: str = "box",
        vote_key: str = "vote_label",
        mask_key: str = "vote_label_mask",
        oriented: bool = True,
        gt_vote_factor: int = 3,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__([pos_key, box_key], allow_missing_keys)
        self.pos_key = pos_key
        self.box_key = box_key
        self.vote_key = vote_key
        self.mask_key = mask_key
        self.oriented = oriented
        self.gt_vote_factor = gt_vote_factor

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        if self.pos_key not in d or self.box_key not in d:
            if self.allow_missing_keys:
                return d

            missing = self.pos_key if self.pos_key not in d else self.box_key
            raise KeyError(f"Key {missing!r} was missing in the data and `allow_missing_keys==False`.")

        pos = d[self.pos_key]
        boxes = d[self.box_key]
        n = pos.shape[0]
        num_slots = self.gt_vote_factor
        votes = torch.zeros(n, 3 * num_slots, device=pos.device, dtype=pos.dtype)
        mask = torch.zeros(n, device=pos.device, dtype=torch.long)
        counts = torch.zeros(n, device=pos.device, dtype=torch.long)

        for k in range(boxes.shape[0]):
            box = boxes[k]
            if self.oriented:
                half_box = torch.cat([box[0:3], box[3:6] / 2, box[6:7]])
                inside = F.points_in_oriented_box(pos, half_box)
            else:
                inside = ((pos - box[0:3]).abs() <= box[3:6] / 2).all(dim=1)
            mask[inside] = 1
            idx = (inside & (counts < num_slots)).nonzero(as_tuple=True)[0]
            offsets = box[0:3] - pos[idx]
            first = counts[idx] == 0
            votes[idx[first]] = offsets[first].repeat(1, num_slots)
            rest = idx[~first]
            cols = counts[rest, None] * 3 + torch.arange(3, device=pos.device)
            votes[rest[:, None], cols] = offsets[~first]
            counts[inside] += 1

        d[self.vote_key] = votes
        d[self.mask_key] = mask
        return d


class EncodeVoteNetTargets(DictTransform):
    r"""Encode oriented GT boxes into the padded label tensors the VoteNet loss consumes.

    Each $(K, 7)$ box row $[c_x, c_y, c_z, d_x, d_y, d_z, \theta]$ (full extents) and its class from
    `class_key` are converted to fixed-size $(M, \ldots)$ targets where $M$ is `max_num_obj`. Headings are
    binned with `angle_to_class`. The size class is the semantic class and the size residual is computed
    against `mean_sizes` (full edge lengths).

    See Also:
        `torch_pointcloud.transforms.functional.angle_to_class`,
        `torch_pointcloud.transforms.functional.class_to_size`

    ![EncodeVoteNetTargets before / after](../../assets/transforms/encode_votenet_targets.png)

    Args:
        box_key: Key of the $(K, 7)$ box tensor (full extents).
        class_key: Key of the $(K,)$ per-box class tensor.
        center_key: Key to write the $(M, 3)$ center labels to.
        heading_class_key: Key to write the $(M,)$ heading class labels to.
        heading_residual_key: Key to write the $(M,)$ heading residual labels to.
        size_class_key: Key to write the $(M,)$ size class labels to.
        size_residual_key: Key to write the $(M, 3)$ size residual labels to.
        sem_cls_key: Key to write the $(M,)$ semantic class labels to.
        box_mask_key: Key to write the $(M,)$ box mask to.
        num_heading_bins: Number of heading bins.
        mean_sizes: Template sizes of shape $(C, 3)$ holding full edge lengths per class.
        max_num_obj: Padded number of objects $M$.
        allow_missing_keys: If `True`, return the data unchanged when `box_key` or `class_key` is missing
            instead of raising.

    Raises:
        ValueError: If `mean_sizes` is not provided.
    """

    def __init__(
        self,
        box_key: str = "box",
        class_key: str = "label",
        center_key: str = "center_label",
        heading_class_key: str = "heading_class_label",
        heading_residual_key: str = "heading_residual_label",
        size_class_key: str = "size_class_label",
        size_residual_key: str = "size_residual_label",
        sem_cls_key: str = "sem_cls_label",
        box_mask_key: str = "box_label_mask",
        num_heading_bins: int = 12,
        mean_sizes: Optional[Union[Tensor, Sequence[Sequence[float]]]] = None,
        max_num_obj: int = 64,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__([box_key, class_key], allow_missing_keys)
        if mean_sizes is None:
            raise ValueError("mean_sizes must be provided with full edge lengths of shape (C, 3).")

        self.box_key = box_key
        self.class_key = class_key
        self.center_key = center_key
        self.heading_class_key = heading_class_key
        self.heading_residual_key = heading_residual_key
        self.size_class_key = size_class_key
        self.size_residual_key = size_residual_key
        self.sem_cls_key = sem_cls_key
        self.box_mask_key = box_mask_key
        self.num_heading_bins = num_heading_bins
        self.mean_sizes = torch.as_tensor(mean_sizes, dtype=torch.float32)
        self.max_num_obj = max_num_obj

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        for key in (self.box_key, self.class_key):
            if key not in d:
                if self.allow_missing_keys:
                    return d
                raise KeyError(f"Key {key!r} was missing in the data and `allow_missing_keys==False`.")

        boxes = d[self.box_key]
        classes = d[self.class_key]
        device = boxes.device
        dtype = boxes.dtype
        m = self.max_num_obj
        k = min(boxes.shape[0], m)
        mean_sizes = self.mean_sizes.to(device=device, dtype=dtype)

        center = torch.zeros(m, 3, device=device, dtype=dtype)
        heading_class = torch.zeros(m, device=device, dtype=torch.long)
        heading_residual = torch.zeros(m, device=device, dtype=dtype)
        size_class = torch.zeros(m, device=device, dtype=torch.long)
        size_residual = torch.zeros(m, 3, device=device, dtype=dtype)
        sem_cls = torch.zeros(m, device=device, dtype=torch.long)
        box_mask = torch.zeros(m, device=device, dtype=dtype)

        if k > 0:
            valid = boxes[:k]
            sem = classes[:k].long()
            center[:k] = valid[:, 0:3]
            cls, residual = F.angle_to_class(valid[:, 6], self.num_heading_bins)
            heading_class[:k] = cls
            heading_residual[:k] = residual
            size_class[:k] = sem
            size_residual[:k] = valid[:, 3:6] - mean_sizes[sem]
            sem_cls[:k] = sem
            box_mask[:k] = 1

        d[self.center_key] = center
        d[self.heading_class_key] = heading_class
        d[self.heading_residual_key] = heading_residual
        d[self.size_class_key] = size_class
        d[self.size_residual_key] = size_residual
        d[self.sem_cls_key] = sem_cls
        d[self.box_mask_key] = box_mask
        return d
