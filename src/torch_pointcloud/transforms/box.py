"""Transforms that build and encode 3D box targets."""

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor

from torch_pointcloud.ops.box3d import boxes_iou_bev, points_in_boxes
from torch_pointcloud.utils.conversion import ensure_tuple
from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.random import Randomizable
from torch_pointcloud.utils.types import KeyCollection

from .base import DictTransform
from .geometry import rotation_matrix
from .utils import relabel

__all__ = [
    "InstanceToBox",
    "PasteBoxes",
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
    boolean `dst_ignore_mask_key` consumed by `box_matches`, which excuses an unmatched prediction only on ignore
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
        dst_ignore_mask_key: Output key for the written boolean ignore mask.
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
        dst_ignore_mask_key: str = "ignore_mask",
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.mapping = {int(k): int(v) for k, v in mapping.items()}
        self.label_key = label_key
        self.ignore_mapping = {int(k): int(v) for k, v in (ignore_mapping or {}).items()}
        self.ignore_fields = dict(ignore_fields or {})
        self.dst_ignore_mask_key = dst_ignore_mask_key

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
        new_labels = relabel(labels, {**self.ignore_mapping, **self.mapping}, default=-1)

        for key in self.iter_keys(d):
            d[key] = d[key][keep]
        d[self.label_key] = new_labels[keep]
        d[self.dst_ignore_mask_key] = ignore[keep]
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


def points_in_oriented_box(pos: Tensor, box: Tensor) -> Tensor:
    r"""Test which points lie inside a single oriented 3D box.

    The point offsets relative to the box center are rotated into the box frame by $-\theta$ about $+z$, then
    compared against the half-extents with an axis-aligned bounding-box test. The heading $\theta$ is in
    radians counterclockwise about $+z$. For a box with zero heading this reduces to a plain axis-aligned
    test.

    Args:
        pos: Coordinate tensor of shape $(N, 3)$.
        box: A single box of shape $(7,)$ as $[c_x, c_y, c_z, h_x, h_y, h_z, \theta]$ with **half**-extents.

    Returns:
        A boolean mask of shape $(N,)$ that is `True` for points inside the box.
    """
    center = box[0:3]
    half = box[3:6]
    heading = box[6]
    rotation = rotation_matrix(float(-heading), axis=2, device=pos.device).to(pos.dtype)
    local = (pos - center) @ rotation.transpose(-1, -2)
    return (local.abs() <= half).all(dim=1)


def cut_boxes(
    data: Mapping[str, Any],
    keys: KeyCollection = DataKeys.POS,
    box_key: str = DataKeys.BOX,
    label_key: str = DataKeys.LABEL,
    attribute_keys: Optional[KeyCollection] = None,
) -> List[Dict[str, Tensor]]:
    r"""Cut the annotated objects of a scene out by their boxes, one dict per object.

    An object holds the points inside its box under `keys`, the coordinates relative to the box center under the
    first key, its box under `box_key`, its label under `label_key` and its value of every `attribute_keys`. A
    point inside several boxes belongs to each of them. The keys are plain strings, so a list of objects saves
    with `torch.save` and loads back with `weights_only=True`.
    [`PasteBoxes`][torch_pointcloud.transforms.box.PasteBoxes] pastes such objects into other scenes.

    Args:
        data: Scene holding the packed points under `keys`, the boxes $(K, 7)$ or wider under `box_key`, the
            labels $(K,)$ under `label_key` and one row per box under each of `attribute_keys`.
        keys: Point keys cut out with the objects, the coordinates first.
        box_key: Key of the boxes, whose first seven columns are read.
        label_key: Key of the labels.
        attribute_keys: Per-box keys cut out with the objects (e.g. a velocity).

    Returns:
        The objects of the scene, in the order of its boxes.

    Example:
        ```python
        import torch
        import torch_pointcloud.transforms.functional as F
        from torch_pointcloud.datasets import KITTI

        objects = []
        for data in KITTI(root="data/KITTI"):
            objects.extend(F.cut_boxes(data, keys=["pos", "intensity"]))
        objects = [obj for obj in objects if len(obj["pos"]) >= 5]
        torch.save(objects, "objects.pt")
        ```
    """
    keys = ensure_tuple(keys)
    attribute_keys = ensure_tuple(attribute_keys, none_as_empty=True)
    pos_key = keys[0]
    boxes = data[box_key]
    if boxes.shape[0] == 0:
        return []

    inside = points_in_boxes(data[pos_key], boxes[:, :7])  # (N, K)
    objects: List[Dict[str, Tensor]] = []
    for k in range(boxes.shape[0]):
        index = inside[:, k].nonzero(as_tuple=True)[0]
        obj = {str(pos_key): data[pos_key][index] - boxes[k, :3]}
        for key in keys[1:]:
            obj[str(key)] = data[key][index]
        obj[str(box_key)] = boxes[k].clone()
        obj[str(label_key)] = data[label_key][k].clone()
        for key in attribute_keys:
            obj[str(key)] = data[key][k].clone()
        objects.append(obj)
    return objects


class PasteBoxes(DictTransform, Randomizable):
    r"""Paste objects cut out by `cut_boxes` into the scene, a target number per label.

    For each label of `num_samples`, objects are drawn at random from `objects` and pasted: the scene points
    inside their boxes are removed, the object points, moved to their box center, are placed before the remaining
    scene points with their features, and the boxes, labels and attributes are appended. A drawn object whose box
    overlaps in bird's-eye view a box of the scene or of another drawn object is left out. With `count_existing`
    the objects of a label already in the scene count towards its target, otherwise the target is pasted in full.
    Per-box keys of the scene other than `box_key`, `label_key` and `attribute_keys` are not extended: drop them
    beforehand (e.g. with `KeepItems`).

    Args:
        objects: Objects of [`cut_boxes`][torch_pointcloud.transforms.box.cut_boxes], holding `keys`, `box_key`,
            `label_key` and `attribute_keys`.
        num_samples: Target number of objects per label, `{label: count}`.
        keys: Point keys pasted: the coordinates first, then the features.
        box_key: Key of the boxes.
        label_key: Key of the labels.
        attribute_keys: Per-box keys appended with the attributes of the pasted objects (e.g. a velocity).
        count_existing: Count the objects already in the scene towards the targets.
        seed: Seed of the transform's own random stream; `None` draws from the global generator (see `Randomizable`).
        allow_missing_keys: If `True`, silently skip absent keys.

    Example:
        ```python
        import torch
        import torch_pointcloud.transforms as T
        import torch_pointcloud.transforms.functional as F

        scene = {
            "pos": torch.randn(1000, 3) * 10,
            "intensity": torch.rand(1000, 1),
            "box": torch.tensor([[2.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0], [10.0, 5.0, 0.0, 1.0, 1.0, 2.0, 0.3]]),
            "label": torch.tensor([0, 1]),
        }
        objects = F.cut_boxes(scene, keys=["pos", "intensity"])  # from one scene here, from a dataset in practice
        paste = T.PasteBoxes(objects, num_samples={0: 20, 1: 15}, keys=["pos", "intensity"])
        out = paste(scene)
        print(out["box"].shape[0])
        ```
    """

    def __init__(
        self,
        objects: Sequence[Mapping[str, Tensor]],
        num_samples: Mapping[int, int],
        keys: KeyCollection = DataKeys.POS,
        box_key: str = DataKeys.BOX,
        label_key: str = DataKeys.LABEL,
        attribute_keys: Optional[KeyCollection] = None,
        count_existing: bool = True,
        seed: Optional[int] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        attribute_keys = ensure_tuple(attribute_keys, none_as_empty=True)
        if len(objects) == 0:
            raise ValueError("`objects` is empty.")
        needed = (*self.keys, box_key, label_key, *attribute_keys)
        missing = [key for key in needed if key not in objects[0]]
        if missing:
            raise ValueError(f"The objects hold no {missing}; `cut_boxes` must store these keys.")

        self.objects = list(objects)
        self.num_samples = dict(num_samples)
        self.box_key = box_key
        self.label_key = label_key
        self.attribute_keys = attribute_keys
        self.count_existing = count_existing
        self.set_random_state(seed)

        # The boxes of all objects for the overlap checks, and the objects of every label to draw from.
        self._boxes = torch.stack([obj[box_key] for obj in self.objects]).float()
        labels = torch.tensor([int(obj[label_key]) for obj in self.objects])
        self._members = {label: (labels == label).nonzero(as_tuple=True)[0] for label in self.num_samples}

    def _draw(self, label: int, count: int) -> Tensor:
        """Indices of `count` objects of `label`, drawn without replacement."""
        members = self._members[label]
        if count <= 0 or members.numel() == 0:
            return members[:0]

        order = torch.randperm(members.numel(), generator=self.R)[:count]
        return members[order]

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        boxes, labels = d[self.box_key], d[self.label_key]
        pos_key = self.keys[0]

        # Draw per label, leaving out the candidates that overlap a scene box or another candidate.
        existing = boxes[:, :7].to(self._boxes)
        chosen: List[Tensor] = []
        for label, target in self.num_samples.items():
            count = target - int((labels == label).sum()) if self.count_existing else target
            drawn = self._draw(label, count)
            if drawn.numel() == 0:
                continue

            candidates = self._boxes[drawn, :7]
            overlap = boxes_iou_bev(candidates, candidates).fill_diagonal_(0).amax(dim=1)
            if existing.shape[0] > 0:
                overlap = overlap + boxes_iou_bev(candidates, existing).amax(dim=1)
            kept = drawn[overlap == 0]
            existing = torch.cat([existing, self._boxes[kept, :7]])
            chosen.append(kept)

        index = torch.cat(chosen) if chosen else torch.empty(0, dtype=torch.long)
        if index.numel() == 0:
            return d

        # The scene points under the pasted boxes go away; the object points come first, at their box center.
        pasted = [self.objects[i] for i in index.tolist()]
        pasted_boxes = self._boxes[index]
        keep = ~points_in_boxes(d[pos_key], pasted_boxes[:, :7].to(d[pos_key])).any(dim=1)
        pos = torch.cat([obj[pos_key] + obj[self.box_key][:3] for obj in pasted])
        d[pos_key] = torch.cat([pos.to(d[pos_key]), d[pos_key][keep]])
        for key in self.keys[1:]:
            features = torch.cat([obj[key] for obj in pasted])
            d[key] = torch.cat([features.to(d[key]), d[key][keep]])
        d[self.box_key] = torch.cat([boxes, pasted_boxes.to(boxes)])
        d[self.label_key] = torch.cat([labels, torch.stack([obj[self.label_key] for obj in pasted]).to(labels)])
        for key in self.attribute_keys:
            attributes = torch.stack([obj[key] for obj in pasted])
            d[key] = torch.cat([d[key], attributes.to(d[key])])
        return d
