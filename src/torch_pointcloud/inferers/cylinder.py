r"""Cylinder inference for instance segmentation of large scenes, merging instances across overlapping cylinders.

Tiles the scene with vertical cylinders centred on a regular grid in the $xy$ plane, predicts the instances of each
cylinder, and merges them into one labelling of the scene by block merging, the cylinders taken in grid order.
"""

from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import torch
from torch import Tensor
from tqdm import tqdm

from torch_pointcloud.ops.mask import mask_to_instance
from torch_pointcloud.ops.superpoint import point_mask_index
from torch_pointcloud.utils.data import DataKeys, collate
from torch_pointcloud.utils.types import InstanceSegmentation3D

from ._utils import apply_transform, check_batch_alignment, index_select_dict
from .inferer import Inferer

MergeCriterion = Literal["iou", "overlap"]


# inspired by: https://github.com/SmartForest-no/SegmentAnyTree/blob/a3561ed8447bbb7938f059ba65a3e9c97d6e2ee9/torch_points3d/core/data_transform/transforms.py#L184
def cylinder_centers(xy: Tensor, spacing: float, *, align: bool = False) -> Tensor:
    r"""Centres of a regular grid covering the bounding box of planar points.

    The grid starts at the minimum corner of the box and steps by `spacing` up to one step past its maximum, the
    first axis outermost. With `align`, the box is taken in the principal axes of the points (each axis signed so
    that its largest coordinate is positive) and the centres are mapped back to the input frame.

    Args:
        xy: Planar point positions, shape $(N, 2)$.
        spacing: Distance between neighbouring centres.
        align: Lay the grid along the principal axes of `xy` instead of the coordinate axes.

    Returns:
        The grid centres, shape $(M, 2)$, in the dtype of `xy`; none without points.

    Shape:
        - xy: $(N, 2)$
        - output: $(M, 2)$

    Example:
        ```pycon
        >>> import torch
        >>> from torch_pointcloud.inferers.cylinder import cylinder_centers
        >>> cylinder_centers(torch.tensor([[0.0, 0.0], [10.0, 4.0]]), spacing=8.0).tolist()
        [[0.0, 0.0], [0.0, 8.0], [8.0, 0.0], [8.0, 8.0], [16.0, 0.0], [16.0, 8.0]]

        ```
    """
    if xy.size(0) == 0:
        return xy.new_zeros(0, 2)

    points = xy.double()
    mean = points.mean(dim=0)
    axes = torch.eye(2, dtype=points.dtype, device=points.device)
    if align:
        centered = points - mean
        covariance = centered.T @ centered / max(points.size(0) - 1, 1)
        axes = torch.linalg.eigh(covariance).eigenvectors.flip(1).T
        signs = torch.sign(axes[torch.arange(2), axes.abs().argmax(dim=1)])
        axes = axes * signs.unsqueeze(1)
    else:
        mean = torch.zeros_like(mean)

    reduced = (points - mean) @ axes.T
    low, high = reduced.min(dim=0).values, reduced.max(dim=0).values
    steps = [torch.arange(float(low[i]), float(high[i]) + spacing, spacing, dtype=points.dtype) for i in range(2)]
    grid = torch.cartesian_prod(*steps).to(points.device)
    return (grid @ axes + mean).to(xy.dtype)


# inspired by: https://github.com/SmartForest-no/SegmentAnyTree/blob/a3561ed8447bbb7938f059ba65a3e9c97d6e2ee9/torch_points3d/metrics/panoptic_tracker_pointgroup_treeins.py#L360
def _merge_cylinder(
    labels: Tensor,
    num_instances: int,
    members: Tensor,
    instance: Tensor,
    criterion: MergeCriterion,
    threshold: float,
) -> Tuple[Tensor, int]:
    """Block merging of one cylinder's `instance` ids (per member point, -1 for none) into the scene `labels`.

    Returns the scene id each cylinder id went to (-1 when it brought no new point) and the new instance count.
    """
    assigned = torch.full((int(instance.max()) + 1,), -1, dtype=torch.long)
    old = labels[members]
    ids, counts = torch.unique(old[old >= 0], return_counts=True)
    sizes = dict(zip(ids.tolist(), counts.tolist()))
    for tree in torch.unique(instance[instance >= 0]).tolist():
        points = instance == tree
        previous = old[points]
        fresh = members[points][previous < 0]
        if fresh.numel() == 0:
            continue

        target = num_instances
        if bool((previous >= 0).any()):
            candidates, intersection = torch.unique(previous[previous >= 0], return_counts=True)
            # float64, so that an overlap equal to the threshold compares as the exact ratio
            intersection = intersection.double()
            stored = torch.tensor([sizes[g] for g in candidates.tolist()], dtype=torch.float64)
            size = int(points.sum())
            if criterion == "iou":
                overlap = intersection / (stored + size - intersection)
            else:
                overlap = torch.maximum(intersection / stored, intersection / size)
            best = int(overlap.argmax())
            if float(overlap[best]) > threshold:
                target = int(candidates[best])
                sizes[target] += fresh.numel()

        if target == num_instances:
            num_instances += 1
        labels[fresh] = target
        assigned[tree] = target
    return assigned, num_instances


class CylinderInstanceSegmentationInferer(Inferer[InstanceSegmentation3D]):
    r"""Instance segmentation of large scenes over overlapping vertical cylinders.

    The cylinders have radius `radius` in the $xy$ plane and unbounded height, centred on the grid of
    `cylinder_centers` with step `spacing` (along the principal axes of the scene with `align`); a point belongs to
    every cylinder whose axis lies within `radius` of it. A `spacing` of at most $\sqrt{2} \cdot \text{radius}$
    puts every point in a cylinder. Each cylinder runs through `transform`, cylinders are collated `sw_batch_size` at
    a time and passed to `predictor`, which returns their instances as an `InstanceSegmentation3D` over the collated
    rows. The instances it flags in `ignore_mask` are left out, which is where its own suppression and thresholds go;
    `transform`'s `inverse` takes the masks of the other ones from the rows to the points of the cylinder.

    Within a cylinder, a point covered by several instances takes the highest-scoring one. The cylinders are then
    merged in grid order by block merging: an instance whose points are all new to the scene, or that matches no
    instance of the scene, gets a new id; otherwise its new points join the instance of the scene with which it
    overlaps most, when that overlap exceeds `merge_threshold`. The overlap is measured on the points of the
    cylinder, by their intersection over union (`"iou"`) or over the smaller of the two (`"overlap"`). The
    instances of a cylinder are merged by increasing score.

    When the predictor also returns `semantic_logits`, they are summed per point over the cylinders.

    Args:
        radius: Radius of the cylinders, in the units of `data[pos_key]`.
        spacing: Step of the grid of cylinder centres, at most $\sqrt{2} \cdot \text{radius}$.
        align: Lay the grid along the principal axes of the scene's $xy$ positions.
        merge_criterion: Overlap measure of block merging, `"iou"` or `"overlap"`.
        merge_threshold: Overlap above which an instance joins an instance of the scene.
        transform: Callable applied to each cylinder before prediction.
        inverse_key: Key under which `transform` records its point-to-row map, when it changes the rows.
        sw_batch_size: Number of cylinders per predictor call.
        pos_key: Key of the point positions.
        batch_key: Key of the per-point batch index.
        progress: Show a progress bar over the cylinders.

    Example:
        ```python
        import torch

        from torch_pointcloud.inferers import CylinderInstanceSegmentationInferer

        def predictor(batch):
            rows = (batch["pos"][:, 2] > 1.0).nonzero().squeeze(1)
            index = torch.zeros(1, dtype=torch.long)
            mask_index = torch.stack([torch.zeros_like(rows), rows])
            return {"mask_index": mask_index, "labels": index, "scores": torch.ones(1), "batch": index}

        data = {"pos": torch.rand(1000, 3) * 20, "batch": torch.zeros(1000, dtype=torch.long)}
        inferer = CylinderInstanceSegmentationInferer(radius=8.0, spacing=8.0, merge_threshold=0.1)
        instances = inferer(data, predictor)
        ```
    """

    def __init__(
        self,
        radius: float,
        spacing: float,
        *,
        align: bool = False,
        merge_criterion: MergeCriterion = "iou",
        merge_threshold: float = 0.5,
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        inverse_key: Optional[str] = None,
        sw_batch_size: int = 1,
        pos_key: str = DataKeys.POS,
        batch_key: str = DataKeys.BATCH,
        progress: bool = False,
    ) -> None:
        if radius <= 0 or spacing <= 0:
            raise ValueError(f"radius and spacing must be positive, got {radius} and {spacing}.")

        if spacing > radius * 2**0.5:
            raise ValueError(
                f"spacing {spacing} leaves points between the cylinders uncovered; it must be at most radius * sqrt(2) "
                f"= {radius * 2**0.5:.4g}."
            )

        if merge_criterion not in ("iou", "overlap"):
            raise ValueError(f"Invalid merge_criterion: {merge_criterion!r}. Expected 'iou' or 'overlap'.")

        if sw_batch_size < 1:
            raise ValueError(f"sw_batch_size must be at least 1, got {sw_batch_size}.")

        self.radius = radius
        self.spacing = spacing
        self.align = align
        self.merge_criterion = merge_criterion
        self.merge_threshold = merge_threshold
        self.transform = transform
        self.inverse_key = inverse_key
        self.sw_batch_size = sw_batch_size
        self.pos_key = pos_key
        self.batch_key = batch_key
        self.progress = progress

    def forward(
        self,
        data: Dict[str, Any],
        predictor: Callable[[Dict[str, Any]], InstanceSegmentation3D],
    ) -> InstanceSegmentation3D:
        r"""Segment every scene of `data` into instances.

        Args:
            data: Packed-batch dict holding `pos_key` $(N, 3)$ and `batch_key` $(N,)$.
            predictor: Callable taking a collated batch of cylinders and returning their instances over its rows.

        Returns:
            The instances of the batch: `mask_index` $(2, M)$, the (instance, point) pairs over its points sorted by
            instance then point, each point in one instance at most; `labels`, `scores` (the highest score among the
            merged cylinder instances) and `batch` $(K,)$; and the summed `semantic_logits` $(N, C)$ when the
            predictor returns them.
        """
        pos, batch = data[self.pos_key], data[self.batch_key]
        check_batch_alignment(pos, batch, self.pos_key, self.batch_key)
        labels = torch.full((pos.size(0),), -1, dtype=torch.long)
        instance_labels: List[int] = []
        instance_scores: List[float] = []
        instance_batch: List[int] = []
        semantic_logits: Optional[Tensor] = None
        num_instances = 0

        for scene in torch.unique(batch).tolist():
            points = (batch == scene).nonzero().squeeze(1)
            xy = pos[points, :2]
            cylinders = []
            for center in cylinder_centers(xy, self.spacing, align=self.align):
                members = ((xy - center) ** 2).sum(dim=1) <= self.radius**2
                if bool(members.any()):
                    cylinders.append(points[members.nonzero().squeeze(1)])

            starts = range(0, len(cylinders), self.sw_batch_size)
            for start in tqdm(starts, desc="Cylinders", disable=not self.progress):
                pack = cylinders[start : start + self.sw_batch_size]
                samples, inverses = [], []
                for members in pack:
                    sample = index_select_dict(data, members, pos.size(0))
                    sample[self.batch_key] = torch.zeros(members.numel(), dtype=torch.long, device=pos.device)
                    sample, inverse = apply_transform(
                        sample,
                        self.transform,
                        pos_key=self.pos_key,
                        inverse_key=self.inverse_key,
                    )
                    samples.append(sample)
                    inverses.append(inverse)

                output = predictor(collate(samples, batch_from=self.pos_key, batch_key=self.batch_key))
                pair_instance, pair_row = output["mask_index"]
                kept = (
                    ~output["ignore_mask"]
                    if "ignore_mask" in output
                    else torch.ones_like(output["batch"], dtype=torch.bool)
                )
                row = 0
                for i, (members, inverse) in enumerate(zip(pack, inverses)):
                    num_rows = int(samples[i][self.pos_key].size(0))
                    rows = slice(row, row + num_rows)
                    row += num_rows
                    own = ((output["batch"] == i) & kept).nonzero().squeeze(1)
                    # The cylinder's instances by increasing score, so that their ids give the merging order.
                    own = own[torch.argsort(output["scores"][own], stable=True)]
                    rank = torch.full_like(output["batch"], -1)
                    rank[own] = torch.arange(own.numel(), device=own.device)
                    pairs = rank[pair_instance] >= 0
                    mask_index = torch.stack([rank[pair_instance[pairs]], pair_row[pairs] - rows.start])
                    if inverse is not None:
                        mask_index = point_mask_index(mask_index, inverse)

                    if "semantic_logits" in output:
                        logits = output["semantic_logits"][rows]
                        logits = logits if inverse is None else logits[inverse]
                        if semantic_logits is None:
                            semantic_logits = logits.new_zeros(pos.size(0), logits.size(1))
                        semantic_logits[members] += logits

                    instance = mask_to_instance(mask_index, output["scores"][own], members.numel()).cpu()
                    if not bool((instance >= 0).any()):
                        continue

                    assigned, num_instances = _merge_cylinder(
                        labels,
                        num_instances,
                        members.cpu(),
                        instance,
                        self.merge_criterion,
                        self.merge_threshold,
                    )
                    scores, classes = output["scores"][own].tolist(), output["labels"][own].tolist()
                    for tree, target in enumerate(assigned.tolist()):
                        if target == len(instance_scores):
                            instance_scores.append(scores[tree])
                            instance_labels.append(classes[tree])
                            instance_batch.append(scene)
                        elif target >= 0:
                            instance_scores[target] = max(instance_scores[target], scores[tree])

        point = (labels >= 0).nonzero().squeeze(1)
        point = point[torch.argsort(labels[point], stable=True)]
        result: InstanceSegmentation3D = {
            "mask_index": torch.stack([labels[point], point]).to(pos.device),
            "labels": torch.tensor(instance_labels, dtype=torch.long, device=pos.device),
            "scores": torch.tensor(instance_scores, dtype=torch.float, device=pos.device),
            "batch": torch.tensor(instance_batch, dtype=torch.long, device=pos.device),
        }
        if semantic_logits is not None:
            result["semantic_logits"] = semantic_logits
        return result
