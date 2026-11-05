from typing import Any, Dict, List

import numpy as np
import pytest
import torch
from torch import Tensor

import torch_pointcloud.transforms as T
from torch_pointcloud.inferers import CylinderInstanceSegmentationInferer, cylinder_centers
from torch_pointcloud.inferers.cylinder import _merge_cylinder
from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.imports import _PYG_LIB_AVAILABLE
from torch_pointcloud.utils.types import InstanceSegmentation3D


def _block_merging(
    labels: np.ndarray, count: int, members: np.ndarray, new: np.ndarray, iou: bool, threshold: float
) -> int:
    """Block merging as the forest panoptic reference trackers write it, one stored tree recomputed at a time."""
    if not (new != -1).any():
        return count
    labelled = labels[members] != -1
    if not labelled.any():
        valid = new != -1
        labels[members[valid]] = new[valid] + count
        return count + int(new.max()) + 1
    if labelled.all():
        return count
    for tree in range(int(new.max()) + 1):
        points = members[new == tree]
        old = points[labels[points] != -1]
        fresh = points[labels[points] == -1]
        if len(old) == 0:
            labels[fresh] = count
            count += 1
        elif len(fresh):
            best, best_overlap = 0, 0.0
            for g in np.unique(labels[old]):
                stored = members[labels[members] == g]
                inter = len(np.intersect1d(stored, points))
                if iou:
                    overlap = inter / len(np.union1d(stored, points))
                else:
                    overlap = max(inter / len(stored), inter / len(points))
                if overlap > best_overlap:
                    best, best_overlap = g, overlap
            if best_overlap > threshold:
                labels[fresh] = best
            else:
                labels[fresh] = count
                count += 1
    return count


def _partition(labels: np.ndarray) -> List[frozenset]:
    return sorted((frozenset(np.nonzero(labels == g)[0].tolist()) for g in np.unique(labels[labels >= 0])), key=min)


@pytest.mark.parametrize(
    ("criterion", "threshold"),
    [pytest.param("iou", 0.1, id="iou"), pytest.param("overlap", 0.3, id="overlap")],
)
def test_merge_cylinder_matches_the_reference_block_merging(criterion: str, threshold: float) -> None:
    rng = np.random.default_rng(0)
    for _ in range(20):
        num_points = 200
        ours = torch.full((num_points,), -1, dtype=torch.long)
        reference = np.full(num_points, -1)
        count = ours_count = 0
        for _ in range(6):
            members = np.sort(rng.choice(num_points, size=rng.integers(20, 120), replace=False))
            new = np.where(rng.random(len(members)) < 0.2, -1, rng.integers(0, 4, len(members)))
            count = _block_merging(reference, count, members, new, criterion == "iou", threshold)
            if (new >= 0).any():
                _, ours_count = _merge_cylinder(
                    ours,
                    ours_count,
                    torch.from_numpy(members),
                    torch.from_numpy(new),
                    criterion,  # type: ignore[arg-type]
                    threshold,
                )
        assert _partition(ours.numpy()) == _partition(reference)


def test_cylinder_centers_cover_the_bounding_box() -> None:
    centers = cylinder_centers(torch.tensor([[0.0, 0.0], [10.0, 4.0]]), spacing=8.0)
    assert centers.tolist() == [[0.0, 0.0], [0.0, 8.0], [8.0, 0.0], [8.0, 8.0], [16.0, 0.0], [16.0, 8.0]]


def test_cylinder_centers_without_points() -> None:
    assert cylinder_centers(torch.zeros(0, 2), spacing=8.0).shape == (0, 2)


def test_cylinder_inferer_covers_every_point_at_the_largest_spacing() -> None:
    """At a spacing of radius * sqrt(2), the corners of a grid cell are its farthest points from the four centres."""
    xy = torch.rand(2000, 2, generator=torch.Generator().manual_seed(0)) * 40
    data: Dict[str, Any] = {
        DataKeys.POS: torch.cat([xy, torch.ones(2000, 1)], dim=1),
        DataKeys.BATCH: torch.zeros(2000, dtype=torch.long),
        DataKeys.INDEX: torch.arange(2000),
    }
    seen = torch.zeros(2000, dtype=torch.long)

    def predictor(batch: Dict[str, Any]) -> InstanceSegmentation3D:
        seen.index_add_(0, batch[DataKeys.INDEX], torch.ones_like(batch[DataKeys.INDEX]))
        empty = torch.zeros(0, dtype=torch.long)
        mask_index = torch.zeros(2, 0, dtype=torch.long)
        return {"mask_index": mask_index, "labels": empty, "scores": torch.zeros(0), "batch": empty}

    CylinderInstanceSegmentationInferer(radius=4.0, spacing=4.0 * 2**0.5)(data, predictor)
    assert bool((seen > 0).all())


def test_cylinder_centers_align_with_the_principal_axes() -> None:
    """A diagonal strip takes two rows of centres along it once aligned, a full square grid otherwise."""
    t = torch.linspace(0.0, 30.0, 200)
    xy = torch.stack([t, t], dim=1) + torch.tensor([[0.3, -0.3]]) * torch.rand(200, 1)
    aligned = cylinder_centers(xy, spacing=8.0, align=True)
    square = cylinder_centers(xy, spacing=8.0)
    assert len(aligned) == 2 * 7
    assert len(square) == 5 * 5
    distance_to_strip = (aligned[:, 0] - aligned[:, 1]).abs() / 2**0.5
    assert (distance_to_strip <= 8.5).all()


def _two_trees() -> Dict[str, Any]:
    """Two thin vertical columns on grid nodes 16 m apart, over a 32 x 16 m ground, all as one scene."""
    generator = torch.Generator().manual_seed(0)
    trees = [
        torch.tensor([[x - 0.5, 7.5, 0.6]]) + torch.rand(300, 3, generator=generator) * torch.tensor([1.0, 1.0, 10.0])
        for x in (8.0, 24.0)
    ]
    ground = torch.rand(400, 3, generator=generator) * torch.tensor([32.0, 16.0, 0.0])
    pos = torch.cat([*trees, ground, torch.zeros(1, 3)])
    return {DataKeys.POS: pos, DataKeys.BATCH: torch.zeros(len(pos), dtype=torch.long)}


def _column_predictor(batch: Dict[str, Any]) -> InstanceSegmentation3D:
    """One instance per column of every cylinder: the points above the ground, grouped by their x."""
    pos, scene = batch[DataKeys.POS], batch[DataKeys.BATCH]
    masks, scenes = [], []
    for b in torch.unique(scene).tolist():
        above = (scene == b) & (pos[:, 2] > 0.5)
        for column in torch.unique(torch.div(pos[above, 0], 16.0, rounding_mode="floor")).tolist():
            masks.append(above & (torch.div(pos[:, 0], 16.0, rounding_mode="floor") == column))
            scenes.append(b)
    mask = torch.stack(masks) if masks else torch.zeros(0, len(pos), dtype=torch.bool)
    return {
        "mask_index": mask.nonzero().t(),
        "labels": torch.zeros(len(masks), dtype=torch.long),
        "scores": torch.linspace(0.5, 0.9, len(masks)),
        "batch": torch.tensor(scenes, dtype=torch.long),
        "semantic_logits": torch.ones(len(pos), 2),
    }


def _dense(instances: InstanceSegmentation3D, num_points: int) -> Tensor:
    masks = torch.zeros(len(instances["scores"]), num_points, dtype=torch.bool)
    masks[instances["mask_index"][0], instances["mask_index"][1]] = True
    return masks


@pytest.mark.parametrize("sw_batch_size", [pytest.param(1, id="one"), pytest.param(4, id="packed")])
def test_cylinder_inferer_merges_each_tree_across_cylinders(sw_batch_size: int) -> None:
    data = _two_trees()
    inferer = CylinderInstanceSegmentationInferer(
        radius=8.0,
        spacing=8.0,
        merge_threshold=0.1,
        sw_batch_size=sw_batch_size,
    )
    out = inferer(data, _column_predictor)

    tree = data[DataKeys.POS][:, 2] > 0.5
    masks = _dense(out, len(tree))
    assert masks.shape == (2, len(tree))
    assert torch.equal(out["mask_index"], masks.nonzero().t())
    assert torch.equal(masks.any(dim=0), tree)
    assert masks[:, :300].all(dim=1).tolist() == [True, False]
    assert out["batch"].tolist() == [0, 0]
    centers = cylinder_centers(data[DataKeys.POS][:, :2], 8.0)
    covering = ((data[DataKeys.POS][:, None, :2] - centers[None]) ** 2).sum(dim=-1) <= 64.0
    torch.testing.assert_close(out["semantic_logits"][:, 0], covering.sum(dim=1).float())


@pytest.mark.skipif(not _PYG_LIB_AVAILABLE, reason="pyg-lib is not installed")
def test_cylinder_inferer_maps_rows_back_through_the_inverse() -> None:
    data = _two_trees()
    transform = T.Voxelize(pos_key=DataKeys.POS, pos_reduce="first", size=0.5, dst_inverse_key=DataKeys.INVERSE)
    seen: List[Tensor] = []

    def predictor(batch: Dict[str, Any]) -> InstanceSegmentation3D:
        seen.append(batch[DataKeys.POS])
        return _column_predictor(batch)

    inferer = CylinderInstanceSegmentationInferer(
        radius=8.0,
        spacing=8.0,
        merge_threshold=0.1,
        transform=transform,
        inverse_key=DataKeys.INVERSE,
    )
    out = inferer(data, predictor)

    assert sum(len(pos) for pos in seen) < 3 * len(data[DataKeys.POS])
    masks = _dense(out, len(data[DataKeys.POS]))
    assert masks.shape == (2, len(data[DataKeys.POS]))
    assert torch.equal(masks.any(dim=0), data[DataKeys.POS][:, 2] > 0.5)


def test_cylinder_inferer_without_instances() -> None:
    data = _two_trees()

    def predictor(batch: Dict[str, Any]) -> InstanceSegmentation3D:
        empty = torch.zeros(0, dtype=torch.long)
        return {
            "mask_index": torch.zeros(2, 0, dtype=torch.long),
            "labels": empty,
            "scores": torch.zeros(0),
            "batch": empty,
        }

    out = CylinderInstanceSegmentationInferer(radius=8.0, spacing=8.0)(data, predictor)
    assert out["mask_index"].shape == (2, 0)
    assert "semantic_logits" not in out


def test_cylinder_inferer_leaves_the_ignored_instances_out() -> None:
    """The predictor flags the instances right of x = 16 m: only the first tree is left."""

    def predictor(batch: Dict[str, Any]) -> InstanceSegmentation3D:
        instances = _column_predictor(batch)
        instance, row = instances["mask_index"]
        right = batch[DataKeys.POS][row, 0] >= 16.0
        instances["ignore_mask"] = torch.zeros_like(instances["batch"], dtype=torch.bool).index_fill_(
            0, instance[right], True
        )
        return instances

    data = _two_trees()
    out = CylinderInstanceSegmentationInferer(radius=8.0, spacing=8.0, merge_threshold=0.1)(data, predictor)
    masks = _dense(out, len(data[DataKeys.POS]))
    assert torch.equal(masks.any(dim=0), (data[DataKeys.POS][:, 2] > 0.5) & (data[DataKeys.POS][:, 0] < 16.0))


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param(dict(radius=0.0, spacing=8.0), id="radius"),
        pytest.param(dict(radius=8.0, spacing=-1.0), id="spacing"),
        pytest.param(dict(radius=4.0, spacing=10.0), id="uncovered"),
        pytest.param(dict(radius=8.0, spacing=8.0, merge_criterion="dice"), id="criterion"),
        pytest.param(dict(radius=8.0, spacing=8.0, sw_batch_size=0), id="sw-batch-size"),
    ],
)
def test_cylinder_inferer_rejects_invalid_arguments(kwargs: Dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        CylinderInstanceSegmentationInferer(**kwargs)
