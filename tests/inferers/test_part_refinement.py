from typing import Any, Dict

import pytest
import torch
from torch import Tensor

from torch_pointcloud.inferers import PartRefinementInferer, SimpleInferer, part_refinement_inference
from torch_pointcloud.utils.data import DataKeys

_PART_IDS = [[0, 1], [2, 3]]


def _shape(labels: Tensor, category: int) -> Dict[str, Any]:
    """Points on a line at x = 0, 1, 2, ... so the nearest neighbours of a point are its index neighbours."""
    n = labels.numel()
    pos = torch.stack([torch.arange(n, dtype=torch.float32), torch.zeros(n), torch.zeros(n)], dim=1)
    return {
        DataKeys.POS: pos,
        DataKeys.BATCH: torch.zeros(n, dtype=torch.long),
        DataKeys.CATEGORY: torch.nn.functional.one_hot(torch.tensor([category]), num_classes=2).float(),
        "labels": labels,
    }


def _label_predictor(window: Dict[str, Any]) -> Tensor:
    return torch.nn.functional.one_hot(window["labels"], num_classes=4).float()


def test_part_refinement_reassigns_rare_labels_by_neighbour_majority() -> None:
    """One stray part-1 point inside a run of part 0 (category 0) is outvoted by its neighbours."""
    labels = torch.tensor([0, 0, 0, 1, 0, 0, 0])
    inferer = PartRefinementInferer(SimpleInferer(), part_ids=_PART_IDS, min_count=2, num_neighbors=3)
    out = inferer(_shape(labels, category=0), predictor=_label_predictor)
    assert out.shape == (7, 4)
    assert out.argmax(dim=1).tolist() == [0] * 7


def test_part_refinement_reassigns_foreign_parts_even_when_frequent() -> None:
    """Part 2 belongs to category 1, so on a category-0 shape it is refined even with many points, and the
    vote excludes the refined label itself: the two part-2 points take part 0 from their neighbours."""
    labels = torch.tensor([0, 0, 2, 2, 0, 0])
    inferer = PartRefinementInferer(SimpleInferer(), part_ids=_PART_IDS, min_count=1, num_neighbors=5)
    out = inferer(_shape(labels, category=0), predictor=_label_predictor)
    assert out.argmax(dim=1).tolist() == [0] * 6


def test_part_refinement_keeps_plausible_predictions() -> None:
    labels = torch.tensor([0, 0, 0, 1, 1, 1])
    inferer = PartRefinementInferer(SimpleInferer(), part_ids=_PART_IDS, min_count=3, num_neighbors=3)
    out = inferer(_shape(labels, category=0), predictor=_label_predictor)
    assert out.argmax(dim=1).tolist() == labels.tolist()


def test_part_refinement_single_label_shape_is_left_alone() -> None:
    """A shape predicted as one part throughout has nothing to vote with and stays as is (even if foreign)."""
    labels = torch.tensor([2, 2, 2, 2])
    inferer = PartRefinementInferer(SimpleInferer(), part_ids=_PART_IDS)
    out = inferer(_shape(labels, category=0), predictor=_label_predictor)
    assert out.argmax(dim=1).tolist() == [2, 2, 2, 2]


def test_part_refinement_handles_packed_shapes_and_index_categories() -> None:
    a = _shape(torch.tensor([0, 0, 1, 0, 0]), category=0)
    b = _shape(torch.tensor([2, 2, 3, 2, 2]), category=1)
    packed = {
        DataKeys.POS: torch.cat([a[DataKeys.POS], b[DataKeys.POS]]),
        DataKeys.BATCH: torch.cat([torch.zeros(5, dtype=torch.long), torch.ones(5, dtype=torch.long)]),
        DataKeys.CATEGORY: torch.tensor([0, 1]),
        "labels": torch.cat([a["labels"], b["labels"]]),
    }
    inferer = PartRefinementInferer(SimpleInferer(), part_ids=_PART_IDS, min_count=2, num_neighbors=3)
    out = inferer(packed, predictor=_label_predictor)
    assert out.argmax(dim=1).tolist() == [0] * 5 + [2] * 5


def test_part_refinement_validates_args_and_keys() -> None:
    data = _shape(torch.tensor([0, 1]), category=0)
    with pytest.raises(ValueError, match="min_count"):
        part_refinement_inference(data, predictor=_label_predictor, part_ids=_PART_IDS, min_count=0)
    with pytest.raises(ValueError, match="num_neighbors"):
        part_refinement_inference(data, predictor=_label_predictor, part_ids=_PART_IDS, num_neighbors=0)
    del data[DataKeys.CATEGORY]
    with pytest.raises(KeyError, match="category"):
        PartRefinementInferer(SimpleInferer(), part_ids=_PART_IDS)(data, predictor=_label_predictor)


def test_part_refinement_keeps_label_when_votes_all_zero() -> None:
    """An isolated blob uniformly predicted one refined-away label has an all-zero vote row; it must keep
    its label instead of falling through to class 0."""
    pos = torch.cat([torch.rand(64, 3), torch.rand(32, 3) + 50.0])
    labels = torch.cat([torch.full((64,), 1, dtype=torch.long), torch.full((32,), 4, dtype=torch.long)])
    scores = torch.nn.functional.one_hot(labels, num_classes=6).float()

    def predictor(data: Dict[str, Any]) -> Tensor:
        return scores

    data: Dict[str, Any] = {
        DataKeys.POS: pos,
        DataKeys.BATCH: torch.zeros(96, dtype=torch.long),
        DataKeys.CATEGORY: torch.tensor([0]),
    }
    out = part_refinement_inference(data, predictor=predictor, part_ids=[[1, 2]], num_neighbors=8, min_count=10)
    assert bool((out[64:].argmax(dim=-1) == 4).all())
