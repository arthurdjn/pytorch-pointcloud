from typing import Any, Dict, List

import pytest
import torch
from torch import Tensor

from torch_pointcloud.inferers import PotentialSphereInferer, potential_sphere_inference
from torch_pointcloud.utils.data import DataKeys


def _room(n: int = 2000, *, seed: int = 0, extent: float = 4.0) -> Dict[str, Any]:
    g = torch.Generator().manual_seed(seed)
    pos = torch.rand(n, 3, generator=g) * torch.tensor([extent, extent, 2.0])
    return {DataKeys.POS: pos, DataKeys.BATCH: torch.zeros(n, dtype=torch.long), "x": torch.rand(n, 2, generator=g)}


def _left_right_predictor(window: Dict[str, Any]) -> Tensor:
    """Class 1 for points right of the sphere centre (x > 0), class 0 otherwise, with high confidence."""
    pos = window[DataKeys.POS]
    logits = torch.zeros(pos.size(0), 2)
    logits[:, 1] = 6.0 * (pos[:, 0] > 0).float() - 3.0
    return logits


def test_potential_sphere_covers_every_point_with_probabilities() -> None:
    data = _room()
    out = PotentialSphereInferer(radius=1.5, num_votes=2.0, seed=0)(data, predictor=_left_right_predictor)
    assert out.shape == (2000, 2)
    assert bool((out.sum(dim=1) > 0).all())
    # EMA of softmax rows never exceeds 1 and every row is a convex blend of probabilities.
    assert float(out.sum(dim=1).max()) <= 1.0 + 1e-5


def test_potential_sphere_more_votes_means_more_spheres() -> None:
    data = _room()
    calls: List[int] = []

    def predictor(window: Dict[str, Any]) -> Tensor:
        calls.append(int(window[DataKeys.POS].size(0)))
        return _left_right_predictor(window)

    PotentialSphereInferer(radius=1.5, num_votes=1.0, seed=0)(data, predictor=predictor)
    few = len(calls)
    calls.clear()
    PotentialSphereInferer(radius=1.5, num_votes=4.0, seed=0)(data, predictor=predictor)
    assert len(calls) > few


def test_potential_sphere_transform_sees_centred_spheres_and_batches_them() -> None:
    """Each sphere is centred on its own centre before `transform`, all its points lie within `radius`, and
    `sw_batch_size` spheres are packed into one predictor call with a fresh batch index."""
    data = _room()
    seen: List[Dict[str, Any]] = []

    def transform(sphere: Dict[str, Any]) -> Dict[str, Any]:
        seen.append(sphere)
        return sphere

    def predictor(window: Dict[str, Any]) -> Tensor:
        assert int(window[DataKeys.BATCH].max()) <= 2
        return _left_right_predictor(window)

    PotentialSphereInferer(radius=1.0, num_votes=1.0, sw_batch_size=3, transform=transform, seed=0)(
        data, predictor=predictor
    )
    assert seen
    for sphere in seen:
        assert float(sphere[DataKeys.POS].norm(dim=1).max()) < 1.0
        assert sphere["x"].shape[0] == sphere[DataKeys.POS].shape[0]


def test_potential_sphere_inner_ratio_limits_the_update() -> None:
    """A point only receives predictions from spheres whose centre is within `inner_ratio * radius`, so with a
    tiny inner ratio some points never get an update and stay at zero (default ratio covers everything)."""
    data = _room(n=400)
    predictor = _left_right_predictor
    tight = PotentialSphereInferer(radius=1.5, num_votes=1.0, inner_ratio=0.05, jitter=0.0, seed=0)(
        data, predictor=predictor
    )
    full = PotentialSphereInferer(radius=1.5, num_votes=1.0, inner_ratio=1.0, jitter=0.0, seed=0)(
        data, predictor=predictor
    )
    assert int((tight.sum(dim=1) == 0).sum()) > int((full.sum(dim=1) == 0).sum())


def test_potential_sphere_batched_matches_per_scene() -> None:
    a, b = _room(n=500, seed=1), _room(n=300, seed=2, extent=3.0)
    packed = {
        DataKeys.POS: torch.cat([a[DataKeys.POS], b[DataKeys.POS]]),
        DataKeys.BATCH: torch.cat([torch.zeros(500, dtype=torch.long), torch.ones(300, dtype=torch.long)]),
        "x": torch.cat([a["x"], b["x"]]),
    }
    inferer = PotentialSphereInferer(radius=1.5, num_votes=1.0, seed=0)
    out = inferer(packed, predictor=_left_right_predictor)
    out_a = inferer(a, predictor=_left_right_predictor)
    assert out.shape == (800, 2)
    # Scene 0 is processed first with the same generator state either way.
    assert torch.allclose(out[:500], out_a)


def test_potential_sphere_row_altering_transform_raises() -> None:
    data = _room(n=200)

    def drop_one(sphere: Dict[str, Any]) -> Dict[str, Any]:
        sphere = dict(sphere)
        sphere[DataKeys.POS] = sphere[DataKeys.POS][:-1]
        return sphere

    with pytest.raises(ValueError, match="row count"):
        PotentialSphereInferer(radius=1.5, num_votes=1.0, transform=drop_one, seed=0)(
            data, predictor=_left_right_predictor
        )


def test_potential_sphere_empty_scene_returns_zero_by_zero() -> None:
    data = {DataKeys.POS: torch.empty(0, 3), DataKeys.BATCH: torch.empty(0, dtype=torch.long)}
    out = PotentialSphereInferer(radius=1.0)(data, predictor=_left_right_predictor)
    assert out.shape == (0, 0)


def test_potential_sphere_validates_args() -> None:
    data = {DataKeys.POS: torch.zeros(1, 3), DataKeys.BATCH: torch.zeros(1, dtype=torch.long)}
    with pytest.raises(ValueError, match="radius"):
        potential_sphere_inference(data, predictor=_left_right_predictor, radius=0.0)
    with pytest.raises(ValueError, match="num_votes"):
        potential_sphere_inference(data, predictor=_left_right_predictor, radius=1.0, num_votes=0.0)
    with pytest.raises(ValueError, match="potential_size"):
        potential_sphere_inference(data, predictor=_left_right_predictor, radius=1.0, potential_size=0.0)
    with pytest.raises(ValueError, match="jitter"):
        potential_sphere_inference(data, predictor=_left_right_predictor, radius=1.0, jitter=-1.0)
    with pytest.raises(ValueError, match="inner_ratio"):
        potential_sphere_inference(data, predictor=_left_right_predictor, radius=1.0, inner_ratio=0.0)
    with pytest.raises(ValueError, match="ema_smoothing"):
        potential_sphere_inference(data, predictor=_left_right_predictor, radius=1.0, ema_smoothing=1.0)
    with pytest.raises(ValueError, match="sw_batch_size"):
        potential_sphere_inference(data, predictor=_left_right_predictor, radius=1.0, sw_batch_size=0)
    with pytest.raises(KeyError, match="pos"):
        potential_sphere_inference(
            {DataKeys.BATCH: torch.zeros(1, dtype=torch.long)}, predictor=_left_right_predictor, radius=1.0
        )
    with pytest.raises(KeyError, match="batch"):
        potential_sphere_inference({DataKeys.POS: torch.zeros(1, 3)}, predictor=_left_right_predictor, radius=1.0)
