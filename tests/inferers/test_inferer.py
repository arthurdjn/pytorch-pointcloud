from typing import Any, Callable, Dict, List, Tuple

import pytest
import torch
from torch import Tensor

from torch_pointcloud.inferers import (
    Inferer,
    KNNWindowInferer,
    PotentialSphereInferer,
    SlidingWindowInferer,
    VoxelPartitionInferer,
)
from torch_pointcloud.utils.data import DataKeys


def test_inferer_is_abstract_and_cannot_be_instantiated() -> None:
    """Inferer cannot be instantiated directly -- it is an abstract base class."""
    with pytest.raises(TypeError):
        Inferer()  # type: ignore[abstract]


def test_subclass_without_forward_cannot_be_instantiated() -> None:
    """A subclass that omits `forward` stays abstract and raises TypeError on instantiation."""

    class Incomplete(Inferer):
        pass

    with pytest.raises(TypeError):
        Incomplete()  # type: ignore[abstract]


def test_call_delegates_to_forward_with_same_args_and_returns_its_value() -> None:
    """Calling an inferer instance invokes `forward` with the same arguments and returns its output."""
    calls: List[Tuple[Dict[str, Any], Callable[[Dict[str, Any]], Tensor]]] = []

    class Spy(Inferer):
        def forward(self, data: Dict[str, Any], predictor: Callable[[Dict[str, Any]], Tensor]) -> Tensor:
            calls.append((data, predictor))
            return predictor(data)

    def predictor(data: Dict[str, Any]) -> Tensor:
        return data["x"] * 2

    data: Dict[str, Any] = {"x": torch.arange(4).float()}
    out = Spy()(data, predictor=predictor)

    assert len(calls) == 1
    assert calls[0][0] is data
    assert calls[0][1] is predictor
    assert torch.equal(out, torch.arange(4).float() * 2)


@pytest.mark.parametrize(
    "inferer_fn",
    [
        pytest.param(lambda: SlidingWindowInferer(block_size=4.0, roi_num_points=16, seed=0), id="sliding_window"),
        pytest.param(lambda: KNNWindowInferer(roi_num_points=16, seed=0), id="knn_window"),
        pytest.param(lambda: VoxelPartitionInferer(voxel_size=0.5, seed=0), id="voxel_partition"),
        pytest.param(lambda: PotentialSphereInferer(radius=1.0, num_votes=1.0, seed=0), id="potential_sphere"),
    ],
)
def test_seeded_inferer_draws_anew_on_each_call_and_replays_from_a_fresh_instance(
    inferer_fn: Callable[[], Inferer],
) -> None:
    """A seeded instance feeds the predictor different fragments on its second call, and a fresh instance built
    with the same seed replays both calls exactly."""
    g = torch.Generator().manual_seed(0)
    data: Dict[str, Any] = {
        DataKeys.POS: torch.randn(256, 3, generator=g),
        DataKeys.BATCH: torch.zeros(256, dtype=torch.long),
    }

    def run(inferer: Inferer) -> List[Tensor]:
        seen: List[Tensor] = []

        def predictor(window: Dict[str, Any]) -> Tensor:
            seen.append(window[DataKeys.POS].clone())
            return torch.zeros(window[DataKeys.POS].size(0), 2)

        inferer(data, predictor=predictor)
        return seen

    inferer = inferer_fn()
    first, second = run(inferer), run(inferer)
    assert len(first) != len(second) or any(not torch.equal(a, b) for a, b in zip(first, second))

    replay = inferer_fn()
    replay_first, replay_second = run(replay), run(replay)
    assert len(first) == len(replay_first) and all(torch.equal(a, b) for a, b in zip(first, replay_first))
    assert len(second) == len(replay_second) and all(torch.equal(a, b) for a, b in zip(second, replay_second))


@pytest.mark.parametrize(
    "inferer_fn",
    [
        pytest.param(lambda: SlidingWindowInferer(block_size=4.0, roi_num_points=16), id="sliding_window"),
        pytest.param(lambda: KNNWindowInferer(roi_num_points=16), id="knn_window"),
        pytest.param(lambda: VoxelPartitionInferer(voxel_size=0.5), id="voxel_partition"),
        pytest.param(lambda: PotentialSphereInferer(radius=1.0, num_votes=1.0), id="potential_sphere"),
    ],
)
def test_unseeded_inferer_follows_the_global_generator(inferer_fn: Callable[[], Inferer]) -> None:
    """With `seed=None` the draws come from the global generator: successive calls differ, and re-seeding it with
    `torch.manual_seed` replays them."""
    g = torch.Generator().manual_seed(0)
    data: Dict[str, Any] = {
        DataKeys.POS: torch.randn(256, 3, generator=g),
        DataKeys.BATCH: torch.zeros(256, dtype=torch.long),
    }

    def run(inferer: Inferer) -> List[Tensor]:
        seen: List[Tensor] = []

        def predictor(window: Dict[str, Any]) -> Tensor:
            seen.append(window[DataKeys.POS].clone())
            return torch.zeros(window[DataKeys.POS].size(0), 2)

        inferer(data, predictor=predictor)
        return seen

    inferer = inferer_fn()
    torch.manual_seed(0)
    first, second = run(inferer), run(inferer)
    assert len(first) != len(second) or any(not torch.equal(a, b) for a, b in zip(first, second))

    torch.manual_seed(0)
    replay_first = run(inferer)
    assert len(first) == len(replay_first) and all(torch.equal(a, b) for a, b in zip(first, replay_first))
