from types import SimpleNamespace
from typing import Any, Dict, List

import pytest
import torch
from torch.utils.data import Dataset

import torch_pointcloud.transforms as T
from torch_pointcloud.datasets import MixDataset


class _ScenesDataset(Dataset):
    def __init__(self, scenes: List[Dict[str, Any]]) -> None:
        self.scenes = scenes

    def __len__(self) -> int:
        return len(self.scenes)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return {key: value.clone() for key, value in self.scenes[index].items()}


def _dataset() -> _ScenesDataset:
    g = torch.Generator().manual_seed(0)
    scenes = [
        {"pos": torch.randn(10 + i, 3, generator=g), "segment": torch.randint(0, 5, (10 + i,), generator=g)}
        for i in range(4)
    ]
    return _ScenesDataset(scenes)


def test_mix_dataset_len_matches_wrapped() -> None:
    dataset = _dataset()
    mixed = MixDataset(dataset, mix=T.Mix3D(keys=("pos", "segment"), instance_key=None))
    assert len(mixed) == len(dataset)


def test_mix_dataset_applies_mix() -> None:
    """A drawn partner is concatenated, so the mixed sample is longer than the source sample."""
    dataset = _dataset()
    g = torch.Generator().manual_seed(1)
    mixed = MixDataset(dataset, mix=T.Mix3D(keys=("pos", "segment"), instance_key=None), generator=g)
    out = mixed[0]
    assert out["pos"].shape[0] > dataset[0]["pos"].shape[0]
    assert out["pos"].shape[0] == out["segment"].shape[0]


def test_mix_dataset_p_zero_returns_source() -> None:
    dataset = _dataset()
    mixed = MixDataset(dataset, mix=T.Mix3D(keys=("pos", "segment"), instance_key=None, p=0.0))
    out = mixed[2]
    assert torch.equal(out["pos"], dataset[2]["pos"])


def test_mix_dataset_post_transform_runs() -> None:
    dataset = _dataset()
    mixed = MixDataset(
        dataset,
        mix=T.Mix3D(keys=("pos", "segment"), instance_key=None),
        transform=T.Scale(keys="pos", scale=0.0),
    )
    out = mixed[0]
    assert torch.equal(out["pos"], torch.zeros_like(out["pos"]))


def _partner_stream(scenes: List[Dict[str, Any]], draws: int = 16) -> List[float]:
    """Partner values drawn by a fresh MixDataset replica whose generator is seeded like every worker's."""
    mixed = MixDataset(
        _ScenesDataset(scenes),
        mix=lambda data, other: {"other": other["pos"]},
        generator=torch.Generator().manual_seed(0),
    )
    return [mixed[0]["other"][0, 0].item() for _ in range(draws)]


def test_mix_dataset_worker_streams_are_distinct(monkeypatch: pytest.MonkeyPatch) -> None:
    """DataLoader workers hold identical generator replicas, so each must derive its own partner stream."""
    scenes = [{"pos": torch.full((1, 3), float(i))} for i in range(64)]

    def stream(worker_id: int) -> List[float]:
        monkeypatch.setattr("torch_pointcloud.datasets.mix.get_worker_info", lambda: SimpleNamespace(id=worker_id))
        return _partner_stream(scenes)

    assert stream(0) == stream(0)
    assert stream(1) == stream(1)
    assert stream(0) != stream(1)


def test_mix_dataset_generator_reproducible_in_main_process() -> None:
    """Outside DataLoader workers, an identically seeded generator yields the same partner stream."""
    scenes = [{"pos": torch.full((1, 3), float(i))} for i in range(64)]
    assert _partner_stream(scenes) == _partner_stream(scenes)
