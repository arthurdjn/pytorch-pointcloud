from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset

import torch_pointcloud.transforms as T
from torch_pointcloud.datasets import MixDataset
from torch_pointcloud.utils.data import PointCloudDataLoader


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
    mixed = MixDataset(dataset, mix=T.Mix3D(keys=("pos", "segment"), instance_key=None), seed=1)
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
    """Partner values drawn by a freshly seeded MixDataset."""
    mixed = MixDataset(
        _ScenesDataset(scenes),
        mix=lambda data, other: {"other": other["pos"]},
        seed=0,
    )
    return [mixed[0]["other"][0, 0].item() for _ in range(draws)]


def _partner(data: Dict[str, Any], other: Dict[str, Any]) -> Dict[str, Any]:
    return {"partner": other["pos"][0, 0].item()}


def _single(sample: Dict[str, Any]) -> Dict[str, Any]:
    return sample


def test_mix_dataset_seeded_workers_draw_distinct_partners() -> None:
    """Every worker gets a copy of the seeded dataset; the loader re-seeds each copy so their partners differ."""
    scenes = [{"pos": torch.full((1, 3), float(i))} for i in range(64)]
    mixed = MixDataset(_ScenesDataset(scenes), mix=_partner, seed=0)
    torch.manual_seed(0)
    partners = [
        sample["partner"] for sample in PointCloudDataLoader(mixed, batch_size=None, num_workers=2, collate_fn=_single)
    ]
    assert partners[0::2] != partners[1::2]


def test_mix_dataset_seed_reproducible_in_main_process() -> None:
    """Outside DataLoader workers, an identical seed yields the same partner stream."""
    scenes = [{"pos": torch.full((1, 3), float(i))} for i in range(64)]
    assert _partner_stream(scenes) == _partner_stream(scenes)
