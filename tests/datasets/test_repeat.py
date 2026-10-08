import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from torch_pointcloud.datasets import RepeatDataset, RepeatSampler


class DummyIndexDataset(Dataset):
    def __init__(self, size: int) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> int:
        if not 0 <= index < self.size:
            raise IndexError(index)
        return index


def test_repeat_dataset_len_multiplies_by_k() -> None:
    dataset = RepeatDataset(DummyIndexDataset(4), k=3)
    assert len(dataset) == 12


def test_repeat_dataset_k_one_is_identity() -> None:
    dataset = RepeatDataset(DummyIndexDataset(5), k=1)
    assert len(dataset) == 5
    assert [dataset[i] for i in range(5)] == [0, 1, 2, 3, 4]


def test_repeat_dataset_index_wraps_around_base_dataset() -> None:
    dataset = RepeatDataset(DummyIndexDataset(3), k=2)
    assert [dataset[i] for i in range(6)] == [0, 1, 2, 0, 1, 2]


def test_repeat_sampler_draws_each_sample_its_own_number_of_times() -> None:
    assert list(RepeatSampler([2, 0, 3, 1], shuffle=False)) == [0, 0, 2, 2, 2, 3]
    assert len(RepeatSampler([2, 0, 3, 1])) == 6
    assert list(RepeatSampler([])) == []
    with pytest.raises(ValueError, match="non-negative"):
        RepeatSampler([1, -1])


def test_repeat_sampler_shuffles_the_same_draws() -> None:
    sampler = RepeatSampler([2, 0, 3, 1], generator=torch.Generator().manual_seed(0))
    first, second = list(sampler), list(sampler)
    assert sorted(first) == sorted(second) == [0, 0, 2, 2, 2, 3]
    assert first != [0, 0, 2, 2, 2, 3] or second != [0, 0, 2, 2, 2, 3]


def test_repeat_sampler_drives_a_dataloader() -> None:
    loader = DataLoader(DummyIndexDataset(4), batch_size=4, sampler=RepeatSampler([2, 0, 3, 1], shuffle=False))
    assert [batch.tolist() for batch in loader] == [[0, 0, 2, 2], [2, 3]]
