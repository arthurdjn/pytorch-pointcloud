import pytest
from torch.utils.data import Dataset

from torch_pointcloud.datasets import RepeatDataset


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


def test_repeat_dataset_draws_each_sample_its_own_number_of_times() -> None:
    dataset = RepeatDataset(list(range(4)), k=[2, 0, 3, 1])  # type: ignore[arg-type]
    assert len(dataset) == 6
    assert [dataset[i] for i in range(6)] == [0, 0, 2, 2, 2, 3]
    with pytest.raises(IndexError):
        dataset[6]
    assert len(RepeatDataset(list(range(2)), k=[0, 0])) == 0  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="counts"):
        RepeatDataset(list(range(3)), k=[1, 2])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-negative"):
        RepeatDataset(list(range(2)), k=[1, -1])  # type: ignore[arg-type]
