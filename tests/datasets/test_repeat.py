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


def test_repeat_dataset_len_multiplies_by_loop() -> None:
    dataset = RepeatDataset(DummyIndexDataset(4), loop=3)
    assert len(dataset) == 12


def test_repeat_dataset_loop_one_is_identity() -> None:
    dataset = RepeatDataset(DummyIndexDataset(5), loop=1)
    assert len(dataset) == 5
    assert [dataset[i] for i in range(5)] == [0, 1, 2, 3, 4]


def test_repeat_dataset_index_wraps_around_base_dataset() -> None:
    dataset = RepeatDataset(DummyIndexDataset(3), loop=2)
    assert [dataset[i] for i in range(6)] == [0, 1, 2, 0, 1, 2]
