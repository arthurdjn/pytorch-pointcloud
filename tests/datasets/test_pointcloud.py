from pathlib import Path

import pytest

from torch_pointcloud.datasets import PointCloudDataset


class DummyPointCloudDataset(PointCloudDataset):
    def __len__(self) -> int:
        return 3

    def __getitem__(self, index: int) -> int:
        return index


def test_pointcloud_dataset_directory_layout(tmp_path: Path) -> None:
    dataset = DummyPointCloudDataset(root=tmp_path)
    assert dataset.name == "DummyPointCloudDataset"
    assert dataset.data_dir == (tmp_path / "DummyPointCloudDataset").absolute().as_posix()
    assert dataset.raw_dir == (tmp_path / "DummyPointCloudDataset" / "raw").absolute().as_posix()
    assert dataset.processed_dir == (tmp_path / "DummyPointCloudDataset" / "processed").absolute().as_posix()


def test_pointcloud_dataset_existence_checks_follow_directories(tmp_path: Path) -> None:
    dataset = DummyPointCloudDataset(root=tmp_path)
    assert not dataset.raw_files_exist()
    assert not dataset.processed_files_exist()

    Path(dataset.raw_dir).mkdir(parents=True)
    Path(dataset.processed_dir).mkdir(parents=True)
    assert dataset.raw_files_exist()
    assert dataset.processed_files_exist()


def test_pointcloud_dataset_base_getitem_and_len_raise(tmp_path: Path) -> None:
    dataset = PointCloudDataset(root=tmp_path)
    with pytest.raises(NotImplementedError):
        _ = dataset[0]
    with pytest.raises(NotImplementedError):
        _ = len(dataset)


def test_pointcloud_dataset_repr_contains_name_and_length(tmp_path: Path) -> None:
    dataset = DummyPointCloudDataset(root=tmp_path)
    assert "DummyPointCloudDataset" in repr(dataset)
    assert "3" in repr(dataset)
