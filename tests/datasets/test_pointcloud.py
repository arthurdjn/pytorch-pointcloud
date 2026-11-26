import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import Mock

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


class DummyCachedDataset(DummyPointCloudDataset):
    def __init__(self, root: Path, complete: bool = True) -> None:
        super().__init__(root)
        self.complete = complete
        self.stamps_during_build: List[Dict[str, Any]] = []

    def processed_files_exist(self) -> bool:
        return Path(self.processed_dir, "data.txt").exists()

    def write_processed_files(self, num_workers: Optional[int] = None, show_progress: bool = True) -> bool:
        self.stamps_during_build.append(json.loads(Path(self.cache_meta_paths[0]).read_text()))
        Path(self.processed_dir, "data.txt").write_text("data")
        return self.complete


def test_pointcloud_dataset_process_marks_the_cache_incomplete_until_it_is_built(tmp_path: Path) -> None:
    dataset = DummyCachedDataset(root=tmp_path)
    Path(dataset.raw_dir).mkdir(parents=True)
    dataset.process()

    assert dataset.stamps_during_build == [{"format_version": 1, "complete": False}]
    assert json.loads(Path(dataset.cache_meta_paths[0]).read_text()) == {"format_version": 1}

    dataset.process()
    assert len(dataset.stamps_during_build) == 1


def test_pointcloud_dataset_partial_build_is_stamped_incomplete_and_rebuilt(tmp_path: Path) -> None:
    dataset = DummyCachedDataset(root=tmp_path, complete=False)
    Path(dataset.raw_dir).mkdir(parents=True)
    dataset.process()
    assert json.loads(Path(dataset.cache_meta_paths[0]).read_text()) == {"format_version": 1, "complete": False}

    dataset.complete = True
    dataset.process()
    assert len(dataset.stamps_during_build) == 2
    assert json.loads(Path(dataset.cache_meta_paths[0]).read_text()) == {"format_version": 1}


def test_pointcloud_dataset_interrupted_rebuild_is_rebuilt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = DummyCachedDataset(root=tmp_path)
    Path(dataset.raw_dir).mkdir(parents=True)
    dataset.process()

    with monkeypatch.context() as context:
        context.setattr(dataset, "write_processed_files", Mock(side_effect=KeyboardInterrupt))
        with pytest.raises(KeyboardInterrupt):
            dataset.process(force=True)
    assert dataset.processed_files_exist()
    assert json.loads(Path(dataset.cache_meta_paths[0]).read_text()) == {"format_version": 1, "complete": False}

    dataset.process()
    assert len(dataset.stamps_during_build) == 2
    assert json.loads(Path(dataset.cache_meta_paths[0]).read_text()) == {"format_version": 1}


def test_pointcloud_dataset_cache_without_stamp_is_served(tmp_path: Path) -> None:
    dataset = DummyCachedDataset(root=tmp_path)
    Path(dataset.raw_dir).mkdir(parents=True)
    Path(dataset.processed_dir).mkdir(parents=True)
    Path(dataset.processed_dir, "data.txt").write_text("legacy")

    dataset.process()
    assert dataset.stamps_during_build == []
    assert not Path(dataset.cache_meta_paths[0]).exists()
    assert Path(dataset.processed_dir, "data.txt").read_text() == "legacy"


def test_pointcloud_dataset_forced_rebuild_without_raw_files_keeps_the_stamp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = DummyCachedDataset(root=tmp_path)
    Path(dataset.raw_dir).mkdir(parents=True)
    dataset.process()
    Path(dataset.raw_dir).rmdir()

    monkeypatch.setattr(dataset, "write_processed_files", Mock(side_effect=RuntimeError("Dataset not found")))
    with pytest.raises(RuntimeError, match="Dataset not found"):
        dataset.process(force=True)
    assert json.loads(Path(dataset.cache_meta_paths[0]).read_text()) == {"format_version": 1}
