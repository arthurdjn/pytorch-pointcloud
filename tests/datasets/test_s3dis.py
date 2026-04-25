# mypy: disable-error-code="arg-type,call-overload,attr-defined"
from pathlib import Path
from typing import Callable
from unittest.mock import Mock, patch

import pytest
import torch

from torch_pointcloud.datasets import S3DIS, S3DISHdf5
from torch_pointcloud.datasets.s3dis import S3DIS_CLASSES, S3DIS_UNK_IDX, load_s3dis_room


def test_load_s3dis_room(data_dir: Path) -> None:
    """Test that the S3DIS room data is loaded correctly"""
    room_dir = data_dir / "S3DIS" / "raw" / "Area_1" / "conferenceRoom_1"
    data = load_s3dis_room(room_dir)

    assert isinstance(data["pos"], torch.Tensor)
    assert isinstance(data["color"], torch.Tensor)
    assert isinstance(data["instance"], torch.Tensor)
    assert isinstance(data["segment"], torch.Tensor)
    assert data["pos"].shape[1] == 3
    assert data["color"].shape[1] == 3
    assert data["instance"].ndim == 1
    assert data["segment"].ndim == 1


def test_s3dis_dataset_not_found() -> None:
    """Raises an error if the dataset is not found"""
    with pytest.raises(RuntimeError, match="Dataset not found"):
        _ = S3DIS(root="not-found", show_progress=False)


@pytest.mark.parametrize("areas", [["Area_1"], ["Area_1", "Area_2"], "all"])
def test_s3dis_dataset_raw_files_exist(data_dir_factory: Callable[..., Path], areas: str | list[str]) -> None:
    """Test that the raw files exist"""
    data_dir = data_dir_factory("S3DIS/raw/**/*")
    dataset = S3DIS(root=data_dir, areas=areas, show_progress=False)
    assert dataset.raw_files_exist()


@pytest.mark.parametrize("areas", [["Area_1"], ["Area_1", "Area_2"], "all"])
def test_s3dis_dataset_raw_files_not_exist(areas: str | list[str]) -> None:
    """Test that an error is raised if the raw files do not exist"""
    with pytest.raises(RuntimeError, match="Dataset not found"):
        _ = S3DIS(root="not-found", areas=areas, show_progress=False)


@pytest.mark.parametrize("areas", [["Area_1"], ["Area_1", "Area_2"], "all"])
def test_s3dis_dataset_processed_files_exist(data_dir_factory: Callable[..., Path], areas: str | list[str]) -> None:
    """Test that the processed files exist"""
    data_dir = data_dir_factory("S3DIS/processed_aligned/**/*")
    dataset = S3DIS(root=data_dir, areas=areas, show_progress=False)
    assert dataset.processed_files_exist()


@pytest.mark.parametrize("areas", [["Area_1"], ["Area_1", "Area_2"], "all"])
@patch("torch_pointcloud.datasets.s3dis.load_s3dis_room", wraps=load_s3dis_room)
def test_s3dis_dataset_split(
    mock_load: Mock,
    data_dir_factory: Callable[..., Path],
    areas: str | list[str],
) -> None:
    """Test that the dataset does not load raw data if the processed data exists"""
    data_dir = data_dir_factory("S3DIS/processed_aligned/**/*")

    dataset = S3DIS(root=data_dir, areas=areas, show_progress=False)
    assert len(dataset) > 0
    _ = list(dataset)

    assert mock_load.call_count == 0


@pytest.mark.parametrize("areas", [["Area_1"], ["Area_1", "Area_2"], "all"])
@patch("torch_pointcloud.datasets.s3dis.load_s3dis_room", wraps=load_s3dis_room)
def test_s3dis_dataset_process_split(
    mock_load: Mock,
    data_dir_factory: Callable[..., Path],
    areas: str | list[str],
) -> None:
    """Test that the dataset loads raw data if the processed data does not exist"""
    data_dir = data_dir_factory("S3DIS/raw/**/*")

    dataset = S3DIS(root=data_dir, areas=areas, show_progress=False)
    assert len(dataset) > 0
    _ = list(dataset)

    assert mock_load.call_count > 0


def test_s3dis_dataset_progress(data_dir_factory: Callable[..., Path], capsys: pytest.CaptureFixture[str]) -> None:
    """Test that the dataset displays a progress bar during processing"""
    data_dir = data_dir_factory("S3DIS/raw/**/*")

    dataset = S3DIS(root=data_dir, show_progress=True)
    assert len(dataset) > 0
    captured = capsys.readouterr()
    assert "Processing" in captured.err
    assert captured.out == ""


def test_s3dis_dataset_without_progress(
    data_dir_factory: Callable[..., Path], capsys: pytest.CaptureFixture[str]
) -> None:
    """Test that the dataset does not display a progress bar during processing"""
    data_dir = data_dir_factory("S3DIS/raw/**/*")

    dataset = S3DIS(root=data_dir, show_progress=False)
    assert len(dataset) > 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == ""


def test_s3dis_dataset_progress_with_cached_processed(
    data_dir_factory: Callable[..., Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test that no progress bar is shown if the processed dataset already exists"""
    data_dir = data_dir_factory("S3DIS/processed_aligned/**/*")

    dataset = S3DIS(root=data_dir, show_progress=True)
    assert len(dataset) > 0
    captured = capsys.readouterr()
    assert "Processing" not in captured.err
    assert captured.out == ""


@pytest.mark.parametrize("classes", [["wall"], ["wall", "floor", "ceiling"]])
def test_s3dis_dataset_classes(data_dir_factory: Callable[..., Path], classes: list[str]) -> None:
    """Test that the dataset loads specific classes"""
    data_dir = data_dir_factory("S3DIS/raw/**/*")

    dataset = S3DIS(root=data_dir, classes=classes, show_progress=False)
    assert len(dataset) > 0
    assert all(cls in dataset.classes for cls in classes)

    class_ids = set([*dataset.class_to_idx.values(), S3DIS_UNK_IDX])
    for data in dataset:
        labels = data["segment"].unique().tolist()
        assert set(labels).issubset(class_ids)


def test_s3dis_dataset_all_classes(data_dir_factory: Callable[..., Path]) -> None:
    """Test that the dataset loads all classes"""
    data_dir = data_dir_factory("S3DIS/raw/**/*")

    dataset = S3DIS(root=data_dir, classes="all", show_progress=False)
    assert len(dataset) > 0


def test_s3dis_dataset_transform(data_dir_factory: Callable[..., Path]) -> None:
    """Test that the dataset is transformed correctly after being processed"""
    data_dir = data_dir_factory("S3DIS/processed_aligned/**/*")

    transform = Mock(side_effect=lambda data: data)
    dataset = S3DIS(root=data_dir, transform=transform, show_progress=False)
    _ = list(dataset)
    assert transform.call_count == len(dataset)


def test_s3dis_tile_blocks(data_dir_factory: Callable[..., Path]) -> None:
    """Test that tile_blocks splits rooms into fixed-size blocks"""
    data_dir = data_dir_factory("S3DIS/processed_aligned/**/*")

    dataset_rooms = S3DIS(root=data_dir, show_progress=False)
    num_rooms = len(dataset_rooms)

    dataset_blocks = S3DIS(
        root=data_dir,
        block_size=1.0,
        block_stride=1.0,
        num_nodes=64,
        min_num_nodes=1,
        show_progress=False,
    )

    assert len(dataset_blocks) >= num_rooms

    for data in dataset_blocks:
        assert data["pos"].shape == (64, 3)
        assert data["segment"].shape == (64,)
        assert "room_max" in data


def test_s3dis_tile_blocks_preserves_cache(data_dir_factory: Callable[..., Path]) -> None:
    """Test that changing tile_blocks does not require reprocessing"""
    data_dir = data_dir_factory("S3DIS/processed_aligned/**/*")

    dataset_a = S3DIS(
        root=data_dir,
        block_size=1.0,
        block_stride=1.0,
        num_nodes=64,
        min_num_nodes=1,
        show_progress=False,
    )
    dataset_b = S3DIS(
        root=data_dir,
        block_size=2.0,
        block_stride=2.0,
        num_nodes=32,
        min_num_nodes=1,
        show_progress=False,
    )

    assert dataset_a.processed_files_exist()
    assert dataset_b.processed_files_exist()
    assert len(dataset_a) != len(dataset_b) or dataset_a[0]["pos"].shape != dataset_b[0]["pos"].shape


# ---------------------------------------------------------------------------
# S3DISHdf5
# ---------------------------------------------------------------------------

HDF5_GLOB = "S3DIS/indoor3d_sem_seg_hdf5_data/**/*"


def test_s3dis_hdf5_not_found() -> None:
    """Raises an error if the dataset is not found"""
    with pytest.raises(RuntimeError, match="Dataset not found"):
        _ = S3DISHdf5(root="not-found", download=False, show_progress=False)


def test_s3dis_hdf5_load(data_dir_factory: Callable[..., Path]) -> None:
    """Test that the HDF5 dataset loads and returns samples with correct keys and shapes"""
    data_dir = data_dir_factory(HDF5_GLOB)

    dataset = S3DISHdf5(root=data_dir, download=False, show_progress=False)
    assert len(dataset) > 0

    sample = dataset[0]
    assert sample["pos"].shape == (4096, 3)
    assert sample["color"].shape == (4096, 3)
    assert sample["norm_pos"].shape == (4096, 3)
    assert sample["segment"].shape == (4096,)


@pytest.mark.parametrize("areas", [["Area_1"], ["Area_5"], ["Area_1", "Area_2"]])
def test_s3dis_hdf5_area_filter(data_dir_factory: Callable[..., Path], areas: list[str]) -> None:
    """Test that area filtering returns only blocks from the requested areas"""
    data_dir = data_dir_factory(HDF5_GLOB)

    dataset_all = S3DISHdf5(root=data_dir, areas="all", download=False, show_progress=False)
    dataset_sub = S3DISHdf5(root=data_dir, areas=areas, download=False, show_progress=False)

    assert 0 < len(dataset_sub) < len(dataset_all)


def test_s3dis_hdf5_all_areas(data_dir_factory: Callable[..., Path]) -> None:
    """Test that loading all areas returns data from every area"""
    data_dir = data_dir_factory(HDF5_GLOB)

    dataset = S3DISHdf5(root=data_dir, areas="all", download=False, show_progress=False)
    assert len(dataset) > 0


def test_s3dis_hdf5_transform(data_dir_factory: Callable[..., Path]) -> None:
    """Test that the transform is called for every sample"""
    data_dir = data_dir_factory(HDF5_GLOB)

    transform = Mock(side_effect=lambda data: data)
    dataset = S3DISHdf5(root=data_dir, transform=transform, download=False, show_progress=False)
    _ = list(dataset)
    assert transform.call_count == len(dataset)


def test_s3dis_hdf5_segment_labels_valid(data_dir_factory: Callable[..., Path]) -> None:
    """Test that segment labels are within the valid class range"""
    data_dir = data_dir_factory(HDF5_GLOB)

    dataset = S3DISHdf5(root=data_dir, download=False, show_progress=False)
    num_classes = len(S3DIS_CLASSES)

    for sample in dataset:
        labels = sample["segment"]
        assert labels.min() >= 0
        assert labels.max() < num_classes


def test_s3dis_hdf5_tensor_dtypes(data_dir_factory: Callable[..., Path]) -> None:
    """Test that returned tensors have expected dtypes"""
    data_dir = data_dir_factory(HDF5_GLOB)

    dataset = S3DISHdf5(root=data_dir, download=False, show_progress=False)
    sample = dataset[0]

    assert sample["pos"].dtype == torch.float32
    assert sample["color"].dtype == torch.float32
    assert sample["norm_pos"].dtype == torch.float32
    assert sample["segment"].dtype == torch.int64
