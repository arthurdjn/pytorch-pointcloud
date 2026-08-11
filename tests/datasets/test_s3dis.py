# mypy: disable-error-code="arg-type,call-overload,attr-defined"
from pathlib import Path
from typing import Any, Callable, Iterator
from unittest.mock import Mock, patch

import pytest
import torch

from torch_pointcloud.datasets import S3DIS, S3DISHdf5
from torch_pointcloud.datasets.s3dis import S3DIS_CLASSES, load_s3dis_room


def test_load_s3dis_room(datasets_dir: Path) -> None:
    """Test that the S3DIS room data is loaded correctly"""
    room_dir = datasets_dir / "S3DIS" / "raw" / "Area_1" / "conferenceRoom_1"
    data = load_s3dis_room(room_dir)

    assert isinstance(data["pos"], torch.Tensor)
    assert isinstance(data["color"], torch.Tensor)
    assert isinstance(data["instance"], torch.Tensor)
    assert isinstance(data["segment"], torch.Tensor)
    assert data["pos"].shape[1] == 3
    assert data["color"].shape[1] == 3
    assert data["instance"].ndim == 1
    assert data["segment"].ndim == 1


def test_load_s3dis_room_order_independent(datasets_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Instance ids and per-point labels do not depend on the filesystem ordering of annotation files"""
    room_dir = datasets_dir / "S3DIS" / "raw" / "Area_1" / "conferenceRoom_1"
    data = load_s3dis_room(room_dir)

    original_rglob = Path.rglob

    def reversed_rglob(self: Path, pattern: str) -> Iterator[Path]:
        return reversed(list(original_rglob(self, pattern)))

    monkeypatch.setattr(Path, "rglob", reversed_rglob)
    data_reversed = load_s3dis_room(room_dir)

    assert torch.equal(data["pos"], data_reversed["pos"])
    assert torch.equal(data["segment"], data_reversed["segment"])
    assert torch.equal(data["instance"], data_reversed["instance"])


def test_s3dis_dataset_not_found() -> None:
    """Raises an error if the dataset is not found"""
    with pytest.raises(RuntimeError, match="Dataset not found"):
        _ = S3DIS(root="not-found", show_progress=False)


@pytest.mark.parametrize("areas", [["Area_1"], ["Area_1", "Area_2"], "all"])
def test_s3dis_dataset_raw_files_exist(datasets_dir_factory: Callable[..., Path], areas: str | list[str]) -> None:
    """Test that the raw files exist"""
    datasets_dir = datasets_dir_factory("S3DIS/raw/**/*")
    dataset = S3DIS(root=datasets_dir, areas=areas, show_progress=False)
    assert dataset.raw_files_exist()


@pytest.mark.parametrize("areas", [["Area_1"], ["Area_1", "Area_2"], "all"])
def test_s3dis_dataset_raw_files_not_exist(areas: str | list[str]) -> None:
    """Test that an error is raised if the raw files do not exist"""
    with pytest.raises(RuntimeError, match="Dataset not found"):
        _ = S3DIS(root="not-found", areas=areas, show_progress=False)


@pytest.mark.parametrize("areas", [["Area_1"], ["Area_1", "Area_2"], "all"])
def test_s3dis_dataset_processed_files_exist(datasets_dir_factory: Callable[..., Path], areas: str | list[str]) -> None:
    """Test that the processed files exist"""
    datasets_dir = datasets_dir_factory("S3DIS/processed_aligned/**/*")
    dataset = S3DIS(root=datasets_dir, areas=areas, show_progress=False)
    assert dataset.processed_files_exist()


@pytest.mark.parametrize("areas", [["Area_1"], ["Area_1", "Area_2"], "all"])
@patch("torch_pointcloud.datasets.s3dis.load_s3dis_room", wraps=load_s3dis_room)
def test_s3dis_dataset_split(
    mock_load: Mock,
    datasets_dir_factory: Callable[..., Path],
    areas: str | list[str],
) -> None:
    """Test that the dataset does not load raw data if the processed data exists"""
    datasets_dir = datasets_dir_factory("S3DIS/processed_aligned/**/*")

    dataset = S3DIS(root=datasets_dir, areas=areas, show_progress=False)
    assert len(dataset) > 0
    _ = list(dataset)

    assert mock_load.call_count == 0


@pytest.mark.parametrize("areas", [["Area_1"], ["Area_1", "Area_2"], "all"])
@patch("torch_pointcloud.datasets.s3dis.load_s3dis_room", wraps=load_s3dis_room)
def test_s3dis_dataset_process_split(
    mock_load: Mock,
    datasets_dir_factory: Callable[..., Path],
    areas: str | list[str],
) -> None:
    """Test that the dataset loads raw data if the processed data does not exist"""
    datasets_dir = datasets_dir_factory("S3DIS/raw/**/*")

    dataset = S3DIS(root=datasets_dir, areas=areas, show_progress=False)
    assert len(dataset) > 0
    _ = list(dataset)

    assert mock_load.call_count > 0


def test_s3dis_dataset_progress(datasets_dir_factory: Callable[..., Path], capsys: pytest.CaptureFixture[str]) -> None:
    """Test that the dataset displays a progress bar during processing"""
    datasets_dir = datasets_dir_factory("S3DIS/raw/**/*")

    dataset = S3DIS(root=datasets_dir, show_progress=True)
    assert len(dataset) > 0
    captured = capsys.readouterr()
    assert "Processing" in captured.err
    assert captured.out == ""


def test_s3dis_dataset_without_progress(
    datasets_dir_factory: Callable[..., Path], capsys: pytest.CaptureFixture[str]
) -> None:
    """Test that the dataset does not display a progress bar during processing"""
    datasets_dir = datasets_dir_factory("S3DIS/raw/**/*")

    dataset = S3DIS(root=datasets_dir, show_progress=False)
    assert len(dataset) > 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == ""


def test_s3dis_dataset_progress_with_cached_processed(
    datasets_dir_factory: Callable[..., Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test that no progress bar is shown if the processed dataset already exists"""
    datasets_dir = datasets_dir_factory("S3DIS/processed_aligned/**/*")

    dataset = S3DIS(root=datasets_dir, show_progress=True)
    assert len(dataset) > 0
    captured = capsys.readouterr()
    assert "Processing" not in captured.err
    assert captured.out == ""


@pytest.mark.parametrize("classes", [["wall"], ["wall", "floor", "ceiling"]])
def test_s3dis_dataset_classes(datasets_dir_factory: Callable[..., Path], classes: list[str]) -> None:
    """Test that the dataset loads specific classes"""
    datasets_dir = datasets_dir_factory("S3DIS/raw/**/*")

    dataset = S3DIS(root=datasets_dir, classes=classes, show_progress=False)
    assert len(dataset) > 0
    assert all(cls in dataset.classes for cls in classes)

    class_ids = set([*dataset.class_to_idx.values(), -1])
    for data in dataset:
        labels = data["segment"].unique().tolist()
        assert set(labels).issubset(class_ids)


def test_s3dis_dataset_class_subset_without_clutter_uses_ignore_index(
    datasets_dir_factory: Callable[..., Path],
) -> None:
    """Unselected classes map to the ignore index -1 when 'clutter' is not selected"""
    datasets_dir = datasets_dir_factory("S3DIS/raw/**/*")

    dataset = S3DIS(root=datasets_dir, classes=["wall", "floor"], show_progress=False)
    labels = torch.cat([data["segment"] for data in dataset])
    assert set(labels.unique().tolist()) <= {-1, 0, 1}
    assert (labels == -1).any()


def test_s3dis_dataset_class_subset_with_clutter_uses_clutter_index(
    datasets_dir_factory: Callable[..., Path],
) -> None:
    """Unselected classes map to the new index of 'clutter' when it is selected"""
    datasets_dir = datasets_dir_factory("S3DIS/raw/**/*")

    dataset = S3DIS(root=datasets_dir, classes=["wall", "clutter"], show_progress=False)
    clutter_idx = dataset.class_to_idx["clutter"]
    assert clutter_idx == 1

    labels = torch.cat([data["segment"] for data in dataset])
    assert set(labels.unique().tolist()) <= {0, clutter_idx}
    assert (labels == clutter_idx).any()


def test_s3dis_dataset_all_classes(datasets_dir_factory: Callable[..., Path]) -> None:
    """Test that the dataset loads all classes"""
    datasets_dir = datasets_dir_factory("S3DIS/raw/**/*")

    dataset = S3DIS(root=datasets_dir, classes="all", show_progress=False)
    assert len(dataset) > 0


def test_s3dis_dataset_transform(datasets_dir_factory: Callable[..., Path]) -> None:
    """Test that the dataset is transformed correctly after being processed"""
    datasets_dir = datasets_dir_factory("S3DIS/processed_aligned/**/*")

    transform = Mock(side_effect=lambda data: data)
    dataset = S3DIS(root=datasets_dir, transform=transform, show_progress=False)
    _ = list(dataset)
    assert transform.call_count == len(dataset)


def test_s3dis_tile_blocks(datasets_dir_factory: Callable[..., Path]) -> None:
    """Test that tile_blocks splits rooms into fixed-size blocks"""
    datasets_dir = datasets_dir_factory("S3DIS/processed_aligned/**/*")

    dataset_rooms = S3DIS(root=datasets_dir, show_progress=False)
    num_rooms = len(dataset_rooms)

    dataset_blocks = S3DIS(
        root=datasets_dir,
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


def test_s3dis_tile_blocks_preserves_cache(datasets_dir_factory: Callable[..., Path]) -> None:
    """Test that changing tile_blocks does not require reprocessing"""
    datasets_dir = datasets_dir_factory("S3DIS/processed_aligned/**/*")

    dataset_a = S3DIS(
        root=datasets_dir,
        block_size=1.0,
        block_stride=1.0,
        num_nodes=64,
        min_num_nodes=1,
        show_progress=False,
    )
    dataset_b = S3DIS(
        root=datasets_dir,
        block_size=2.0,
        block_stride=2.0,
        num_nodes=32,
        min_num_nodes=1,
        show_progress=False,
    )

    assert dataset_a.processed_files_exist()
    assert dataset_b.processed_files_exist()
    assert len(dataset_a) != len(dataset_b) or dataset_a[0]["pos"].shape != dataset_b[0]["pos"].shape


def test_s3dis_download_retries_corrupt_archive_with_overwrite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A corrupt archive is re-downloaded with `overwrite=True` and a second MD5 mismatch raises"""
    calls: list[tuple[str, Any]] = []

    def fake_download(url: str, file_path: Any = "", **kwargs: Any) -> str:
        calls.append((Path(file_path).name, kwargs.get("overwrite", False)))
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        Path(file_path).write_bytes(b"corrupt")
        return str(file_path)

    monkeypatch.setattr("torch_pointcloud.datasets.s3dis.download_url", fake_download)

    archive = tmp_path / "S3DIS" / "raw" / "Stanford3dDataset_v1.2_Aligned_Version.zip"
    archive.parent.mkdir(parents=True)
    archive.write_bytes(b"corrupt")

    with pytest.raises(RuntimeError, match="MD5"):
        _ = S3DIS(root=tmp_path, download=True, show_progress=False)

    assert ("Stanford3dDataset_v1.2_Aligned_Version.zip", True) in calls


def test_s3dis_dataset_leftover_archive_detected(datasets_dir_factory: Callable[..., Path]) -> None:
    """An archive without the extraction marker means an interrupted extraction, so the raw tree is rejected"""
    datasets_dir = datasets_dir_factory("S3DIS/raw/**/*")
    archive_path = datasets_dir / "S3DIS" / "raw" / "Stanford3dDataset_v1.2_Aligned_Version.zip"
    archive_path.write_bytes(b"partial")

    with pytest.raises(RuntimeError, match="Dataset not found"):
        _ = S3DIS(root=datasets_dir, areas=["Area_1"], show_progress=False)

    (datasets_dir / "S3DIS" / "raw" / ".extraction_complete").touch()
    dataset = S3DIS(root=datasets_dir, areas=["Area_1"], show_progress=False)
    assert len(dataset) > 0


def test_s3dis_download_marks_extraction_complete(
    datasets_dir_factory: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A successful download() ends by writing the extraction marker next to the kept archive"""

    def fake_download(url: str, file_path: Any = "", **kwargs: Any) -> str:
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        Path(file_path).write_bytes(b"archive")
        return str(file_path)

    def fake_extract(zip_path: Any, out_dir: Any, **kwargs: Any) -> str:
        ceiling_path = Path(out_dir, "Area_5", "hallway_6", "Annotations", "ceiling_1.txt")
        ceiling_path.parent.mkdir(parents=True, exist_ok=True)
        ceiling_path.write_text("0.0 0.0 0.0 0 0 0\n")
        return str(out_dir)

    datasets_dir = datasets_dir_factory("S3DIS/processed_aligned/**/*")
    dataset = S3DIS(root=datasets_dir, areas=["Area_1"], show_progress=False)
    monkeypatch.setattr("torch_pointcloud.datasets.s3dis.download_url", fake_download)
    monkeypatch.setattr("torch_pointcloud.datasets.s3dis.extract_zip", fake_extract)
    monkeypatch.setattr("torch_pointcloud.datasets.s3dis.is_hash_valid", lambda *args, **kwargs: True)

    dataset.download()

    assert (Path(dataset.raw_dir) / ".extraction_complete").exists()


def test_s3dis_hdf5_download_retries_corrupt_archive_with_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A corrupt HDF5 archive is re-downloaded with `overwrite=True` and a second MD5 mismatch raises"""
    calls: list[tuple[str, Any]] = []

    def fake_download(url: str, file_path: Any = "", **kwargs: Any) -> str:
        calls.append((Path(file_path).name, kwargs.get("overwrite", False)))
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        Path(file_path).write_bytes(b"corrupt")
        return str(file_path)

    monkeypatch.setattr("torch_pointcloud.datasets.s3dis.download_url", fake_download)

    with pytest.raises(RuntimeError, match="MD5"):
        _ = S3DISHdf5(root=tmp_path, download=True, show_progress=False)

    assert ("indoor3d_sem_seg_hdf5_data.zip", True) in calls


HDF5_GLOB = "S3DIS/indoor3d_sem_seg_hdf5_data/**/*"


def test_s3dis_hdf5_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing dataset raises without attempting a download (the default is `download=False`)"""
    download_mock = Mock()
    monkeypatch.setattr("torch_pointcloud.datasets.s3dis.download_url", download_mock)
    with pytest.raises(RuntimeError, match="Dataset not found"):
        _ = S3DISHdf5(root="not-found", show_progress=False)
    download_mock.assert_not_called()


def test_s3dis_hdf5_force_download_implies_download(
    datasets_dir_factory: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """`force_download=True` triggers the download even when `download` is left False"""
    datasets_dir = datasets_dir_factory(HDF5_GLOB)
    mock = Mock()
    monkeypatch.setattr(S3DISHdf5, "download", mock)

    _ = S3DISHdf5(root=datasets_dir, force_download=True, show_progress=False)
    mock.assert_called_once_with(force=True, show_progress=False)


def test_s3dis_hdf5_load(datasets_dir_factory: Callable[..., Path]) -> None:
    """Test that the HDF5 dataset loads and returns samples with correct keys and shapes"""
    datasets_dir = datasets_dir_factory(HDF5_GLOB)

    dataset = S3DISHdf5(root=datasets_dir, download=False, show_progress=False)
    assert len(dataset) > 0

    sample = dataset[0]
    assert sample["pos"].shape == (4096, 3)
    assert sample["color"].shape == (4096, 3)
    assert sample["norm_pos"].shape == (4096, 3)
    assert sample["segment"].shape == (4096,)


@pytest.mark.parametrize("areas", [["Area_1"], ["Area_5"], ["Area_1", "Area_2"]])
def test_s3dis_hdf5_area_filter(datasets_dir_factory: Callable[..., Path], areas: list[str]) -> None:
    """Test that area filtering returns only blocks from the requested areas"""
    datasets_dir = datasets_dir_factory(HDF5_GLOB)

    dataset_all = S3DISHdf5(root=datasets_dir, areas="all", download=False, show_progress=False)
    dataset_sub = S3DISHdf5(root=datasets_dir, areas=areas, download=False, show_progress=False)

    assert 0 < len(dataset_sub) < len(dataset_all)


def test_s3dis_hdf5_all_areas(datasets_dir_factory: Callable[..., Path]) -> None:
    """Test that loading all areas returns data from every area"""
    datasets_dir = datasets_dir_factory(HDF5_GLOB)

    dataset = S3DISHdf5(root=datasets_dir, areas="all", download=False, show_progress=False)
    assert len(dataset) > 0


def test_s3dis_hdf5_transform(datasets_dir_factory: Callable[..., Path]) -> None:
    """Test that the transform is called for every sample"""
    datasets_dir = datasets_dir_factory(HDF5_GLOB)

    transform = Mock(side_effect=lambda data: data)
    dataset = S3DISHdf5(root=datasets_dir, transform=transform, download=False, show_progress=False)
    _ = list(dataset)
    assert transform.call_count == len(dataset)


def test_s3dis_hdf5_segment_labels_valid(datasets_dir_factory: Callable[..., Path]) -> None:
    """Test that segment labels are within the valid class range"""
    datasets_dir = datasets_dir_factory(HDF5_GLOB)

    dataset = S3DISHdf5(root=datasets_dir, download=False, show_progress=False)
    num_classes = len(S3DIS_CLASSES)

    for sample in dataset:
        labels = sample["segment"]
        assert labels.min() >= 0
        assert labels.max() < num_classes


def test_s3dis_hdf5_tensor_dtypes(datasets_dir_factory: Callable[..., Path]) -> None:
    """Test that returned tensors have expected dtypes"""
    datasets_dir = datasets_dir_factory(HDF5_GLOB)

    dataset = S3DISHdf5(root=datasets_dir, download=False, show_progress=False)
    sample = dataset[0]

    assert sample["pos"].dtype == torch.float32
    assert sample["color"].dtype == torch.float32
    assert sample["norm_pos"].dtype == torch.float32
    assert sample["segment"].dtype == torch.int64


def test_s3dis_hdf5_class_subset_without_clutter_uses_ignore_index(
    datasets_dir_factory: Callable[..., Path],
) -> None:
    """Unselected classes map to the ignore index -1 when 'clutter' is not selected"""
    datasets_dir = datasets_dir_factory(HDF5_GLOB)

    dataset = S3DISHdf5(root=datasets_dir, classes=["wall", "floor"], download=False, show_progress=False)
    assert dataset.class_to_idx == {"wall": 0, "floor": 1}

    labels = torch.cat([sample["segment"] for sample in dataset])
    assert set(labels.unique().tolist()) <= {-1, 0, 1}
    assert (labels == -1).any()


def test_s3dis_hdf5_class_subset_with_clutter_uses_clutter_index(
    datasets_dir_factory: Callable[..., Path],
) -> None:
    """Unselected classes map to the new index of 'clutter' when it is selected"""
    datasets_dir = datasets_dir_factory(HDF5_GLOB)

    dataset = S3DISHdf5(root=datasets_dir, classes=["wall", "clutter"], download=False, show_progress=False)
    clutter_idx = dataset.class_to_idx["clutter"]
    assert clutter_idx == 1

    labels = torch.cat([sample["segment"] for sample in dataset])
    assert set(labels.unique().tolist()) <= {0, clutter_idx}
    assert (labels == clutter_idx).any()


def test_s3dis_hdf5_class_subset_matches_full_labels(datasets_dir_factory: Callable[..., Path]) -> None:
    """The remapped subset labels agree point-wise with the full 13-class labels"""
    datasets_dir = datasets_dir_factory(HDF5_GLOB)

    dataset_all = S3DISHdf5(root=datasets_dir, download=False, show_progress=False)
    dataset_sub = S3DISHdf5(root=datasets_dir, classes=["floor", "wall"], download=False, show_progress=False)

    full = dataset_all[0]["segment"]
    sub = dataset_sub[0]["segment"]
    floor_idx = S3DIS_CLASSES.index("floor")
    wall_idx = S3DIS_CLASSES.index("wall")
    assert torch.equal(sub == 0, full == floor_idx)
    assert torch.equal(sub == 1, full == wall_idx)
    assert torch.equal(sub == -1, (full != floor_idx) & (full != wall_idx))


def test_s3dis_hdf5_invalid_class_raises(datasets_dir_factory: Callable[..., Path]) -> None:
    """An unknown class name raises instead of being silently ignored"""
    datasets_dir = datasets_dir_factory(HDF5_GLOB)

    with pytest.raises(ValueError, match="Unknown class"):
        _ = S3DISHdf5(root=datasets_dir, classes=["wall", "spaceship"], download=False, show_progress=False)  # type: ignore[list-item]


def test_s3dis_hdf5_getitem_tensors_own_their_memory(datasets_dir_factory: Callable[..., Path]) -> None:
    """In-place edits on a returned sample never reach the cached numpy block"""
    datasets_dir = datasets_dir_factory(HDF5_GLOB)

    dataset = S3DISHdf5(root=datasets_dir, download=False, show_progress=False)
    original = dataset[0]["pos"].clone()
    dataset[0]["pos"].fill_(123.0)
    assert torch.equal(dataset[0]["pos"], original)


def test_s3dis_dataset_getitem_returns_shallow_copy(datasets_dir_factory: Callable[..., Path]) -> None:
    """User edits on a returned sample dict never reach the in-memory cache"""
    datasets_dir = datasets_dir_factory("S3DIS/processed_aligned/**/*")

    dataset = S3DIS(root=datasets_dir, show_progress=False)
    sample = dataset[0]
    assert sample is not dataset.data[0]
    sample["extra"] = 1
    assert "extra" not in dataset[0]
