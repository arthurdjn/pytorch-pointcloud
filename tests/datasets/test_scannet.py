# mypy: disable-error-code="arg-type,call-overload,attr-defined"
from pathlib import Path
from typing import Callable
from unittest.mock import Mock, patch

import pytest
import torch

from torch_pointcloud.datasets import ScanNet
from torch_pointcloud.datasets.scannet import load_scannet_scene


def test_load_scannet_scene(data_dir: Path) -> None:
    """Test that the ScanNet scene data is loaded correctly"""
    scene_dir = data_dir / "ScanNet" / "raw" / "v2" / "scans" / "scene0000_00"
    data = load_scannet_scene(
        mesh_path=scene_dir / "scene0000_00_vh_clean_2.ply",
        meta_path=scene_dir / "scene0000_00.txt",
        aggregation_path=scene_dir / "scene0000_00.aggregation.json",
        segments_path=scene_dir / "scene0000_00_vh_clean_2.0.010000.segs.json",
    )

    assert isinstance(data["points"], torch.Tensor)
    assert isinstance(data["colors"], torch.Tensor)
    assert isinstance(data["normals"], torch.Tensor)
    assert isinstance(data["instances"], torch.Tensor)
    assert isinstance(data["labels"], torch.Tensor)
    assert data["points"].shape[1] == 3
    assert data["colors"].shape[1] == 3
    assert data["normals"].shape[1] == 3
    assert data["instances"].ndim == 1
    assert data["labels"].ndim == 1


def test_scannet_dataset_not_found() -> None:
    """Raises an error if the dataset is not found"""
    with pytest.raises(RuntimeError, match="Dataset not found"):
        _ = ScanNet(root="not-found", show_progress=False)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_scannet_dataset_raw_files_exist(data_dir_factory: Callable[..., Path], split: str) -> None:
    """Test that the raw files exist"""
    data_dir = data_dir_factory("ScanNet/raw/**/*")
    dataset = ScanNet(root=data_dir, split=split, show_progress=False)
    assert dataset.raw_files_exist()


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_scannet_dataset_raw_files_not_exist(split: str) -> None:
    """Test that an error is raised if the raw files do not exist"""
    with pytest.raises(RuntimeError, match="Dataset not found"):
        _ = ScanNet(root="not-found", split=split, show_progress=False)


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_scannet_dataset_processed_files_exist(data_dir_factory: Callable[..., Path], split: str) -> None:
    """Test that the processed files exist"""
    data_dir = data_dir_factory("ScanNet/processed/**/*")
    dataset = ScanNet(root=data_dir, split=split, show_progress=False)
    assert dataset.processed_files_exist()


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize("split", ["train", "val", "test"])
@patch("torch_pointcloud.datasets.scannet.load_scannet_scene", wraps=load_scannet_scene)
def test_scannet_dataset_split(mock_load: Mock, data_dir_factory: Callable[..., Path], split: str) -> None:
    """Test that the dataset does not load raw data if the processed data exists"""
    data_dir = data_dir_factory("ScanNet/processed/**/*")

    dataset = ScanNet(root=data_dir, split=split, show_progress=False)
    assert len(dataset) > 0
    _ = list(dataset)

    assert mock_load.call_count == 0


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize("split", ["train", "val", "test"])
@patch("torch_pointcloud.datasets.scannet.load_scannet_scene", wraps=load_scannet_scene)
def test_scannet_dataset_process_split(mock_load: Mock, data_dir_factory: Callable[..., Path], split: str) -> None:
    """Test that the dataset loads raw data if the processed data does not exist"""
    data_dir = data_dir_factory("ScanNet/raw/**/*")

    dataset = ScanNet(root=data_dir, split=split, show_progress=False)
    assert len(dataset) > 0
    _ = list(dataset)

    assert mock_load.call_count == len(dataset)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_scannet_dataset_missing_scenes(data_dir_factory: Callable[..., Path]) -> None:
    """Test that warnings are shown during processing if some scenes are missing"""
    data_dir = data_dir_factory("ScanNet/raw/**/*")

    # Remove a scene's mesh file to trigger the warning
    mesh_files = list(data_dir.glob("**/scene0000_00_vh_clean_2.ply"))
    for mesh_file in mesh_files:
        mesh_file.unlink()

    with pytest.warns(RuntimeWarning, match="Scene 'scene0000_00' is missing a mesh file"):
        _ = ScanNet(root=data_dir, show_progress=False)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_scannet_dataset_corrupted_ply(data_dir_factory: Callable[..., Path]) -> None:
    """Test that corrupted PLY files are skipped during processing"""
    data_dir = data_dir_factory("ScanNet/raw/**/*")

    # Corrupt a scene's PLY file
    mesh_files = list(data_dir.glob("**/scene0000_00_vh_clean_2.ply"))
    for mesh_file in mesh_files:
        mesh_file.unlink()
        with open(mesh_file, "w") as f:
            f.write("invalid PLY data")

    with pytest.warns(RuntimeWarning, match="Error loading scene 'scene0000_00'"):
        dataset = ScanNet(root=data_dir, show_progress=False)
        assert len(dataset) > 0  # Other valid scenes should still be loaded


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_scannet_dataset_corrupted_segments(data_dir_factory: Callable[..., Path]) -> None:
    """Test that corrupted segments files are skipped during processing"""
    data_dir = data_dir_factory("ScanNet/raw/**/*")

    # Corrupt a scene's segments file
    segs_files = list(data_dir.glob("**/scene0000_00_vh_clean_2.0.010000.segs.json"))
    for segs_file in segs_files:
        segs_file.unlink()
        with open(segs_file, "w") as f:
            f.write("invalid JSON data")

    with pytest.warns(RuntimeWarning, match="Error loading scene 'scene0000_00'"):
        dataset = ScanNet(root=data_dir, show_progress=False)
        assert len(dataset) > 0  # Other valid scenes should still be loaded


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_scannet_dataset_corrupted_aggregation(data_dir_factory: Callable[..., Path]) -> None:
    """Test that corrupted aggregation files are skipped during processing"""
    data_dir = data_dir_factory("ScanNet/raw/**/*")

    # Corrupt a scene's aggregation file
    agg_files = list(data_dir.glob("**/scene0000_00.aggregation.json"))
    for agg_file in agg_files:
        agg_file.unlink()
        with open(agg_file, "w") as f:
            f.write("invalid JSON data")

    with pytest.warns(RuntimeWarning, match="Error loading scene 'scene0000_00'"):
        dataset = ScanNet(root=data_dir, show_progress=False)
        assert len(dataset) > 0  # Other valid scenes should still be loaded


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_scannet_dataset_progress(data_dir_factory: Callable[..., Path], capsys: pytest.CaptureFixture[str]) -> None:
    """Test that the dataset displays a progress bar during processing"""
    data_dir = data_dir_factory("ScanNet/raw/**/*")

    dataset = ScanNet(root=data_dir, show_progress=True)
    assert len(dataset) > 0
    captured = capsys.readouterr()
    assert "Processing" in captured.err
    assert captured.out == ""


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_scannet_dataset_without_progress(
    data_dir_factory: Callable[..., Path], capsys: pytest.CaptureFixture[str]
) -> None:
    """Test that the dataset does not display a progress bar during processing"""
    data_dir = data_dir_factory("ScanNet/raw/**/*")

    dataset = ScanNet(root=data_dir, show_progress=False)
    assert len(dataset) > 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == ""


def test_scannet_dataset_progress_with_cached_processed(
    data_dir_factory: Callable[..., Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test that no progress bar is shown if the processed dataset already exists"""
    data_dir = data_dir_factory("ScanNet/processed/**/*")

    dataset = ScanNet(root=data_dir, show_progress=True)
    assert len(dataset) > 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == ""


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize("classes", [["wall"], ["wall", "floor", "ceiling"]])
def test_scannet_dataset_classes(data_dir_factory: Callable[..., Path], classes: list[str]) -> None:
    """Test that the dataset loads specific classes"""
    data_dir = data_dir_factory("ScanNet/raw/**/*")

    dataset = ScanNet(root=data_dir, classes=classes, with_unk=True, show_progress=False)
    assert len(dataset) > 0
    assert all(cls in dataset.classes for cls in classes)

    class_ids = set(dataset.class_to_idx.values())
    for data in dataset:
        labels = data["labels"].unique().tolist()
        assert set(labels).issubset(class_ids)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_scannet_dataset_all_classes(data_dir_factory: Callable[..., Path]) -> None:
    """Test that the dataset loads all classes"""
    data_dir = data_dir_factory("ScanNet/raw/**/*")

    dataset = ScanNet(root=data_dir, classes="all", show_progress=False)
    assert len(dataset.classes) > 0
    assert len(dataset) > 0


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_scannet_dataset_pre_transform(data_dir_factory: Callable[..., Path]) -> None:
    """Test that the dataset is transformed correctly before being processed"""
    data_dir = data_dir_factory("ScanNet/raw/**/*")

    pre_transform = Mock(side_effect=lambda x: x)
    _ = ScanNet(root=data_dir, pre_transform=pre_transform, show_progress=False)
    assert pre_transform.call_count > 0


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_scannet_dataset_pre_filter(data_dir_factory: Callable[..., Path]) -> None:
    """Test that the dataset is filtered correctly before being processed"""
    data_dir = data_dir_factory("ScanNet/raw/**/*")

    pre_filter = Mock(side_effect=lambda x: True)
    _ = ScanNet(root=data_dir, pre_filter=pre_filter, show_progress=False)
    assert pre_filter.call_count > 0


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_scannet_dataset_transform(data_dir_factory: Callable[..., Path]) -> None:
    """Test that the dataset is transformed correctly after being processed"""
    data_dir = data_dir_factory("ScanNet/processed/**/*")

    transform = Mock(side_effect=lambda data: data)
    dataset = ScanNet(root=data_dir, transform=transform, show_progress=False)
    _ = list(dataset)
    assert transform.call_count == len(dataset)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize("version", ["v1", "v2"])
def test_scannet_dataset_versions(data_dir_factory: Callable[..., Path], version: str) -> None:
    """Test that the dataset loads different versions correctly"""
    data_dir = data_dir_factory("ScanNet/raw/**/*")

    dataset = ScanNet(root=data_dir, version=version, show_progress=False)
    assert len(dataset) > 0


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize("label_name,label_id", [("nyu40class", "nyu40id"), ("raw_category", "id")])
def test_scannet_dataset_label_columns(
    data_dir_factory: Callable[..., Path],
    label_name: str,
    label_id: str,
) -> None:
    """Test that the dataset loads different label columns correctly"""
    data_dir = data_dir_factory("ScanNet/raw/**/*")

    dataset = ScanNet(root=data_dir, label_name=label_name, label_id=label_id, show_progress=False)
    assert len(dataset) > 0
