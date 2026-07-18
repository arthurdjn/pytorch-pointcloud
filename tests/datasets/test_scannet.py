# mypy: disable-error-code="arg-type,call-overload,attr-defined"
import io
import json
import shutil
from pathlib import Path
from typing import Callable
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import torch

from torch_pointcloud.datasets import ScanNet, ScanNet20, ScanNet200
from torch_pointcloud.datasets.scannet import (
    SCANNET20_CLASSES,
    SCANNET20_LABELS,
    SCANNET200_LABELS,
    SCANNET_UNK_CLS,
    load_scannet_scene,
)


def test_load_scannet_scene(datasets_dir: Path) -> None:
    """Test that the ScanNet scene data is loaded correctly"""
    scene_dir = datasets_dir / "ScanNet" / "raw" / "v2" / "scans" / "scene0191_00"
    data = load_scannet_scene(
        mesh_path=scene_dir / "scene0191_00_vh_clean_2.ply",
        meta_path=scene_dir / "scene0191_00.txt",
        aggregation_path=scene_dir / "scene0191_00.aggregation.json",
        segments_path=scene_dir / "scene0191_00_vh_clean_2.0.010000.segs.json",
    )

    assert isinstance(data["pos"], torch.Tensor)
    assert isinstance(data["color"], torch.Tensor)
    assert isinstance(data["normal"], torch.Tensor)
    assert isinstance(data["instance"], torch.Tensor)
    assert isinstance(data["segment"], torch.Tensor)
    assert data["pos"].shape[1] == 3
    assert data["color"].shape[1] == 3
    assert data["normal"].shape[1] == 3
    assert data["instance"].ndim == 1
    assert data["segment"].ndim == 1
    # Unlabeled vertices take instance -1, never 0: ScanNet `objectId`s are 0-based, and a shared id 0
    # would merge unlabeled points into the first object's instance (corrupting derived boxes).
    assert (data["instance"] == -1).any()
    assert (data["instance"] == 0).any()
    assert torch.unique(data["segment"][data["instance"] == 0]).numel() == 1


def test_scannet_dataset_not_found() -> None:
    """Raises an error if the dataset is not found"""
    with pytest.raises(RuntimeError, match="Dataset not found"):
        _ = ScanNet(root="not-found", show_progress=False)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_scannet_dataset_raw_files_exist(datasets_dir_factory: Callable[..., Path], split: str) -> None:
    """Test that the raw files exist"""
    datasets_dir = datasets_dir_factory("ScanNet/raw/**/*")
    dataset = ScanNet(root=datasets_dir, split=split, show_progress=False)
    assert dataset.raw_files_exist()


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_scannet_dataset_raw_files_not_exist(split: str) -> None:
    """Test that an error is raised if the raw files do not exist"""
    with pytest.raises(RuntimeError, match="Dataset not found"):
        _ = ScanNet(root="not-found", split=split, show_progress=False)


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_scannet_dataset_processed_files_exist(datasets_dir_factory: Callable[..., Path], split: str) -> None:
    """Test that the processed files exist"""
    datasets_dir = datasets_dir_factory("ScanNet/processed/**/*")
    dataset = ScanNet(root=datasets_dir, split=split, show_progress=False)
    assert dataset.processed_files_exist()


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize("split", ["train", "val", "test"])
@patch("torch_pointcloud.datasets.scannet.load_scannet_scene", wraps=load_scannet_scene)
def test_scannet_dataset_split(mock_load: Mock, datasets_dir_factory: Callable[..., Path], split: str) -> None:
    """Test that the dataset does not load raw data if the processed data exists"""
    datasets_dir = datasets_dir_factory("ScanNet/processed/**/*")

    dataset = ScanNet(root=datasets_dir, split=split, show_progress=False)
    assert len(dataset) > 0
    _ = list(dataset)

    assert mock_load.call_count == 0


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize("split", ["train", "val", "test"])
@patch("torch_pointcloud.datasets.scannet.load_scannet_scene", wraps=load_scannet_scene)
def test_scannet_dataset_process_split(mock_load: Mock, datasets_dir_factory: Callable[..., Path], split: str) -> None:
    """Test that the dataset loads raw data if the processed data does not exist"""
    datasets_dir = datasets_dir_factory("ScanNet/raw/**/*")

    dataset = ScanNet(root=datasets_dir, split=split, show_progress=False)
    assert len(dataset) > 0
    _ = list(dataset)

    assert mock_load.call_count == len(dataset)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_scannet_dataset_missing_scenes(datasets_dir_factory: Callable[..., Path]) -> None:
    """Test that warnings are shown during processing if some scenes are missing"""
    datasets_dir = datasets_dir_factory("ScanNet/raw/**/*")

    # Remove a scene's mesh file to trigger the warning
    mesh_files = list(datasets_dir.glob("**/scene0191_00_vh_clean_2.ply"))
    for mesh_file in mesh_files:
        mesh_file.unlink()

    with pytest.warns(RuntimeWarning, match="Scene 'scene0191_00' is missing a mesh file"):
        _ = ScanNet(root=datasets_dir, show_progress=False)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_scannet_dataset_corrupted_ply(datasets_dir_factory: Callable[..., Path]) -> None:
    """Test that corrupted PLY files are skipped during processing"""
    datasets_dir = datasets_dir_factory("ScanNet/raw/**/*")

    # Corrupt a scene's PLY file
    mesh_files = list(datasets_dir.glob("**/scene0191_00_vh_clean_2.ply"))
    for mesh_file in mesh_files:
        mesh_file.unlink()
        with open(mesh_file, "w") as f:
            f.write("invalid PLY data")

    with pytest.warns(RuntimeWarning, match="Error loading scene 'scene0191_00'"):
        dataset = ScanNet(root=datasets_dir, show_progress=False)
        assert len(dataset) > 0  # Other valid scenes should still be loaded


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_scannet_dataset_corrupted_segments(datasets_dir_factory: Callable[..., Path]) -> None:
    """Test that corrupted segments files are skipped during processing"""
    datasets_dir = datasets_dir_factory("ScanNet/raw/**/*")

    # Corrupt a scene's segments file
    segs_files = list(datasets_dir.glob("**/scene0191_00_vh_clean_2.0.010000.segs.json"))
    for segs_file in segs_files:
        segs_file.unlink()
        with open(segs_file, "w") as f:
            f.write("invalid JSON data")

    with pytest.warns(RuntimeWarning, match="Error loading scene 'scene0191_00'"):
        dataset = ScanNet(root=datasets_dir, show_progress=False)
        assert len(dataset) > 0  # Other valid scenes should still be loaded


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_scannet_dataset_corrupted_aggregation(datasets_dir_factory: Callable[..., Path]) -> None:
    """Test that corrupted aggregation files are skipped during processing"""
    datasets_dir = datasets_dir_factory("ScanNet/raw/**/*")

    # Corrupt a scene's aggregation file
    agg_files = list(datasets_dir.glob("**/scene0191_00.aggregation.json"))
    for agg_file in agg_files:
        agg_file.unlink()
        with open(agg_file, "w") as f:
            f.write("invalid JSON data")

    with pytest.warns(RuntimeWarning, match="Error loading scene 'scene0191_00'"):
        dataset = ScanNet(root=datasets_dir, show_progress=False)
        assert len(dataset) > 0  # Other valid scenes should still be loaded


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_scannet_dataset_progress(
    datasets_dir_factory: Callable[..., Path], capsys: pytest.CaptureFixture[str]
) -> None:
    """Test that the dataset displays a progress bar during processing"""
    datasets_dir = datasets_dir_factory("ScanNet/raw/**/*")

    dataset = ScanNet(root=datasets_dir, show_progress=True)
    assert len(dataset) > 0
    captured = capsys.readouterr()
    assert "Processing" in captured.err
    assert captured.out == ""


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_scannet_dataset_without_progress(
    datasets_dir_factory: Callable[..., Path], capsys: pytest.CaptureFixture[str]
) -> None:
    """Test that the dataset does not display a progress bar during processing"""
    datasets_dir = datasets_dir_factory("ScanNet/raw/**/*")

    dataset = ScanNet(root=datasets_dir, show_progress=False)
    assert len(dataset) > 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == ""


def test_scannet_dataset_progress_with_cached_processed(
    datasets_dir_factory: Callable[..., Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test that no processing progress bar is shown if the processed dataset already exists"""
    datasets_dir = datasets_dir_factory("ScanNet/processed/**/*")

    dataset = ScanNet(root=datasets_dir, show_progress=True)
    assert len(dataset) > 0
    captured = capsys.readouterr()
    assert "Processing" not in captured.err
    assert "Loading" in captured.err
    assert captured.out == ""


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_scannet_dataset_all_classes(datasets_dir_factory: Callable[..., Path]) -> None:
    """Test that the dataset loads all classes"""
    datasets_dir = datasets_dir_factory("ScanNet/raw/**/*")

    dataset = ScanNet(root=datasets_dir, show_progress=False)
    assert len(dataset.classes) > 0
    assert len(dataset) > 0


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_scannet_dataset_transform(datasets_dir_factory: Callable[..., Path]) -> None:
    """Test that the dataset is transformed correctly after being processed"""
    datasets_dir = datasets_dir_factory("ScanNet/processed/**/*")

    transform = Mock(side_effect=lambda data: data)
    dataset = ScanNet(root=datasets_dir, transform=transform, show_progress=False)
    _ = list(dataset)
    assert transform.call_count == len(dataset)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize("version", ["v1", "v2"])
def test_scannet_dataset_versions(datasets_dir_factory: Callable[..., Path], version: str) -> None:
    """Test that the dataset loads different versions correctly"""
    datasets_dir = datasets_dir_factory("ScanNet/raw/**/*")

    dataset = ScanNet(root=datasets_dir, version=version, show_progress=False)
    assert len(dataset) > 0


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize("label_name,label_id", [("nyu40class", "nyu40id"), ("raw_category", "id")])
def test_scannet_dataset_label_columns(
    datasets_dir_factory: Callable[..., Path],
    label_name: str,
    label_id: str,
) -> None:
    """Test that the dataset loads different label columns correctly"""
    datasets_dir = datasets_dir_factory("ScanNet/raw/**/*")

    dataset = ScanNet(root=datasets_dir, label_name=label_name, label_id=label_id, show_progress=False)
    assert len(dataset) > 0


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_scannet_dataset_segment_ids_match_class_to_idx(datasets_dir_factory: Callable[..., Path]) -> None:
    """Stored segment ids follow the positional `class_to_idx` mapping for a non-default `label_name`"""
    datasets_dir = datasets_dir_factory("ScanNet/raw/**/*")

    dataset = ScanNet(root=datasets_dir, split="train", label_name="raw_category", label_id="id", show_progress=False)
    scene = next(data for data in dataset.data if data["scene"] == "scene0191_00")
    assert scene["segment"].max() < len(dataset.classes)

    aggregation_path = datasets_dir / "ScanNet" / "raw" / "v2" / "scans" / "scene0191_00"
    aggregation = json.loads((aggregation_path / "scene0191_00.aggregation.json").read_text())
    checked = 0
    for group in aggregation["segGroups"]:
        mask = scene["instance"] == group["objectId"]
        if not mask.any():
            continue
        expected = dataset.class_to_idx[group["label"]]
        assert scene["segment"][mask].unique().tolist() == [expected]
        checked += 1
    assert checked > 0


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_scannet_dataset_process_writes_completion_marker(datasets_dir_factory: Callable[..., Path]) -> None:
    """Processing a split ends with an atomic `meta.json` completion marker"""
    datasets_dir = datasets_dir_factory("ScanNet/raw/**/*")

    dataset = ScanNet(root=datasets_dir, split="train", show_progress=False)
    meta_path = Path(dataset.processed_dir) / "train" / "meta.json"
    meta = json.loads(meta_path.read_text())
    assert meta["format_version"] == 1
    assert meta["label_name"] == "nyu40class"


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_scannet_dataset_legacy_cache_without_marker_loads(datasets_dir_factory: Callable[..., Path]) -> None:
    """A complete cache without a completion marker (legacy layout) still loads"""
    datasets_dir = datasets_dir_factory("ScanNet/raw/**/*")

    dataset = ScanNet(root=datasets_dir, split="train", show_progress=False)
    (Path(dataset.processed_dir) / "train" / "meta.json").unlink()

    reloaded = ScanNet(root=datasets_dir, split="train", show_progress=False)
    assert len(reloaded) == len(dataset)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_scannet_dataset_interrupted_cache_detected(datasets_dir_factory: Callable[..., Path]) -> None:
    """An unmarked cache with a torn scene raises instead of loading it as unlabeled"""
    datasets_dir = datasets_dir_factory("ScanNet/raw/**/*")

    dataset = ScanNet(root=datasets_dir, split="train", show_progress=False)
    (Path(dataset.processed_dir) / "train" / "meta.json").unlink()
    (Path(dataset.processed_dir) / "train" / dataset.data[0]["scene"] / "normal.npy").unlink()

    with pytest.raises(RuntimeError, match="force_process"):
        _ = ScanNet(root=datasets_dir, split="train", show_progress=False)

    reprocessed = ScanNet(root=datasets_dir, split="train", show_progress=False, force_process=True)
    assert len(reprocessed) == len(dataset)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_scannet_dataset_missing_scene_detected(datasets_dir_factory: Callable[..., Path]) -> None:
    """An unmarked cache missing a scene listed in the split file raises"""
    datasets_dir = datasets_dir_factory("ScanNet/raw/**/*")

    dataset = ScanNet(root=datasets_dir, split="train", show_progress=False)
    (Path(dataset.processed_dir) / "train" / "meta.json").unlink()
    shutil.rmtree(Path(dataset.processed_dir) / "train" / dataset.data[0]["scene"])

    with pytest.raises(RuntimeError, match="force_process"):
        _ = ScanNet(root=datasets_dir, split="train", show_progress=False)


def test_scannet_dataset_download_rejects_malicious_scan_id(
    datasets_dir_factory: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A remote scan id with path separators is rejected before any write"""
    datasets_dir = datasets_dir_factory("ScanNet/processed/**/*")
    dataset = ScanNet(root=datasets_dir, split="train", show_progress=False)

    monkeypatch.setattr("torch_pointcloud.datasets.scannet.download_url", Mock())
    urlopen_mock = MagicMock()
    urlopen_mock.return_value.__enter__.return_value = io.BytesIO(b"../../x\n")
    monkeypatch.setattr("torch_pointcloud.datasets.scannet.urlopen", urlopen_mock)

    with pytest.raises(RuntimeError, match="Invalid scan id"):
        dataset.download(force=True)


def test_scannet20_classes_align_with_remapped_labels(datasets_dir_factory: Callable[..., Path]) -> None:
    """`ScanNet20.classes[i]` names the remapped segmentation label `i`"""
    datasets_dir = datasets_dir_factory("ScanNet/processed_20/**/*")

    dataset = ScanNet20(root=datasets_dir, split="train", show_progress=False)
    assert len(dataset.classes) == len(SCANNET20_LABELS)
    assert dataset.classes == [SCANNET_UNK_CLS, *SCANNET20_CLASSES]
    assert dataset.class_to_idx["wall"] == SCANNET20_LABELS.index(1)  # nyu40id 1
    assert dataset.class_to_idx["otherfurniture"] == SCANNET20_LABELS.index(39)  # nyu40id 39

    for data in dataset:
        assert data["segment"].max() < len(dataset.classes)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_scannet200_classes_from_fixture(datasets_dir_factory: Callable[..., Path]) -> None:
    """`ScanNet200.classes` resolves the TSV names in remap order (regression: `label_name='raw'` raised)"""
    datasets_dir = datasets_dir_factory("ScanNet/raw/**/*")

    dataset = ScanNet200(root=datasets_dir, split="train", show_progress=False)
    assert len(dataset.classes) == len(SCANNET200_LABELS)
    assert dataset.classes[0] == SCANNET_UNK_CLS
    assert dataset.classes[1] == "wall"  # TSV id 1
    assert dataset.classes[2] == "chair"  # TSV id 2
    assert dataset.classes[3] == "floor"  # TSV id 3
    assert dataset.classes[-1] == "mattress"  # TSV id 1191
    assert dataset.class_to_idx["chair"] == SCANNET200_LABELS.index(2)

    for data in dataset:
        assert data["segment"].max() < len(dataset.classes)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_scannet200_legacy_cache_relabels_raw_ids(datasets_dir_factory: Callable[..., Path]) -> None:
    """A cache without a marker holds raw TSV ids and is relabelled with the raw-id table"""
    datasets_dir = datasets_dir_factory("ScanNet/raw/**/*")

    split_file = datasets_dir / "ScanNet" / "raw" / "metadata" / "scannetv2_train.txt"
    scene_ids = sorted(line.strip() for line in split_file.read_text().splitlines() if line.strip())
    raw_ids = np.array([0, 1, 2, 1163], dtype=np.int32)
    for scene_id in scene_ids:
        scene_dir = datasets_dir / "ScanNet" / "processed_200" / "train" / scene_id
        scene_dir.mkdir(parents=True)
        np.save(scene_dir / "pos.npy", np.zeros((4, 3), dtype=np.float32))
        np.save(scene_dir / "color.npy", np.zeros((4, 3), dtype=np.float32))
        np.save(scene_dir / "normal.npy", np.zeros((4, 3), dtype=np.float32))
        np.save(scene_dir / "segment.npy", raw_ids)

    dataset = ScanNet200(root=datasets_dir, split="train", show_progress=False)
    expected = [0, SCANNET200_LABELS.index(1), SCANNET200_LABELS.index(2), SCANNET200_LABELS.index(1163)]
    assert dataset.data[0]["segment"].tolist() == expected


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_scannet20_load_matches_relabelled_base_segments(datasets_dir_factory: Callable[..., Path]) -> None:
    """The shared `load` + remap table reproduces the 20-class relabelling of the base segments"""
    datasets_dir = datasets_dir_factory("ScanNet/raw/**/*")

    base = ScanNet(root=datasets_dir, split="train", show_progress=False)
    dataset = ScanNet20(root=datasets_dir, split="train", show_progress=False)
    assert len(dataset) == len(base)

    lookup = {label: idx for idx, label in enumerate(SCANNET20_LABELS)}
    for base_scene, scene in zip(base.data, dataset.data):
        assert scene["scene"] == base_scene["scene"]
        expected = torch.tensor([lookup.get(int(label), 0) for label in base_scene["segment"]])
        assert torch.equal(scene["segment"], expected)
