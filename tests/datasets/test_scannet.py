# mypy: disable-error-code="arg-type,call-overload,attr-defined"
import io
import json
import shutil
from pathlib import Path
from typing import Callable
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest
import torch

from torch_pointcloud.datasets import ScanNet, ScanNet20, ScanNet200
from torch_pointcloud.datasets.scannet import (
    SCANNET20_CLASSES,
    SCANNET20_LABELS,
    SCANNET200_LABELS,
    SCANNET_UNK_CLS,
    load_scannet_scene,
    select_scannet_classes,
    tile_scannet_scene,
)
from torch_pointcloud.utils.data import DataKeys


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
    # Whole scenes are read on access, so a cached split loads without a pass over the data.
    assert "Loading" not in captured.err
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
    scene = next(data for data in dataset if data["scene"] == "scene0191_00")
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
    assert meta["format_version"] == 2
    assert meta["label_name"] == "nyu40class"
    assert dataset[0][DataKeys.COLOR].dtype == torch.uint8


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
    (Path(dataset.processed_dir) / "train" / dataset[0]["scene"] / "normal.npy").unlink()

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
    shutil.rmtree(Path(dataset.processed_dir) / "train" / dataset[0]["scene"])

    with pytest.raises(RuntimeError, match="force_process"):
        _ = ScanNet(root=datasets_dir, split="train", show_progress=False)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_scannet_dataset_skipped_scene_not_marked_complete(datasets_dir_factory: Callable[..., Path]) -> None:
    """A processing run that skips a scene writes no completion marker, so the next construction raises"""
    datasets_dir = datasets_dir_factory("ScanNet/raw/**/*")
    for mesh_file in datasets_dir.glob("**/scene0191_00_vh_clean_2.ply"):
        mesh_file.unlink()

    dataset = ScanNet(root=datasets_dir, split="train", show_progress=False)
    assert not (Path(dataset.processed_dir) / "train" / "meta.json").exists()

    with pytest.raises(RuntimeError, match="force_process"):
        _ = ScanNet(root=datasets_dir, split="train", show_progress=False)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_scannet_dataset_all_scenes_skipped_raises(datasets_dir_factory: Callable[..., Path]) -> None:
    """Construction raises instead of silently serving zero scenes when every scene fails to process"""
    datasets_dir = datasets_dir_factory("ScanNet/raw/**/*")
    for mesh_file in datasets_dir.glob("**/*_vh_clean_2.ply"):
        mesh_file.unlink()

    with pytest.raises(RuntimeError, match="No processed scenes"):
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
    assert dataset.classes[SCANNET200_LABELS.index(1163)] == "object"  # first TSV row wins over the later `stick` row
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
    assert dataset[0]["segment"].tolist() == expected


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_scannet20_load_matches_relabelled_base_segments(datasets_dir_factory: Callable[..., Path]) -> None:
    """The shared `load` + remap table reproduces the 20-class relabelling of the base segments"""
    datasets_dir = datasets_dir_factory("ScanNet/raw/**/*")

    base = ScanNet(root=datasets_dir, split="train", show_progress=False)
    dataset = ScanNet20(root=datasets_dir, split="train", show_progress=False)
    assert len(dataset) == len(base)

    lookup = {label: idx for idx, label in enumerate(SCANNET20_LABELS)}
    for base_scene, scene in zip(base, dataset):
        assert scene["scene"] == base_scene["scene"]
        expected = torch.tensor([lookup.get(int(label), 0) for label in base_scene["segment"]])
        assert torch.equal(scene["segment"], expected)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_scannet_dataset_superpoint_absent_by_default(datasets_dir_factory: Callable[..., Path]) -> None:
    """Without `return_superpoint` no `superpoint` key is emitted."""
    datasets_dir = datasets_dir_factory("ScanNet/raw/**/*")
    dataset = ScanNet(root=datasets_dir, split="train", show_progress=False)
    for data in dataset:
        assert DataKeys.SUPERPOINT not in data


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_scannet_dataset_return_superpoint(datasets_dir_factory: Callable[..., Path]) -> None:
    """`return_superpoint=True` emits per-point int64 superpoint ids matching the raw segs.json."""
    datasets_dir = datasets_dir_factory("ScanNet/raw/**/*")
    dataset = ScanNet(root=datasets_dir, split="train", return_superpoint=True, show_progress=False)
    for data in dataset:
        superpoint = data[DataKeys.SUPERPOINT]
        assert superpoint.dtype == torch.int64
        assert superpoint.shape == (data[DataKeys.POS].shape[0],)

    scene = next(data for data in dataset if data[DataKeys.SCENE] == "scene0191_00")
    segs_path = (
        datasets_dir / "ScanNet" / "raw" / "v2" / "scans" / "scene0191_00"
    ) / "scene0191_00_vh_clean_2.0.010000.segs.json"
    assert scene[DataKeys.SUPERPOINT].tolist() == json.loads(segs_path.read_text())["segIndices"]


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_scannet_dataset_superpoint_count_mismatch_raises(datasets_dir_factory: Callable[..., Path]) -> None:
    """A segs.json whose vertex count disagrees with the processed points raises a clear error on access."""
    datasets_dir = datasets_dir_factory("ScanNet/raw/**/*")
    _ = ScanNet(root=datasets_dir, split="train", show_progress=False)

    segs_path = (
        datasets_dir / "ScanNet" / "raw" / "v2" / "scans" / "scene0191_00"
    ) / "scene0191_00_vh_clean_2.0.010000.segs.json"
    segs = json.loads(segs_path.read_text())
    segs["segIndices"] = segs["segIndices"][:-1]
    segs_path.write_text(json.dumps(segs))

    dataset = ScanNet(root=datasets_dir, split="train", return_superpoint=True, show_progress=False)
    with pytest.raises(RuntimeError, match="superpoint/point count mismatch"):
        for _ in dataset:
            pass


def test_scannet_dataset_superpoint_requires_raw_scans(datasets_dir_factory: Callable[..., Path]) -> None:
    """`return_superpoint=True` with only a processed cache raises the manual-download error on access."""
    datasets_dir = datasets_dir_factory("ScanNet/processed/**/*")
    dataset = ScanNet(root=datasets_dir, split="train", return_superpoint=True, show_progress=False)
    with pytest.raises(RuntimeError, match="download the raw dataset"):
        _ = dataset[0]


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_scannet20_dataset_return_superpoint(datasets_dir_factory: Callable[..., Path]) -> None:
    """`ScanNet20` forwards `return_superpoint` and keeps the ids alongside the relabelled segments."""
    datasets_dir = datasets_dir_factory("ScanNet/raw/**/*")
    dataset = ScanNet20(root=datasets_dir, split="val", return_superpoint=True, show_progress=False)
    sample = dataset[0]
    assert sample[DataKeys.SUPERPOINT].dtype == torch.int64
    assert sample[DataKeys.SUPERPOINT].shape == (sample[DataKeys.POS].shape[0],)


def _synthetic_scene(num_points: int = 400) -> dict:
    generator = torch.Generator().manual_seed(0)
    pos = torch.rand(num_points, 3, generator=generator) * torch.tensor([3.0, 3.0, 2.0])
    return {
        DataKeys.POS: pos,
        DataKeys.COLOR: torch.rand(num_points, 3, generator=generator),
        DataKeys.SEGMENT: torch.randint(0, 5, (num_points,), generator=generator),
        DataKeys.SCENE: "scene_synthetic",
    }


def test_tile_scannet_scene_blocks_have_fixed_size_and_metadata() -> None:
    """Each block has exactly `num_nodes` rows across all per-point keys plus the tiling metadata keys"""
    scene = _synthetic_scene()
    blocks = tile_scannet_scene(scene, block_size=1.5, block_stride=0.75, num_nodes=128, min_num_nodes=1)

    assert len(blocks) > 1
    for block in blocks:
        assert block[DataKeys.POS].shape == (128, 3)
        assert block[DataKeys.COLOR].shape == (128, 3)
        assert block[DataKeys.SEGMENT].shape == (128,)
        assert block[DataKeys.SCENE] == "scene_synthetic"
        assert torch.equal(block[DataKeys.SCENE_MAX], scene[DataKeys.POS].max(dim=0).values)
        assert block[DataKeys.BLOCK_CENTER].shape == (3,)
        assert DataKeys.SCENE_INDEX not in block
        assert DataKeys.NUM_SCENE_POINTS not in block


def test_tile_scannet_scene_point_indices_map_back_to_scene() -> None:
    """`point_indices` recovers each block's rows from the source scene and stays inside the block window"""
    scene = _synthetic_scene()
    blocks = tile_scannet_scene(scene, block_size=1.5, block_stride=0.75, num_nodes=128, min_num_nodes=1)

    for block in blocks:
        indices = block[DataKeys.POINT_INDICES]
        assert torch.equal(block[DataKeys.POS], scene[DataKeys.POS][indices])
        assert torch.equal(block[DataKeys.SEGMENT], scene[DataKeys.SEGMENT][indices])
        offsets = (block[DataKeys.POS][:, :2] - block[DataKeys.BLOCK_CENTER][:2]).abs()
        assert (offsets <= 1.5 / 2 + 1e-6).all()


def test_tile_scannet_scene_emits_scene_index_when_requested() -> None:
    """Passing `scene_index` adds `scene_index` and `num_scene_points` to every block"""
    scene = _synthetic_scene(num_points=300)
    blocks = tile_scannet_scene(scene, block_size=1.5, block_stride=0.75, num_nodes=64, min_num_nodes=1, scene_index=3)

    assert len(blocks) > 0
    for block in blocks:
        assert block[DataKeys.SCENE_INDEX] == 3
        assert block[DataKeys.NUM_SCENE_POINTS] == 300


def test_tile_scannet_scene_drops_blocks_below_min_num_nodes() -> None:
    """Blocks covering only a sparse far-away cluster are dropped by `min_num_nodes`"""
    generator = torch.Generator().manual_seed(0)
    dense = torch.rand(200, 3, generator=generator)
    sparse = torch.rand(5, 3, generator=generator) + torch.tensor([10.0, 10.0, 0.0])
    scene = {DataKeys.POS: torch.cat([dense, sparse])}

    blocks = tile_scannet_scene(scene, block_size=1.0, block_stride=1.0, num_nodes=64, min_num_nodes=50)
    assert len(blocks) > 0
    for block in blocks:
        assert (block[DataKeys.POINT_INDICES] < 200).all()


def test_tile_scannet_scene_oversamples_small_blocks() -> None:
    """A block with fewer raw points than `num_nodes` is filled by resampling its own points"""
    scene = _synthetic_scene(num_points=60)
    blocks = tile_scannet_scene(scene, block_size=10.0, block_stride=10.0, num_nodes=128, min_num_nodes=1)

    assert len(blocks) == 1
    assert blocks[0][DataKeys.POS].shape == (128, 3)
    assert blocks[0][DataKeys.POINT_INDICES].unique().numel() <= 60


def test_tile_scannet_scene_blocks_own_their_scene_max() -> None:
    """Editing one block's `scene_max` in place must not leak into the other blocks"""
    scene = _synthetic_scene()
    blocks = tile_scannet_scene(scene, block_size=1.5, block_stride=0.75, num_nodes=64, min_num_nodes=1)

    assert len(blocks) > 1
    expected = blocks[1][DataKeys.SCENE_MAX].clone()
    blocks[0][DataKeys.SCENE_MAX].mul_(0.0)
    assert torch.equal(blocks[1][DataKeys.SCENE_MAX], expected)


def _synthetic_labels() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "raw_category": ["couch", "wall", "chair", "floor"],
            "id": [4, 1, 3, 2],
            "nyu40class": ["sofa", "wall", "chair", "floor"],
            "nyu40id": [6, 1, 5, 2],
        }
    )


def test_select_scannet_classes_all_returns_column_order() -> None:
    labels = _synthetic_labels()
    assert select_scannet_classes(labels, "raw_category", sort_by="id") == ["wall", "floor", "chair", "couch"]
    assert select_scannet_classes(labels, "nyu40class", sort_by="nyu40id") == ["wall", "floor", "chair", "sofa"]


def test_select_scannet_classes_subset_is_kept_verbatim() -> None:
    labels = _synthetic_labels()
    assert select_scannet_classes(labels, "raw_category", values=["wall", "chair"]) == ["wall", "chair"]


def test_select_scannet_classes_warns_and_drops_unknown_values() -> None:
    labels = _synthetic_labels()
    with pytest.warns(UserWarning, match="not present"):
        selected = select_scannet_classes(labels, "raw_category", values=["wall", "spaceship"])
    assert selected == ["wall"]


def test_select_scannet_classes_invalid_values_raises() -> None:
    labels = _synthetic_labels()
    with pytest.raises(ValueError, match="Invalid values"):
        select_scannet_classes(labels, "raw_category", values=42)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_scannet_dataset_cache_meta_mismatch_raises(datasets_dir_factory: Callable[..., Path]) -> None:
    """A processed cache written with different label columns raises instead of serving stale labels"""
    datasets_dir = datasets_dir_factory("ScanNet/raw/**/*")
    _ = ScanNet(root=datasets_dir, split="train", show_progress=False)

    with pytest.raises(RuntimeError, match="force_process=True"):
        _ = ScanNet(root=datasets_dir, split="train", label_name="raw_category", label_id="id", show_progress=False)

    dataset = ScanNet(
        root=datasets_dir,
        split="train",
        label_name="raw_category",
        label_id="id",
        show_progress=False,
        force_process=True,
    )
    assert len(dataset) > 0


def test_scannet_dataset_getitem_returns_shallow_copy(datasets_dir_factory: Callable[..., Path]) -> None:
    """User edits on a returned sample dict never reach the in-memory cache"""
    datasets_dir = datasets_dir_factory("ScanNet/processed/**/*")
    dataset = ScanNet(root=datasets_dir, show_progress=False)

    sample = dataset[0]
    assert sample is not dataset[0]
    sample["extra"] = 1
    assert "extra" not in dataset[0]
