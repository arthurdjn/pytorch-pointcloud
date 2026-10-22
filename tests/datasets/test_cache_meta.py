"""Every processed cache is stamped with its `cache_meta()` and refused once the stamp no longer matches."""

import json
from pathlib import Path
from typing import Callable

import pytest

from torch_pointcloud.datasets import (
    KITTI,
    S3DIS,
    ModelNet10,
    ModelNetNormalResampled,
    NuScenesMini,
    PointCloudDataset,
    ScanNet20,
    ScanObjectNN,
    ShapeNetPart,
    SunRGBD,
)

Builder = Callable[[Path, bool], PointCloudDataset]

DATASETS = [
    pytest.param(
        "KITTI",
        lambda root, force: KITTI(root=root, train=True, fov=False, force_process=force),
        "training/meta.json",
        id="kitti",
    ),
    pytest.param(
        "SunRGBD",
        lambda root, force: SunRGBD(root=root, train=False, show_progress=False, force_process=force),
        "val/meta.json",
        id="sunrgbd",
    ),
    pytest.param(
        "ModelNet10",
        lambda root, force: ModelNet10(root=root, train=True, show_progress=False, force_process=force),
        "train.meta.json",
        id="modelnet10",
    ),
    pytest.param(
        "ModelNetNormalResampled",
        lambda root, force: ModelNetNormalResampled(
            root=root, variant="10", train=True, show_progress=False, force_process=force
        ),
        "modelnet10_train.meta.json",
        id="modelnet-resampled",
    ),
    pytest.param(
        "ScanObjectNN",
        lambda root, force: ScanObjectNN(
            root=root, train=False, partition="split1", background=False, show_progress=False, force_process=force
        ),
        "split1_nobg/test_objectdataset.meta.json",
        id="scanobjectnn",
    ),
    pytest.param(
        "ShapeNetPart",
        lambda root, force: ShapeNetPart(root=root, split="test", show_progress=False, force_process=force),
        "test/meta.json",
        id="shapenetpart",
    ),
    pytest.param(
        "S3DIS",
        lambda root, force: S3DIS(
            root=root, areas=("Area_5",), aligned=True, download=False, show_progress=False, force_process=force
        ),
        "Area_5/meta.json",
        id="s3dis",
    ),
    pytest.param(
        "ScanNet",
        lambda root, force: ScanNet20(root=root, split="val", show_progress=False, force_process=force),
        "val/meta.json",
        id="scannet20",
    ),
    pytest.param(
        "NuScenesMini",
        lambda root, force: NuScenesMini(root=root, show_progress=False, force_process=force),
        "meta.json",
        id="nuscenes-mini",
    ),
]


@pytest.mark.parametrize(("name", "build", "meta_name"), DATASETS)
def test_process_stamps_cache_meta(
    datasets_dir_factory: Callable[..., Path], name: str, build: Builder, meta_name: str
) -> None:
    """Processing writes `cache_meta()` next to the cache, with the class's `format_version`."""
    datasets_dir = datasets_dir_factory(f"{name}/**/*")
    dataset = build(datasets_dir, True)
    meta = json.loads(Path(dataset.processed_dir, meta_name).read_text())
    assert meta == dataset.cache_meta()
    assert meta["format_version"] == type(dataset).format_version


@pytest.mark.parametrize(("name", "build", "meta_name"), DATASETS)
def test_stale_format_version_raises(
    datasets_dir_factory: Callable[..., Path], name: str, build: Builder, meta_name: str
) -> None:
    """A cache stamped with another `format_version` raises instead of being served, until it is regenerated."""
    datasets_dir = datasets_dir_factory(f"{name}/**/*")
    dataset = build(datasets_dir, True)
    meta_path = Path(dataset.processed_dir, meta_name)
    meta = json.loads(meta_path.read_text())
    meta["format_version"] += 1
    meta_path.write_text(json.dumps(meta))

    with pytest.raises(RuntimeError, match="force_process=True"):
        build(datasets_dir, False)

    rebuilt = build(datasets_dir, True)
    assert json.loads(meta_path.read_text()) == rebuilt.cache_meta()


@pytest.mark.parametrize(("name", "build", "meta_name"), DATASETS)
def test_cache_without_meta_is_served(
    datasets_dir_factory: Callable[..., Path], name: str, build: Builder, meta_name: str
) -> None:
    """A cache from before the stamp existed is served as is and left unstamped."""
    datasets_dir = datasets_dir_factory(f"{name}/**/*")
    dataset = build(datasets_dir, True)
    meta_path = Path(dataset.processed_dir, meta_name)
    meta_path.unlink()

    legacy = build(datasets_dir, False)
    assert len(legacy) == len(dataset)
    assert not meta_path.exists()
