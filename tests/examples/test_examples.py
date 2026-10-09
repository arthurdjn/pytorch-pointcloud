"""Run every example script end to end on the dummy datasets shipped under `tests/data/datasets`."""

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterator, Tuple

import pytest
from _pytest.mark.structures import ParameterSet

from torch_pointcloud.utils.imports import (
    _CUDA_AVAILABLE,
    _DWCONV_AVAILABLE,
    _MAMBA_SSM_AVAILABLE,
    _OCNN_AVAILABLE,
    _PYG_LIB_AVAILABLE,
    _SPCONV_AVAILABLE,
    _TORCHSPARSE_AVAILABLE,
)

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"
DATASETS_DIR = Path(__file__).resolve().parents[1] / "data" / "datasets"
TIMEOUT = 900

_REQUIRES_CUDA = pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA is not available")
_REQUIRES_SPCONV = pytest.mark.skipif(not _SPCONV_AVAILABLE, reason="spconv is not installed")
_REQUIRES_TORCHSPARSE = pytest.mark.skipif(not _TORCHSPARSE_AVAILABLE, reason="torchsparse is not installed")
_REQUIRES_OCNN = pytest.mark.skipif(not _OCNN_AVAILABLE, reason="ocnn is not installed")
_REQUIRES_DWCONV = pytest.mark.skipif(not _DWCONV_AVAILABLE, reason="dwconv is not installed")
_REQUIRES_MAMBA = pytest.mark.skipif(not _MAMBA_SSM_AVAILABLE, reason="mamba-ssm is not installed")
_REQUIRES_PYG_LIB = pytest.mark.skipif(not _PYG_LIB_AVAILABLE, reason="pyg-lib is not installed")

_CLUSTER = (_REQUIRES_PYG_LIB,)
_GPU_SPCONV = (_REQUIRES_SPCONV, _REQUIRES_CUDA)
_GPU_TORCHSPARSE = (_REQUIRES_TORCHSPARSE, _REQUIRES_CUDA)
_GPU_OCTREE = (_REQUIRES_OCNN, _REQUIRES_DWCONV, _REQUIRES_CUDA)
_GPU_MAMBA = (_REQUIRES_MAMBA, _REQUIRES_CUDA, _REQUIRES_PYG_LIB)

# Benchmarks load registry weights, so they also carry the `pretrained` marker. Checkpoints without a dummy
# dataset here (SUN RGB-D, ModelNet40 HDF5, ScanNet200, the KITTI split files) and the unreleased SphereFormer
# weights have no row.
BENCHMARKS = [
    pytest.param("spunet/segmentation_scannet_benchmark.py", ("--limit", "1"), marks=_GPU_SPCONV, id="spunet/scannet"),
    pytest.param("kpconv/segmentation_s3dis_benchmark.py", ("--limit", "1"), marks=_CLUSTER, id="kpconv/s3dis"),
    pytest.param("pointnext/segmentation_s3dis_benchmark.py", ("--limit", "1"), marks=_CLUSTER, id="pointnext/s3dis"),
    pytest.param(
        "pointnext/part_segmentation_shapenetpart_benchmark.py",
        ("--limit", "4"),
        marks=_CLUSTER,
        id="pointnext/shapenetpart",
    ),
    pytest.param(
        "pointnet2/segmentation_s3dis_benchmark.py",
        ("--model", "pointnet2.s3dis-area5.xu-yan", "--limit", "1"),
        marks=_CLUSTER,
        id="pointnet2/s3dis-xu-yan",
    ),
    pytest.param(
        "pointnet2/segmentation_s3dis_benchmark.py",
        ("--model", "pointnet2.s3dis-area5.openpoints", "--limit", "1"),
        marks=_CLUSTER,
        id="pointnet2/s3dis-openpoints",
    ),
    pytest.param(
        "pointnet2/classification_scanobjectnn_benchmark.py",
        ("--model", "pointnet2.scanobjectnn-hardest.openpoints", "--limit", "8"),
        marks=_CLUSTER,
        id="pointnet2/scanobjectnn",
    ),
    pytest.param("pvcnn/segmentation_s3dis_benchmark.py", ("--limit", "1"), marks=_CLUSTER, id="pvcnn/s3dis"),
    pytest.param("sonata/segmentation_scannet_benchmark.py", ("--limit", "1"), marks=_GPU_SPCONV, id="sonata/scannet"),
    pytest.param(
        "concerto/segmentation_scannet_benchmark.py", ("--limit", "1"), marks=_GPU_SPCONV, id="concerto/scannet"
    ),
    pytest.param("utonia/segmentation_scannet_benchmark.py", ("--limit", "1"), marks=_GPU_SPCONV, id="utonia/scannet"),
    pytest.param(
        "randlanet/segmentation_semantickitti_benchmark.py",
        ("--limit", "1"),
        marks=_CLUSTER,
        id="randlanet/semantickitti",
    ),
    pytest.param(
        "spvcnn/segmentation_semantickitti_benchmark.py",
        ("--limit", "1"),
        marks=_GPU_TORCHSPARSE,
        id="spvcnn/semantickitti",
    ),
    pytest.param("dgcnn/segmentation_scannet_benchmark.py", ("--limit", "1"), marks=_CLUSTER, id="dgcnn/scannet"),
    pytest.param(
        "dgcnn/part_segmentation_shapenetpart_benchmark.py", ("--limit", "4"), marks=_CLUSTER, id="dgcnn/shapenetpart"
    ),
    pytest.param(
        "octformer/segmentation_scannet_benchmark.py",
        ("--model", "octformer-base.scannet20.octree-nn", "--limit", "1"),
        marks=_GPU_OCTREE,
        id="octformer/scannet",
    ),
    pytest.param("octformer/classification_modelnet40_benchmark.py", (), marks=_GPU_OCTREE, id="octformer/modelnet40"),
    pytest.param(
        "point_bert/classification_scanobjectnn_benchmark.py",
        ("--model", "point-bert-base.scanobjectnn-hardest.xumin-yu", "--limit", "8"),
        marks=_CLUSTER,
        id="point_bert/scanobjectnn",
    ),
    pytest.param(
        "point_m2ae/classification_scanobjectnn_benchmark.py",
        ("--limit", "8"),
        marks=_CLUSTER,
        id="point_m2ae/scanobjectnn",
    ),
    pytest.param(
        "point_mae/part_segmentation_shapenetpart_benchmark.py",
        ("--limit", "8"),
        marks=_CLUSTER,
        id="point_mae/shapenetpart",
    ),
    pytest.param(
        "point_m2ae/part_segmentation_shapenetpart_benchmark.py",
        ("--limit", "8"),
        marks=_CLUSTER,
        id="point_m2ae/shapenetpart",
    ),
    pytest.param(
        "threedetr/detection_scannet_benchmark.py",
        ("--model", "3detr-m.scannet.fair", "--limit", "2"),
        marks=_CLUSTER,
        id="3detr/scannet",
    ),
    pytest.param("pointrcnn/detection_kitti_benchmark.py", ("--limit", "2"), marks=_CLUSTER, id="pointrcnn/kitti"),
    pytest.param(
        "voxelnext/detection_nuscenes_benchmark.py",
        ("--split", "mini", "--limit", "2"),
        marks=_GPU_SPCONV,
        id="voxelnext/nuscenes",
    ),
    pytest.param(
        "lion/detection_nuscenes_benchmark.py",
        ("--split", "mini", "--limit", "2"),
        marks=_GPU_MAMBA,
        id="lion/nuscenes",
    ),
]

_SMOKE = ("--limit-train-batches", "2", "--limit-test-batches", "2", "--epochs", "1")
# The reproduction scripts (one per reference recipe) validate on `--limit-val-batches` scenes.
_SMOKE_RECIPE = ("--limit-train-batches", "2", "--limit-val-batches", "2", "--epochs", "1", "--eval-every", "1")
# One classification and one segmentation dataset per training script; VoteNet has no SUN RGB-D dummy data.
TRAININGS: list[ParameterSet] = []


@pytest.fixture(scope="module")
def examples_datasets_dir(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """One writable copy of the dummy datasets for the whole module, so dataset caches never touch the fixtures."""
    dest = tmp_path_factory.mktemp("datasets")
    shutil.copytree(DATASETS_DIR, dest, dirs_exist_ok=True)
    yield dest
    shutil.rmtree(dest, ignore_errors=True)


def _run(script: str, args: Tuple[str, ...], root: Path) -> None:
    command = [sys.executable, str(EXAMPLES_DIR / script), "--root", str(root), "--num-workers", "0", *args]
    result = subprocess.run(command, capture_output=True, text=True, timeout=TIMEOUT)
    assert result.returncode == 0, f"{' '.join(command)}\n{result.stdout[-2000:]}\n{result.stderr[-4000:]}"


def test_every_script_has_a_row() -> None:
    """Every example script is exercised by at least one row, or is named here with the reason it is not."""
    without_dummy_data = {
        "dgcnn/classification_modelnet40_benchmark.py",
        "pointnext/classification_modelnet40_benchmark.py",
        "sphereformer/segmentation_semantickitti_benchmark.py",
        "votenet/detection_sunrgbd_benchmark.py",
    }
    covered = {str(param.values[0]) for param in BENCHMARKS + TRAININGS}
    scripts = {path.relative_to(EXAMPLES_DIR).as_posix() for path in EXAMPLES_DIR.glob("*/*.py")}
    assert scripts - covered == without_dummy_data


@pytest.mark.example
@pytest.mark.pretrained
@pytest.mark.parametrize("script, args", BENCHMARKS)
def test_benchmark_script(script: str, args: Tuple[str, ...], examples_datasets_dir: Path) -> None:
    _run(script, args, examples_datasets_dir)


@pytest.mark.example
@pytest.mark.parametrize("script, args", TRAININGS)
def test_training_script(script: str, args: Tuple[str, ...], examples_datasets_dir: Path) -> None:
    _run(script, args, examples_datasets_dir)
