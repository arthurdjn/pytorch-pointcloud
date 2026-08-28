"""Run every example script end to end on the dummy datasets shipped under `tests/data/datasets`."""

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterator, Tuple

import pytest

from torch_pointcloud.utils.imports import (
    _CUDA_AVAILABLE,
    _DWCONV_AVAILABLE,
    _MAMBA_SSM_AVAILABLE,
    _OCNN_AVAILABLE,
    _SPCONV_AVAILABLE,
    _TORCH_CLUSTER_AVAILABLE,
    _TORCH_SCATTER_AVAILABLE,
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
_REQUIRES_TORCH_CLUSTER = pytest.mark.skipif(not _TORCH_CLUSTER_AVAILABLE, reason="torch-cluster is not installed")
_REQUIRES_TORCH_SCATTER = pytest.mark.skipif(not _TORCH_SCATTER_AVAILABLE, reason="torch-scatter is not installed")

_CLUSTER = (_REQUIRES_TORCH_CLUSTER,)
_CLUSTER_SCATTER = (_REQUIRES_TORCH_CLUSTER, _REQUIRES_TORCH_SCATTER)
_GPU_SPCONV = (_REQUIRES_SPCONV, _REQUIRES_CUDA)
_GPU_SPCONV_SCATTER = (_REQUIRES_SPCONV, _REQUIRES_TORCH_SCATTER, _REQUIRES_CUDA)
_GPU_TORCHSPARSE = (_REQUIRES_TORCHSPARSE, _REQUIRES_CUDA)
_GPU_OCTREE = (_REQUIRES_OCNN, _REQUIRES_DWCONV, _REQUIRES_CUDA)
_GPU_MAMBA = (_REQUIRES_MAMBA, _REQUIRES_CUDA, _REQUIRES_TORCH_CLUSTER, _REQUIRES_TORCH_SCATTER)

# Benchmarks load registry weights, so they also carry the `pretrained` marker. Checkpoints without a dummy
# dataset here (SUN RGB-D, ModelNet40 HDF5, ScanNet200, the KITTI split files) and the unreleased SphereFormer
# weights have no row.
BENCHMARKS = [
    pytest.param("spunet_benchmark_segmentation.py", ("--limit", "1"), marks=_GPU_SPCONV, id="spunet/scannet"),
    pytest.param("kpconv_benchmark_segmentation.py", ("--limit", "1"), marks=_CLUSTER_SCATTER, id="kpconv/s3dis"),
    pytest.param("pointnext_benchmark_segmentation.py", ("--limit", "1"), marks=_CLUSTER_SCATTER, id="pointnext/s3dis"),
    pytest.param(
        "pointnext_benchmark_part_segmentation.py",
        ("--limit", "4"),
        marks=_CLUSTER_SCATTER,
        id="pointnext/shapenetpart",
    ),
    pytest.param(
        "pointnext_benchmark_classification.py",
        ("--model", "pointnext-sm.scanobjectnn.openpoints", "--limit", "8"),
        marks=_CLUSTER_SCATTER,
        id="pointnext/scanobjectnn",
    ),
    pytest.param(
        "pointnet2_benchmark_segmentation.py",
        ("--model", "pointnet2.s3dis-area5.xu-yan", "--limit", "1"),
        marks=_CLUSTER_SCATTER,
        id="pointnet2/s3dis-xu-yan",
    ),
    pytest.param(
        "pointnet2_benchmark_segmentation.py",
        ("--model", "pointnet2.s3dis-area5.openpoints", "--limit", "1"),
        marks=_CLUSTER_SCATTER,
        id="pointnet2/s3dis-openpoints",
    ),
    pytest.param(
        "pointnet2_benchmark_classification.py",
        ("--model", "pointnet2-msg.modelnet40.xu-yan", "--limit", "8"),
        marks=_CLUSTER_SCATTER,
        id="pointnet2/modelnet40",
    ),
    pytest.param(
        "pointnet2_benchmark_classification.py",
        ("--model", "pointnet2.scanobjectnn.openpoints", "--limit", "8"),
        marks=_CLUSTER_SCATTER,
        id="pointnet2/scanobjectnn",
    ),
    pytest.param("pvcnn_benchmark_segmentation.py", ("--limit", "1"), marks=_CLUSTER_SCATTER, id="pvcnn/s3dis"),
    pytest.param("sonata_benchmark_segmentation.py", ("--limit", "1"), marks=_GPU_SPCONV_SCATTER, id="sonata/scannet"),
    pytest.param(
        "concerto_benchmark_segmentation.py", ("--limit", "1"), marks=_GPU_SPCONV_SCATTER, id="concerto/scannet"
    ),
    pytest.param("utonia_benchmark_segmentation.py", ("--limit", "1"), marks=_GPU_SPCONV_SCATTER, id="utonia/scannet"),
    pytest.param(
        "ptv3_benchmark_segmentation.py",
        ("--model", "ptv3-base.scannet20.pointcept", "--limit", "1"),
        marks=_GPU_SPCONV_SCATTER,
        id="ptv3/scannet",
    ),
    pytest.param(
        "randlanet_benchmark_segmentation.py", ("--limit", "1"), marks=_CLUSTER_SCATTER, id="randlanet/semantickitti"
    ),
    pytest.param(
        "spvcnn_benchmark_segmentation.py", ("--limit", "1"), marks=_GPU_TORCHSPARSE, id="spvcnn/semantickitti"
    ),
    pytest.param(
        "dgcnn_benchmark_segmentation.py", ("--dataset", "scannet", "--limit", "1"), marks=_CLUSTER, id="dgcnn/scannet"
    ),
    pytest.param(
        "dgcnn_benchmark_segmentation.py",
        ("--dataset", "s3dis", "--area", "5", "--limit", "4"),
        marks=_CLUSTER,
        id="dgcnn/s3dis",
    ),
    pytest.param("dgcnn_benchmark_part_segmentation.py", ("--limit", "4"), marks=_CLUSTER, id="dgcnn/shapenetpart"),
    pytest.param(
        "octformer_benchmark_segmentation.py",
        ("--model", "octformer-base.scannet20.octree-nn", "--limit", "1"),
        marks=_GPU_OCTREE,
        id="octformer/scannet",
    ),
    pytest.param("octformer_benchmark_classification.py", (), marks=_GPU_OCTREE, id="octformer/modelnet40"),
    pytest.param(
        "oneformer3d_benchmark_segmentation.py",
        ("--model", "oneformer3d-base.scannet20.danila-rukhovich", "--limit", "1"),
        marks=_GPU_SPCONV_SCATTER,
        id="oneformer3d/scannet",
    ),
    pytest.param(
        "oneformer3d_benchmark_segmentation.py",
        ("--model", "oneformer3d-base.s3dis-area5.danila-rukhovich", "--limit", "1"),
        marks=_GPU_SPCONV_SCATTER,
        id="oneformer3d/s3dis",
    ),
    pytest.param(
        "point_bert_benchmark_classification.py",
        ("--model", "point-bert-base.modelnet40.xumin-yu", "--limit", "8"),
        marks=_CLUSTER_SCATTER,
        id="point_bert/modelnet40",
    ),
    pytest.param(
        "point_bert_benchmark_classification.py",
        ("--model", "point-bert-base.scanobjectnn-hardest.xumin-yu", "--limit", "8"),
        marks=_CLUSTER_SCATTER,
        id="point_bert/scanobjectnn",
    ),
    pytest.param(
        "point_mae_benchmark_classification.py",
        ("--model", "point-mae-base.modelnet40.yatian-pang", "--limit", "8"),
        marks=_CLUSTER_SCATTER,
        id="point_mae/modelnet40",
    ),
    pytest.param(
        "point_mae_benchmark_part_segmentation.py",
        ("--limit", "8"),
        marks=_CLUSTER_SCATTER,
        id="point_mae/shapenetpart",
    ),
    pytest.param(
        "point_m2ae_benchmark_classification.py",
        ("--model", "point-m2ae-base.modelnet40.renrui-zhang", "--limit", "8"),
        marks=_CLUSTER_SCATTER,
        id="point_m2ae/modelnet40",
    ),
    pytest.param(
        "point_m2ae_benchmark_part_segmentation.py",
        ("--limit", "8"),
        marks=_CLUSTER_SCATTER,
        id="point_m2ae/shapenetpart",
    ),
    pytest.param(
        "pointgpt_benchmark_classification.py",
        ("--model", "pointgpt-s.scanobjectnn-objonly.guangyan-chen", "--limit", "8"),
        marks=_CLUSTER_SCATTER,
        id="pointgpt/scanobjectnn",
    ),
    pytest.param(
        "point_mamba_benchmark_classification.py",
        ("--model", "point-mamba-base.modelnet40.dingkang-liang", "--limit", "8"),
        marks=_GPU_MAMBA,
        id="point_mamba/modelnet40",
    ),
    pytest.param(
        "pointmlp_benchmark_classification.py",
        ("--model", "pointmlp-base.scanobjectnn.xu-ma", "--limit", "8"),
        marks=_CLUSTER_SCATTER,
        id="pointmlp/scanobjectnn",
    ),
    pytest.param(
        "pointconv_benchmark_classification.py", ("--limit", "8"), marks=_CLUSTER_SCATTER, id="pointconv/modelnet40"
    ),
    pytest.param(
        "votenet_benchmark_detection.py",
        ("--model", "votenet.scannet.fair", "--limit", "2"),
        marks=_CLUSTER_SCATTER,
        id="votenet/scannet",
    ),
    pytest.param(
        "3detr_benchmark_detection.py",
        ("--model", "3detr-m.scannet.fair", "--limit", "2"),
        marks=_CLUSTER_SCATTER,
        id="3detr/scannet",
    ),
    pytest.param(
        "second_benchmark_detection.py",
        ("--model", "second.kitti.openpcdet", "--limit", "2"),
        marks=_GPU_SPCONV,
        id="second/kitti",
    ),
    pytest.param(
        "second_benchmark_detection.py",
        ("--model", "second-multihead.nuscenes.openpcdet", "--limit", "2"),
        marks=_GPU_SPCONV,
        id="second/nuscenes",
    ),
    pytest.param(
        "pointpillars_benchmark_detection.py",
        ("--model", "pointpillars.kitti.openpcdet", "--limit", "2"),
        marks=_GPU_SPCONV,
        id="pointpillars/kitti",
    ),
    pytest.param(
        "pointpillars_benchmark_detection.py",
        ("--model", "pointpillars-multihead.nuscenes.openpcdet", "--limit", "2"),
        marks=_GPU_SPCONV,
        id="pointpillars/nuscenes",
    ),
    pytest.param("pointrcnn_benchmark_detection.py", ("--limit", "2"), marks=_CLUSTER_SCATTER, id="pointrcnn/kitti"),
    pytest.param("voxelnext_benchmark_detection.py", ("--limit", "2"), marks=_GPU_SPCONV, id="voxelnext/nuscenes"),
    pytest.param("lion_benchmark_detection.py", ("--limit", "2"), marks=_GPU_MAMBA, id="lion/nuscenes"),
]

_SMOKE = ("--limit-train-batches", "2", "--limit-test-batches", "2", "--epochs", "1")
# One classification and one segmentation dataset per training script; VoteNet has no SUN RGB-D dummy data.
TRAININGS = [
    pytest.param("dgcnn_classification.py", (*_SMOKE, "--dataset", "modelnet10"), marks=_CLUSTER, id="dgcnn/cls"),
    pytest.param("dgcnn_segmentation.py", (*_SMOKE, "--dataset", "shapenetpart"), marks=_CLUSTER, id="dgcnn/seg"),
    pytest.param(
        "kpconv_classification.py", (*_SMOKE, "--dataset", "modelnet10"), marks=_CLUSTER_SCATTER, id="kpconv/cls"
    ),
    pytest.param(
        "kpconv_segmentation.py", (*_SMOKE, "--dataset", "shapenetpart"), marks=_CLUSTER_SCATTER, id="kpconv/seg"
    ),
    pytest.param(
        "octformer_classification.py", (*_SMOKE, "--dataset", "modelnet10"), marks=_GPU_OCTREE, id="octformer/cls"
    ),
    pytest.param(
        "octformer_segmentation.py", (*_SMOKE, "--dataset", "shapenetpart"), marks=_GPU_OCTREE, id="octformer/seg"
    ),
    pytest.param(
        "pointcnn_classification.py", (*_SMOKE, "--dataset", "modelnet10"), marks=_CLUSTER_SCATTER, id="pointcnn/cls"
    ),
    pytest.param(
        "pointcnn_segmentation.py", (*_SMOKE, "--dataset", "shapenetpart"), marks=_CLUSTER_SCATTER, id="pointcnn/seg"
    ),
    pytest.param(
        "pointconv_classification.py", (*_SMOKE, "--dataset", "modelnet10"), marks=_CLUSTER_SCATTER, id="pointconv/cls"
    ),
    pytest.param(
        "point_mamba_classification.py", (*_SMOKE, "--dataset", "modelnet10"), marks=_GPU_MAMBA, id="point_mamba/cls"
    ),
    pytest.param(
        "pointmlp_classification.py", (*_SMOKE, "--dataset", "modelnet10"), marks=_CLUSTER_SCATTER, id="pointmlp/cls"
    ),
    pytest.param(
        "pointmlp_segmentation.py", (*_SMOKE, "--dataset", "shapenetpart"), marks=_CLUSTER_SCATTER, id="pointmlp/seg"
    ),
    pytest.param(
        "pointnet2_classification.py", (*_SMOKE, "--dataset", "modelnet10"), marks=_CLUSTER_SCATTER, id="pointnet2/cls"
    ),
    pytest.param(
        "pointnet2_segmentation.py", (*_SMOKE, "--dataset", "shapenetpart"), marks=_CLUSTER_SCATTER, id="pointnet2/seg"
    ),
    pytest.param("pointnet_classification.py", (*_SMOKE, "--dataset", "modelnet10"), marks=_CLUSTER, id="pointnet/cls"),
    pytest.param("pointnet_segmentation.py", (*_SMOKE, "--dataset", "shapenetpart"), marks=_CLUSTER, id="pointnet/seg"),
    pytest.param(
        "pointnext_classification.py", (*_SMOKE, "--dataset", "modelnet10"), marks=_CLUSTER_SCATTER, id="pointnext/cls"
    ),
    pytest.param(
        "pointnext_segmentation.py", (*_SMOKE, "--dataset", "shapenetpart"), marks=_CLUSTER_SCATTER, id="pointnext/seg"
    ),
    pytest.param(
        "point_transformer_classification.py",
        (*_SMOKE, "--dataset", "modelnet10"),
        marks=_CLUSTER_SCATTER,
        id="point_transformer/cls",
    ),
    pytest.param(
        "point_transformer_segmentation.py",
        (*_SMOKE, "--dataset", "shapenetpart"),
        marks=_CLUSTER_SCATTER,
        id="point_transformer/seg",
    ),
    pytest.param(
        "point_transformer_v2_classification.py",
        (*_SMOKE, "--dataset", "modelnet10"),
        marks=_CLUSTER_SCATTER,
        id="point_transformer_v2/cls",
    ),
    pytest.param(
        "point_transformer_v2_segmentation.py",
        (*_SMOKE, "--dataset", "shapenetpart"),
        marks=_CLUSTER_SCATTER,
        id="point_transformer_v2/seg",
    ),
    pytest.param(
        "point_transformer_v3_classification.py",
        (*_SMOKE, "--dataset", "modelnet10"),
        marks=_GPU_SPCONV_SCATTER,
        id="point_transformer_v3/cls",
    ),
    pytest.param(
        "point_transformer_v3_segmentation.py",
        (*_SMOKE, "--dataset", "shapenetpart"),
        marks=_GPU_SPCONV_SCATTER,
        id="point_transformer_v3/seg",
    ),
    pytest.param(
        "pvcnn2_classification.py", (*_SMOKE, "--dataset", "modelnet10"), marks=_CLUSTER_SCATTER, id="pvcnn2/cls"
    ),
    pytest.param(
        "pvcnn2_segmentation.py", (*_SMOKE, "--dataset", "shapenetpart"), marks=_CLUSTER_SCATTER, id="pvcnn2/seg"
    ),
    pytest.param(
        "pvcnn_classification.py", (*_SMOKE, "--dataset", "modelnet10"), marks=_CLUSTER_SCATTER, id="pvcnn/cls"
    ),
    pytest.param(
        "pvcnn_segmentation.py", (*_SMOKE, "--dataset", "shapenetpart"), marks=_CLUSTER_SCATTER, id="pvcnn/seg"
    ),
    pytest.param(
        "randlanet_classification.py", (*_SMOKE, "--dataset", "modelnet10"), marks=_CLUSTER_SCATTER, id="randlanet/cls"
    ),
    pytest.param(
        "randlanet_segmentation.py", (*_SMOKE, "--dataset", "shapenetpart"), marks=_CLUSTER_SCATTER, id="randlanet/seg"
    ),
    pytest.param(
        "spvcnn_classification.py", (*_SMOKE, "--dataset", "modelnet10"), marks=_GPU_TORCHSPARSE, id="spvcnn/cls"
    ),
    pytest.param(
        "spvcnn_segmentation.py", (*_SMOKE, "--dataset", "shapenetpart"), marks=_GPU_TORCHSPARSE, id="spvcnn/seg"
    ),
]


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
        "dgcnn_benchmark_classification.py",
        "sphereformer_benchmark_segmentation.py",
        "votenet_detection.py",
    }
    covered = {str(param.values[0]) for param in BENCHMARKS + TRAININGS}
    scripts = {path.name for path in EXAMPLES_DIR.glob("*.py")}
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
