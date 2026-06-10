"""Numerical non-regression tests for pretrained models, grouped by dataset.

Each test loads two real samples from [tests/data/datasets/](../data/datasets/),
applies the model's registered transforms, runs the forward, and compares the
output against a `.safetensors` snapshot under [tests/data/models/](../data/models/).

Tests are gated behind `--run-pretrained` (skipped in CI). Pass `--force-regen`
to overwrite snapshots.

Regenerate snapshots:

```bash
uv run --no-sync pytest tests/models/test_pretrained.py --run-pretrained --force-regen
```
"""

import inspect
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import pytest
import torch
from safetensors.torch import load_file, save_file
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from torch_pointcloud.datasets import (
    S3DIS,
    ModelNetNormalResampled,
    S3DISHdf5,
    ScanNet20,
    ScanObjectNN,
    SemanticKITTI,
    ShapeNetPart,
)
from torch_pointcloud.models import create_model
from torch_pointcloud.utils.data import collate
from torch_pointcloud.utils.imports import (
    _DWCONV_AVAILABLE,
    _FLASH_ATTN_AVAILABLE,
    _MAMBA_SSM_AVAILABLE,
    _SPCONV_AVAILABLE,
    _TORCH_CLUSTER_AVAILABLE,
    _TORCH_SCATTER_AVAILABLE,
    _TORCHSPARSE_AVAILABLE,
)
from torch_pointcloud.utils.voxelization import hard_voxelize

ATOL = 5e-3
RTOL = 5e-2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATASETS_DIR = DATA_DIR / "datasets"
SNAPSHOTS_DIR = DATA_DIR / "models"


def _scannet20_blocks(**kwargs: Any) -> ScanNet20:
    # tile_scannet_scene samples points per block with torch RNG; seed so the snapshot stays reproducible.
    torch.manual_seed(0)
    return ScanNet20(
        root=DATASETS_DIR,
        split="val",
        use_axis_alignment=False,
        block_size=1.5,
        block_stride=0.75,
        num_nodes=8192,
        show_progress=False,
        **kwargs,
    )


DATASET_REGISTRY: Dict[str, Callable[..., Dataset]] = {
    "modelnet_resampled": partial(
        ModelNetNormalResampled,
        root=DATASETS_DIR,
        variant="40",
        train=False,
        show_progress=False,
    ),
    "s3dis_hdf5": partial(
        S3DISHdf5,
        root=DATASETS_DIR,
        areas=("Area_5",),
        show_progress=False,
    ),
    "s3dis": partial(
        S3DIS,
        root=DATASETS_DIR,
        areas=("Area_5",),
        aligned=True,
        download=False,
        show_progress=False,
    ),
    "shapenetpart": partial(
        ShapeNetPart,
        root=DATASETS_DIR,
        split="test",
        show_progress=False,
    ),
    "scanobjectnn": partial(
        ScanObjectNN,
        root=DATASETS_DIR,
        split="split1",
        background=False,
        train=False,
        show_progress=False,
    ),
    "scannet20": partial(
        ScanNet20,
        root=DATASETS_DIR,
        split="val",
        show_progress=False,
    ),
    "scannet20_blocks": _scannet20_blocks,
    "semantickitti": partial(
        SemanticKITTI,
        root=DATASETS_DIR,
        sequences=("00",),
    ),
}

PRETRAINED_MODELS: List[Tuple[str, str, str]] = [
    # ModelNet40 based models
    ("pointmlp-base.modelnet40", "classification", "modelnet_resampled"),
    ("pointmlp-elite.modelnet40", "classification", "modelnet_resampled"),
    ("point-mamba-base.modelnet40", "classification", "modelnet_resampled"),
    ("pointnext-sm-c64.modelnet40", "classification", "modelnet_resampled"),
    ("pointconv-density-base.modelnet40", "classification", "modelnet_resampled"),
    ("dgcnn-antao.modelnet40.1024", "classification", "modelnet_resampled"),
    ("dgcnn-antao.modelnet40.2048", "classification", "modelnet_resampled"),
    ("pointnet2-yanx27-ssg.modelnet40", "classification", "modelnet_resampled"),
    ("pointnet2-yanx27-msg.modelnet40", "classification", "modelnet_resampled"),
    ("pointnet2-openpoints.modelnet40", "classification", "modelnet_resampled"),
    ("point-mae-base.modelnet40", "classification", "modelnet_resampled"),
    ("point-mae-base.modelnet40-8k", "classification", "modelnet_resampled"),
    ("point-bert-base.modelnet40", "classification", "modelnet_resampled"),
    ("point-bert-base.modelnet40-4k", "classification", "modelnet_resampled"),
    ("point-bert-base.modelnet40-8k", "classification", "modelnet_resampled"),
    ("point-m2ae-base.modelnet40", "classification", "modelnet_resampled"),
    *[(f"pointgpt-cguangyan-{s}.modelnet40", "classification", "modelnet_resampled") for s in ("s", "b", "l")],
    *[(f"pointgpt-cguangyan-{s}.modelnet40-8k", "classification", "modelnet_resampled") for s in ("s", "b", "l")],
    # S3DIS based models
    ("kpfcnn-base.s3dis", "segmentation", "s3dis_hdf5"),
    ("kpfcnn-base-sm.s3dis", "segmentation", "s3dis_hdf5"),
    ("kpfcnn-base-deform.s3dis", "segmentation", "s3dis_hdf5"),
    ("kpfcnn-base-sm-deform.s3dis", "segmentation", "s3dis_hdf5"),
    *[(f"pointnext-sm.s3dis-area{i}", "segmentation", "s3dis_hdf5") for i in range(1, 7)],
    *[(f"pointnext-base.s3dis-area{i}", "segmentation", "s3dis_hdf5") for i in range(1, 7)],
    *[(f"pointnext-lg.s3dis-area{i}", "segmentation", "s3dis_hdf5") for i in range(1, 7)],
    *[(f"pointnext-xl.s3dis-area{i}", "segmentation", "s3dis_hdf5") for i in range(1, 6)],
    *[(f"dgcnn-antao.s3dis.area{i}", "segmentation", "s3dis_hdf5") for i in range(1, 7)],
    ("pointnet2-yanx27.s3dis-area5", "segmentation", "s3dis_hdf5"),
    *[(f"pointnet2-openpoints.s3dis-area{i}", "segmentation", "s3dis") for i in range(1, 7)],
    ("pvcnn-mit-han-lab.s3dis-area5", "segmentation", "s3dis_hdf5"),
    ("ptv3-base.s3dis-area5", "segmentation", "s3dis"),
    # ShapenetPart based models
    ("pointnext-sm.shapenetpart", "segmentation", "shapenetpart"),
    ("pointnext-sm-c64.shapenetpart", "segmentation", "shapenetpart"),
    ("pointnext-sm-c160.shapenetpart", "segmentation", "shapenetpart"),
    ("dgcnn-antao.shapenetpart", "segmentation", "shapenetpart"),
    ("point-mae-base.shapenetpart", "segmentation", "shapenetpart"),
    ("point-m2ae-base.shapenetpart", "segmentation", "shapenetpart"),
    # ScanObjectNN based models
    ("pointmlp-base.scanobjectnn", "classification", "scanobjectnn"),
    ("pointmlp-elite.scanobjectnn", "classification", "scanobjectnn"),
    ("point-mamba-base.scanobjectnn", "classification", "scanobjectnn"),
    ("point-mamba-base.scanobjectnn-nobg", "classification", "scanobjectnn"),
    ("point-mamba-base.scanobjectnn-augmentedrot-scale75", "classification", "scanobjectnn"),
    ("pointnext-sm.scanobjectnn", "classification", "scanobjectnn"),
    ("pointnet2-openpoints.scanobjectnn", "classification", "scanobjectnn"),
    ("point-mae-base.scanobjectnn-objbg", "classification", "scanobjectnn"),
    ("point-mae-base.scanobjectnn-objonly", "classification", "scanobjectnn"),
    ("point-mae-base.scanobjectnn-hardest", "classification", "scanobjectnn"),
    ("point-bert-base.scanobjectnn-objonly", "classification", "scanobjectnn"),
    ("point-bert-base.scanobjectnn-objbg", "classification", "scanobjectnn"),
    ("point-bert-base.scanobjectnn-hardest", "classification", "scanobjectnn"),
    ("point-m2ae-base.scanobjectnn-hardest", "classification", "scanobjectnn"),
    ("point-m2ae-base.scanobjectnn-objbg", "classification", "scanobjectnn"),
    *[
        (f"pointgpt-cguangyan-{s}.scanobjectnn-{v}", "classification", "scanobjectnn")
        for s in ("s", "b", "l")
        for v in ("hardest", "objbg", "objonly")
    ],
    # ScanNet20 based models
    ("sonata-lp.scannet20", "segmentation", "scannet20"),
    ("concerto-large-lp.scannet20", "segmentation", "scannet20"),
    ("utonia-lp.scannet20", "segmentation", "scannet20"),
    ("ptv3-base.scannet20", "segmentation", "scannet20"),
    ("ptv3-base.scannet200", "segmentation", "scannet20"),
    ("octformer-base.scannet20", "segmentation", "scannet20"),
    ("octformer-base.scannet200", "segmentation", "scannet20"),
    ("dgcnn-antao.scannet20", "segmentation", "scannet20_blocks"),
    ("spunet-v1m1.scannet20", "segmentation", "scannet20"),
    # SemanticKITTI based models
    ("randlanet-tsunghanwu.semantickitti", "segmentation", "semantickitti"),
    ("spvcnn-30gmacs.semantickitti", "segmentation", "semantickitti"),
    ("spvcnn-47gmacs.semantickitti", "segmentation", "semantickitti"),
    ("spvcnn-119gmacs.semantickitti", "segmentation", "semantickitti"),
]


def _skip_if_deps_missing(model_name: str) -> None:
    """Skip if a backend the model needs isn't installed / no CUDA."""
    if model_name.startswith("point-mamba") and not _MAMBA_SSM_AVAILABLE:
        pytest.skip("mamba_ssm is not installed")
    if model_name.startswith("octformer") and not _DWCONV_AVAILABLE:
        pytest.skip("dwconv is not installed")
    if model_name.startswith(("sonata", "concerto", "utonia")) and not _FLASH_ATTN_AVAILABLE:
        pytest.skip("flash_attn is not installed")
    if model_name.startswith("spvcnn") and not _TORCHSPARSE_AVAILABLE:
        pytest.skip("torchsparse is not installed")
    if model_name.startswith("spunet") and not _SPCONV_AVAILABLE:
        pytest.skip("spconv is not installed")
    if model_name.startswith("pointgpt") and not _TORCH_CLUSTER_AVAILABLE:
        pytest.skip("torch-cluster is not installed")
    if model_name.startswith(("point-mamba", "spvcnn", "spunet")) and not torch.cuda.is_available():
        pytest.skip(f"{model_name} requires CUDA, none available")


def _check_output(
    output: Tensor,
    model_name: str,
    force_regen: bool,
    models_dir: Path,
) -> None:
    """Compare `output` against the safetensors snapshot, regenerating on miss/force."""
    fname = f"{model_name}.safetensors"
    local = models_dir / fname
    actual = output.detach().cpu().contiguous().to(torch.float32)
    src = SNAPSHOTS_DIR / fname
    if force_regen or not src.exists():
        src.parent.mkdir(parents=True, exist_ok=True)
        save_file({"output": actual}, src.as_posix())
        pytest.fail(f"Regenerated {src.as_posix()!r}")

    expected = load_file(local.as_posix())["output"]
    assert tuple(actual.shape) == tuple(expected.shape), (
        f"{model_name}: output shape changed {tuple(expected.shape)} -> {tuple(actual.shape)}"
    )
    if not torch.allclose(actual, expected, atol=ATOL, rtol=RTOL):
        diff = (actual - expected).abs().max().item()
        raise AssertionError(f"{model_name}: output values changed (max abs diff {diff:.6g}, atol={ATOL}, rtol={RTOL})")


@pytest.mark.pretrained
@pytest.mark.parametrize("model_name,task,dataset_name", PRETRAINED_MODELS)
def test_pretrained_model(
    model_name: str,
    task: str,
    dataset_name: str,
    force_regen: bool,
    models_dir_factory: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("torch_pointcloud.utils.cluster.FPS_RANDOM_START", False)

    _skip_if_deps_missing(model_name)
    models_dir = models_dir_factory("*.safetensors")

    # Load the pretrained model
    model, info = create_model(model_name, task=task, pretrained=True, return_info=True)  # type: ignore[call-overload]

    # Load the dataset / dataloader
    dataset = DATASET_REGISTRY[dataset_name](transform=info["transforms"])
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate)

    # Get first batch of data
    data = next(iter(dataloader))

    # Move data to selected device
    data = {k: v.to(DEVICE) if hasattr(v, "to") else v for k, v in data.items()}
    model = model.to(DEVICE)

    # Inspect and keep only required arguments from forward method
    sig = inspect.signature(model.forward)
    kwargs = {a: data.get(a) for a in sig.parameters if a != "self"}
    if "depth" in kwargs and "octree" in data:
        kwargs["depth"] = data["octree"].depth

    # Run inference (eval mode)
    model.eval()
    with torch.no_grad():
        output = model(**kwargs)

    # Ensure output matches snapshot
    _check_output(output, model_name, force_regen, models_dir)


@pytest.mark.pretrained
@pytest.mark.parametrize(
    "model_name,dataset_name",
    [
        ("oneformer3d-base.s3dis-area5", "s3dis"),
        ("oneformer3d-base.scannet20", "scannet20"),
    ],
)
def test_pretrained_oneformer3d(
    model_name: str,
    dataset_name: str,
    force_regen: bool,
    models_dir_factory: Callable[..., Path],
) -> None:
    """OneFormer3D returns a dict of per-scene lists, so it needs its own snapshot test.

    Run on a single fixture scene. ScanNet exercises superpoint pooling (the fixtures
    have no real superpoints, so one per voxel is used); S3DIS runs on voxel features.
    The snapshot is the concatenation of `cls_preds` (and `sem_preds` when present).
    """
    if not (_SPCONV_AVAILABLE and _TORCH_SCATTER_AVAILABLE):
        pytest.skip("spconv / torch_scatter is not installed")
    if not torch.cuda.is_available():
        pytest.skip("oneformer3d requires CUDA, none available")
    models_dir = models_dir_factory("*.safetensors")

    model, info = create_model(model_name, task="segmentation", pretrained=True, return_info=True)
    dataset = DATASET_REGISTRY[dataset_name](transform=info["transforms"])
    data = collate([dataset[0]])
    data = {k: v.to(DEVICE) if hasattr(v, "to") else v for k, v in data.items()}
    model = model.to(DEVICE).eval()

    x, pos_grid, batch = data["x"], data["pos_grid"], data["batch"].long()
    with torch.no_grad():
        if model.superpoint_pooling:
            out = model(x, pos_grid.long(), batch, data["inverse"], data["inverse"])
        else:
            out = model(x, pos_grid.long(), batch)

    parts = list(out["cls_preds"])
    if "sem_preds" in out:
        parts += list(out["sem_preds"])
    reduced = torch.cat([p.reshape(-1) for p in parts])
    _check_output(reduced, model_name, force_regen, models_dir)


@pytest.mark.pretrained
@pytest.mark.parametrize("model_name", ["votenet-fair-base.scannet", "votenet-fair-base.sunrgbd"])
def test_pretrained_votenet(
    model_name: str,
    force_regen: bool,
    models_dir_factory: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """VoteNet returns a dict of dense proposal tensors, so it needs its own snapshot test.

    There is no in-repo SUN RGB-D / ScanNet *detection* fixture, so a fixed synthetic cloud with
    deterministic FPS is used; this still pins the pretrained weights + decode against regressions.
    The snapshot is the concatenation of the proposal `center`, `objectness_scores` and `sem_cls_scores`.
    """
    if not (_TORCH_CLUSTER_AVAILABLE and _TORCH_SCATTER_AVAILABLE):
        pytest.skip("torch-cluster / torch-scatter is not installed")

    monkeypatch.setattr("torch_pointcloud.utils.cluster.FPS_RANDOM_START", False)
    models_dir = models_dir_factory("*.safetensors")

    model, _ = create_model(model_name, task="detection", pretrained=True, return_info=True)
    model = model.to(DEVICE).eval()

    # Seed the synthetic input *after* `create_model`: building a pretrained model random-inits its
    # parameters (consuming RNG) before loading weights, so seeding earlier makes the input depend on it.
    torch.manual_seed(0)
    n_per_scene, batch_size = 3000, 2
    pos = (torch.rand(n_per_scene * batch_size, 3) * 4.0).to(DEVICE)
    x = torch.rand(n_per_scene * batch_size, model.in_channels).to(DEVICE)
    batch = torch.arange(batch_size).repeat_interleave(n_per_scene).to(DEVICE)

    with torch.no_grad():
        out = model(x, pos, batch)

    reduced = torch.cat(
        [out["center"].reshape(-1), out["objectness_scores"].reshape(-1), out["sem_cls_scores"].reshape(-1)]
    )
    _check_output(reduced, model_name, force_regen, models_dir)


@pytest.mark.pretrained
@pytest.mark.parametrize("model_name", ["3detr-fair.scannet", "3detr-fair.sunrgbd", "3detr-fair-m.scannet"])
def test_pretrained_detr3d(
    model_name: str,
    force_regen: bool,
    models_dir_factory: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """3DETR returns a dict of per-query predictions, so it needs its own snapshot test.

    There is no in-repo ScanNet / SUN RGB-D *detection* fixture, so a fixed synthetic cloud with
    deterministic query FPS is used; this still pins the pretrained weights against regressions. The
    snapshot is the logits-level `sem_cls_logits` + `center_unnormalized` + `size_unnormalized` +
    `angle_logits`, not the decoded `angle_continuous`: the decode argmaxes the angle bins and wraps the
    result through a ±pi branch cut, which flips under fp32 noise; the raw logits are continuous and stable.
    """
    if not (_TORCH_CLUSTER_AVAILABLE and _TORCH_SCATTER_AVAILABLE):
        pytest.skip("torch-cluster / torch-scatter is not installed")

    monkeypatch.setattr("torch_pointcloud.utils.cluster.FPS_RANDOM_START", False)
    models_dir = models_dir_factory("*.safetensors")

    model, _ = create_model(model_name, task="detection", pretrained=True, return_info=True)
    model = model.to(DEVICE).eval()

    # Seed the input *after* `create_model`: building a pretrained model random-inits its parameters
    # (consuming RNG) before loading weights, so seeding earlier would couple the input to that init.
    torch.manual_seed(0)
    n_per_scene, batch_size = 3000, 2
    pos = (torch.rand(n_per_scene * batch_size, 3) * 4.0).to(DEVICE)
    batch = torch.arange(batch_size).repeat_interleave(n_per_scene).to(DEVICE)

    with torch.no_grad():
        out = model(None, pos, batch)

    reduced = torch.cat(
        [
            out["sem_cls_logits"].reshape(-1),
            out["center_unnormalized"].reshape(-1),
            out["size_unnormalized"].reshape(-1),
            out["angle_logits"].reshape(-1),
        ]
    )
    _check_output(reduced, model_name, force_regen, models_dir)


@pytest.mark.pretrained
@pytest.mark.parametrize("model_name", ["lion-mamba-happinesslz.nuscenes"])
def test_pretrained_lion(
    model_name: str,
    force_regen: bool,
    models_dir_factory: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LION returns a TransFusion-head dict, so it needs its own snapshot test.

    There is no in-repo nuScenes *detection* fixture, so a fixed synthetic cloud inside the model's
    point-cloud range is used (LION voxelizes internally); this still pins the pretrained weights against
    regressions. The snapshot is the order-invariant `dense_heatmap` (the pre-query BEV class heatmap), not
    the per-query outputs: those are gathered at the top-k heatmap positions, and a ~4e-4 heatmap
    perturbation flips an argsort tie, reordering the queries; the dense heatmap has no such tie sensitivity.
    """
    if not (_MAMBA_SSM_AVAILABLE and _SPCONV_AVAILABLE):
        pytest.skip("mamba-ssm / spconv is not installed")
    if not torch.cuda.is_available():
        pytest.skip("lion requires CUDA, none available")

    monkeypatch.setattr("torch_pointcloud.utils.cluster.FPS_RANDOM_START", False)
    models_dir = models_dir_factory("*.safetensors")

    model, _ = create_model(model_name, task="detection", pretrained=True, return_info=True)
    model = model.to(DEVICE).eval()

    pc_range = (-54.0, -54.0, -5.0, 54.0, 54.0, 3.0)
    # Seed the input *after* `create_model`: building a pretrained model random-inits its parameters
    # (consuming RNG) before loading weights, so seeding earlier would couple the input to that init.
    torch.manual_seed(0)
    n_per_scene, batch_size = 8000, 2
    pos = torch.rand(n_per_scene * batch_size, 3)
    for d in range(3):
        pos[:, d] = pos[:, d] * (pc_range[d + 3] - pc_range[d]) + pc_range[d]
    x = torch.rand(n_per_scene * batch_size, model.in_channels - 3)
    pos, x = pos.to(DEVICE), x.to(DEVICE)
    batch = torch.arange(batch_size).repeat_interleave(n_per_scene).to(DEVICE)

    with torch.no_grad():
        out = model(x, pos, batch)

    _check_output(out["dense_heatmap"].reshape(-1), model_name, force_regen, models_dir)


ANCHOR_DETECTION_MODELS: List[Tuple[str, Tuple[float, ...], Tuple[float, ...], int, int]] = [
    ("pointpillars-openpcdet.kitti", (0.0, -39.68, -3.0, 69.12, 39.68, 1.0), (0.16, 0.16, 4.0), 32, 40000),
    ("second-openpcdet.kitti", (0.0, -39.68, -3.0, 69.12, 39.68, 1.0), (0.05, 0.05, 0.1), 5, 40000),
    ("pointpillars-openpcdet-multihead.nuscenes", (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0), (0.2, 0.2, 8.0), 20, 30000),
    ("second-openpcdet-multihead.nuscenes", (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0), (0.1, 0.1, 0.2), 10, 60000),
]


@pytest.mark.pretrained
@pytest.mark.parametrize("model_name,pc_range,voxel_size,max_num_points,max_num_voxels", ANCHOR_DETECTION_MODELS)
def test_pretrained_anchor_detection(
    model_name: str,
    pc_range: Tuple[float, ...],
    voxel_size: Tuple[float, ...],
    max_num_points: int,
    max_num_voxels: int,
    force_regen: bool,
    models_dir_factory: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PointPillars / SECOND emit a dict of dense anchor predictions, so they need their own snapshot test.

    There is no in-repo KITTI / nuScenes *detection* fixture small enough for a forward, so a fixed
    synthetic cloud inside the model's point-cloud range is used; this still pins the pretrained weights
    against regressions. The synthetic points are hard-voxelized exactly as the registered `HardVoxelize`
    transform would, then fed to the model as voxel keys. TF32 is disabled so the spconv forward is
    deterministic to ~1e-5 across processes (TF32 rounding otherwise flips a borderline box's direction
    bin). The snapshot is the raw RPN `cls` + `box`, not the decoded `batch_box`: the decode reconstructs
    heading via `atan2`, whose ±pi branch cut flips a box near that heading by 2*pi under even fp32-level
    noise, which a value snapshot cannot track; the raw sincos box residuals are continuous and stable.
    """
    if not _SPCONV_AVAILABLE:
        pytest.skip("spconv is not installed")
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    models_dir = models_dir_factory("*.safetensors")

    model, _ = create_model(model_name, task="detection", pretrained=True, return_info=True)
    model = model.to(DEVICE).eval()

    # Seed the input *after* `create_model`: building a pretrained model random-inits its parameters
    # (consuming RNG) before loading weights, so seeding earlier would couple the input to that init.
    torch.manual_seed(0)
    n_per_scene, batch_size = 8000, 2
    pos = torch.rand(n_per_scene * batch_size, 3)
    for d in range(3):
        pos[:, d] = pos[:, d] * (pc_range[d + 3] - pc_range[d]) + pc_range[d]
    x = torch.rand(n_per_scene * batch_size, model.in_channels - 3)
    pos, x = pos.to(DEVICE), x.to(DEVICE)
    batch = torch.arange(batch_size).repeat_interleave(n_per_scene).to(DEVICE)

    points = torch.cat([pos, x], dim=1)
    voxels, voxel_indices, num_points = hard_voxelize(
        points, batch, voxel_size, pc_range, max_num_points, max_num_voxels
    )
    with torch.no_grad():
        out = model(voxels, voxel_indices[:, 1:], num_points, voxel_indices[:, 0])

    parts: List[Tensor] = []
    for key in ("cls", "box"):
        value = out[key]
        parts += [t.reshape(-1) for t in value] if isinstance(value, (list, tuple)) else [value.reshape(-1)]
    _check_output(torch.cat(parts), model_name, force_regen, models_dir)
