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
    _FVDB_AVAILABLE,
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
    ("pointmlp-base.modelnet40.xu-ma", "classification", "modelnet_resampled"),
    ("pointmlp-elite.modelnet40.xu-ma", "classification", "modelnet_resampled"),
    ("point-mamba-base.modelnet40.dingkang-liang", "classification", "modelnet_resampled"),
    ("pointnext-sm-c64.modelnet40.openpoints", "classification", "modelnet_resampled"),
    ("pointconv-density-base.modelnet40.wenxuan-wu", "classification", "modelnet_resampled"),
    ("dgcnn.modelnet40-1024.an-tao", "classification", "modelnet_resampled"),
    ("dgcnn.modelnet40-2048.an-tao", "classification", "modelnet_resampled"),
    ("pointnet2-ssg.modelnet40.xu-yan", "classification", "modelnet_resampled"),
    ("pointnet2-msg.modelnet40.xu-yan", "classification", "modelnet_resampled"),
    ("pointnet2.modelnet40.openpoints", "classification", "modelnet_resampled"),
    ("point-mae-base.modelnet40.yatian-pang", "classification", "modelnet_resampled"),
    ("point-mae-base.modelnet40-8k.yatian-pang", "classification", "modelnet_resampled"),
    ("point-bert-base.modelnet40.xumin-yu", "classification", "modelnet_resampled"),
    ("point-bert-base.modelnet40-4k.xumin-yu", "classification", "modelnet_resampled"),
    ("point-bert-base.modelnet40-8k.xumin-yu", "classification", "modelnet_resampled"),
    ("point-m2ae-base.modelnet40.renrui-zhang", "classification", "modelnet_resampled"),
    *[(f"pointgpt-cguangyan-{s}.modelnet40", "classification", "modelnet_resampled") for s in ("s", "b", "l")],
    *[(f"pointgpt-cguangyan-{s}.modelnet40-8k", "classification", "modelnet_resampled") for s in ("s", "b", "l")],
    # S3DIS based models
    ("kpfcnn-base.s3dis.hugues-thomas", "segmentation", "s3dis_hdf5"),
    ("kpfcnn-base-sm.s3dis.hugues-thomas", "segmentation", "s3dis_hdf5"),
    ("kpfcnn-base-deform.s3dis.hugues-thomas", "segmentation", "s3dis_hdf5"),
    ("kpfcnn-base-sm-deform.s3dis.hugues-thomas", "segmentation", "s3dis_hdf5"),
    *[(f"pointnext-sm.s3dis-area{i}", "segmentation", "s3dis_hdf5") for i in range(1, 7)],
    *[(f"pointnext-base.s3dis-area{i}", "segmentation", "s3dis_hdf5") for i in range(1, 7)],
    *[(f"pointnext-lg.s3dis-area{i}", "segmentation", "s3dis_hdf5") for i in range(1, 7)],
    *[(f"pointnext-xl.s3dis-area{i}", "segmentation", "s3dis_hdf5") for i in range(1, 6)],
    *[(f"dgcnn-antao.s3dis.area{i}", "segmentation", "s3dis_hdf5") for i in range(1, 7)],
    ("pointnet2.s3dis-area5.xu-yan", "segmentation", "s3dis_hdf5"),
    *[(f"pointnet2-openpoints.s3dis-area{i}", "segmentation", "s3dis") for i in range(1, 7)],
    ("pvcnn.s3dis-area5.mit-han-lab", "segmentation", "s3dis_hdf5"),
    ("ptv3-base.s3dis-area5.pointcept", "segmentation", "s3dis"),
    # ShapenetPart based models
    ("pointnext-sm.shapenetpart.openpoints", "segmentation", "shapenetpart"),
    ("pointnext-sm-c64.shapenetpart.openpoints", "segmentation", "shapenetpart"),
    ("pointnext-sm-c160.shapenetpart.openpoints", "segmentation", "shapenetpart"),
    ("dgcnn.shapenetpart.an-tao", "segmentation", "shapenetpart"),
    ("point-mae-base.shapenetpart.yatian-pang", "segmentation", "shapenetpart"),
    ("point-m2ae-base.shapenetpart.renrui-zhang", "segmentation", "shapenetpart"),
    # ScanObjectNN based models
    ("pointmlp-base.scanobjectnn.xu-ma", "classification", "scanobjectnn"),
    ("pointmlp-elite.scanobjectnn.xu-ma", "classification", "scanobjectnn"),
    ("point-mamba-base.scanobjectnn.dingkang-liang", "classification", "scanobjectnn"),
    ("point-mamba-base.scanobjectnn-nobg.dingkang-liang", "classification", "scanobjectnn"),
    ("point-mamba-base.scanobjectnn-augmentedrot-scale75.dingkang-liang", "classification", "scanobjectnn"),
    ("pointnext-sm.scanobjectnn.openpoints", "classification", "scanobjectnn"),
    ("pointnet2.scanobjectnn.openpoints", "classification", "scanobjectnn"),
    ("point-mae-base.scanobjectnn-objbg.yatian-pang", "classification", "scanobjectnn"),
    ("point-mae-base.scanobjectnn-objonly.yatian-pang", "classification", "scanobjectnn"),
    ("point-mae-base.scanobjectnn-hardest.yatian-pang", "classification", "scanobjectnn"),
    ("point-bert-base.scanobjectnn-objonly.xumin-yu", "classification", "scanobjectnn"),
    ("point-bert-base.scanobjectnn-objbg.xumin-yu", "classification", "scanobjectnn"),
    ("point-bert-base.scanobjectnn-hardest.xumin-yu", "classification", "scanobjectnn"),
    ("point-m2ae-base.scanobjectnn-hardest.renrui-zhang", "classification", "scanobjectnn"),
    ("point-m2ae-base.scanobjectnn-objbg.renrui-zhang", "classification", "scanobjectnn"),
    *[
        (f"pointgpt-cguangyan-{s}.scanobjectnn-{v}", "classification", "scanobjectnn")
        for s in ("s", "b", "l")
        for v in ("hardest", "objbg", "objonly")
    ],
    # ScanNet20 based models
    ("sonata-lp.scannet20.fair", "segmentation", "scannet20"),
    ("concerto-large-lp.scannet20.pointcept", "segmentation", "scannet20"),
    ("utonia-lp.scannet20.pointcept", "segmentation", "scannet20"),
    ("ptv3-base.scannet20.pointcept", "segmentation", "scannet20"),
    ("ptv3-base.scannet200.pointcept", "segmentation", "scannet20"),
    ("octformer-base.scannet20.octree-nn", "segmentation", "scannet20"),
    ("octformer-base.scannet200.octree-nn", "segmentation", "scannet20"),
    ("dgcnn.scannet20.an-tao", "segmentation", "scannet20_blocks"),
    ("spunet-v1m1.scannet20.pointcept", "segmentation", "scannet20"),
    # SemanticKITTI based models
    ("randlanet.semantickitti.tsung-han-wu", "segmentation", "semantickitti"),
    ("spvcnn-30gmacs.semantickitti.mit-han-lab", "segmentation", "semantickitti"),
    ("spvcnn-47gmacs.semantickitti.mit-han-lab", "segmentation", "semantickitti"),
    ("spvcnn-119gmacs.semantickitti.mit-han-lab", "segmentation", "semantickitti"),
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


@pytest.fixture(autouse=True)
def _torchsparse_kmap_mode() -> None:
    """torch >= 2.10 rejects torchsparse's default `hashmap_on_the_fly` downsample kmap builder
    (its legacy `make_variable` call hits `set_stride` on a detached coords tensor). The `hashmap`
    builder takes a different C++ path and is unaffected."""
    if not _TORCHSPARSE_AVAILABLE:
        return

    import torchsparse.nn.functional as spF

    config = spF.conv_config.get_default_conv_config()
    config.kmap_mode = "hashmap"
    spF.conv_config.set_global_conv_config(config)


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
    dataset = DATASET_REGISTRY[dataset_name](transform=info["transform"])
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
        ("oneformer3d-base.s3dis-area5.danila-rukhovich", "s3dis"),
        ("oneformer3d-base.scannet20.danila-rukhovich", "scannet20"),
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
    dataset = DATASET_REGISTRY[dataset_name](transform=info["transform"])
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
@pytest.mark.parametrize("model_name", ["votenet.scannet.fair", "votenet.sunrgbd.fair"])
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
@pytest.mark.parametrize("model_name", ["3detr.scannet.fair", "3detr.sunrgbd.fair", "3detr-m.scannet.fair"])
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
@pytest.mark.parametrize("model_name", ["pointrcnn.kitti.openpcdet"])
def test_pretrained_pointrcnn(
    model_name: str,
    force_regen: bool,
    models_dir_factory: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PointRCNN returns a dict of per-ROI stage-2 predictions, so it needs its own snapshot test.

    There is no in-repo KITTI *detection* fixture, so a fixed synthetic cloud inside the model's
    point-cloud range with deterministic FPS is used; this still pins the pretrained weights against
    regressions. The snapshot is the stage-2 confidence logit `rcnn_cls` + the refined box center/size
    `boxes[:, :6]`, not `boxes[:, 6]`: the stage-1 box coder reconstructs heading via `atan2`, whose ±pi
    branch cut flips a box near that heading under fp32 noise; the center/size residuals are stable.
    """
    if not (_TORCH_CLUSTER_AVAILABLE and _TORCH_SCATTER_AVAILABLE):
        pytest.skip("torch-cluster / torch-scatter is not installed")

    monkeypatch.setattr("torch_pointcloud.utils.cluster.FPS_RANDOM_START", False)
    models_dir = models_dir_factory("*.safetensors")

    model, _ = create_model(model_name, task="detection", pretrained=True, return_info=True)
    model = model.to(DEVICE).eval()

    pc_range = (0.0, -40.0, -3.0, 70.4, 40.0, 1.0)
    # Seed the input *after* `create_model`: building a pretrained model random-inits its parameters
    # (consuming RNG) before loading weights, so seeding earlier would couple the input to that init.
    torch.manual_seed(0)
    n_per_scene, batch_size = 16384, 2
    pos = torch.rand(n_per_scene * batch_size, 3)
    for d in range(3):
        pos[:, d] = pos[:, d] * (pc_range[d + 3] - pc_range[d]) + pc_range[d]
    x = torch.rand(n_per_scene * batch_size, model.in_channels - 3)
    pos, x = pos.to(DEVICE), x.to(DEVICE)
    batch = torch.arange(batch_size).repeat_interleave(n_per_scene).to(DEVICE)

    with torch.no_grad():
        out = model(x, pos, batch)

    reduced = torch.cat([out["rcnn_cls"].reshape(-1), out["boxes"][:, :6].reshape(-1)])
    _check_output(reduced, model_name, force_regen, models_dir)


@pytest.mark.pretrained
@pytest.mark.parametrize("model_name", ["lion-mamba.nuscenes.zhe-liu"])
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


@pytest.mark.pretrained
@pytest.mark.parametrize("model_name", ["voxel-mamba.waymo"])
def test_seeded_voxel_mamba(
    model_name: str,
    force_regen: bool,
    models_dir_factory: Callable[..., Path],
) -> None:
    """Voxel Mamba has no public weights, so a fixed-seed build pins the numerics instead.

    There is no in-repo Waymo fixture, so a fixed synthetic cloud inside the model's point-cloud
    range is used; the forward is deterministic run-to-run. The snapshot is the concatenation of the
    six flattened center-head maps.
    """
    if not (_MAMBA_SSM_AVAILABLE and _SPCONV_AVAILABLE):
        pytest.skip("mamba-ssm / spconv is not installed")
    if not torch.cuda.is_available():
        pytest.skip("voxel-mamba requires CUDA, none available")
    models_dir = models_dir_factory("*.safetensors")

    # Seed the build too: with no checkpoint to load, the seeded random init stands in for weights.
    torch.manual_seed(0)
    model = create_model(model_name, task="detection").to(DEVICE).eval()

    pc_range = (-74.88, -74.88, -2.0, 74.88, 74.88, 4.0)
    torch.manual_seed(0)
    n_per_scene = 6000
    pos = torch.rand(n_per_scene, 3)
    for d in range(3):
        pos[:, d] = pos[:, d] * (pc_range[d + 3] - pc_range[d]) + pc_range[d]
    x = torch.rand(n_per_scene, model.in_channels - 3)
    pos, x = pos.to(DEVICE), x.to(DEVICE)
    batch = torch.zeros(n_per_scene, dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        out = model(x, pos, batch)

    reduced = torch.cat([out[k].reshape(-1) for k in ("center", "center_z", "dim", "rot", "iou", "heatmap")])
    _check_output(reduced, model_name, force_regen, models_dir)


@pytest.mark.pretrained
@pytest.mark.parametrize("model_name", ["voxelnext.nuscenes.openpcdet"])
def test_pretrained_voxelnext(
    model_name: str,
    force_regen: bool,
    models_dir_factory: Callable[..., Path],
) -> None:
    """VoxelNeXt returns per-voxel sparse predictions, so it needs its own snapshot test.

    There is no in-repo nuScenes *detection* fixture, so a fixed synthetic cloud inside the model's
    point-cloud range is hard-voxelized exactly as the registered transform would. spconv emits its
    sparse output rows in a nondeterministic ORDER (the per-voxel values are stable to ~3e-6 and the
    voxel set is bitwise run-stable), so every head tensor is sorted by its flattened BEV voxel index
    before snapshotting. The snapshot concatenates the sorted voxel indices with the per-group `hm`
    and box-attribute maps.
    """
    if not _SPCONV_AVAILABLE:
        pytest.skip("spconv is not installed")
    if not torch.cuda.is_available():
        pytest.skip("voxelnext requires CUDA, none available")
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

    points = torch.cat([pos, x], dim=1)
    voxels, voxel_indices, num_points = hard_voxelize(points, batch, (0.075, 0.075, 0.2), pc_range, 10, 160000)
    with torch.no_grad():
        out = model(voxels, voxel_indices[:, 1:], num_points, voxel_indices[:, 0])

    bev = out["voxel_indices"].long()
    size = int(bev.max().item()) + 1
    order = torch.argsort((bev[:, 0] * size + bev[:, 1]) * size + bev[:, 2])
    parts = [bev[order].reshape(-1).float()]
    for key in ("hm", "center", "center_z", "dim", "rot", "vel"):
        parts += [t[order].reshape(-1) for t in out[key]]
    _check_output(torch.cat(parts), model_name, force_regen, models_dir)


ANCHOR_DETECTION_MODELS: List[Tuple[str, Tuple[float, ...], Tuple[float, ...], int, int]] = [
    ("pointpillars.kitti.openpcdet", (0.0, -39.68, -3.0, 69.12, 39.68, 1.0), (0.16, 0.16, 4.0), 32, 40000),
    ("second.kitti.openpcdet", (0.0, -39.68, -3.0, 69.12, 39.68, 1.0), (0.05, 0.05, 0.1), 5, 40000),
    ("pointpillars-multihead.nuscenes.openpcdet", (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0), (0.2, 0.2, 8.0), 20, 30000),
    ("second-multihead.nuscenes.openpcdet", (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0), (0.1, 0.1, 0.2), 10, 60000),
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


@pytest.mark.pretrained
@pytest.mark.parametrize(
    "model_name",
    [
        "xcube-vae-coarse.shapenet-chair.nvidia",
        "xcube-vae-fine.shapenet-chair.nvidia",
        "xcube-vae-coarse.shapenet-car.nvidia",
        "xcube-vae-fine.shapenet-car.nvidia",
        "xcube-vae-coarse.shapenet-plane.nvidia",
        "xcube-vae-fine.shapenet-plane.nvidia",
    ],
)
def test_pretrained_xcube_vae(
    model_name: str,
    force_regen: bool,
    models_dir_factory: Callable[..., Path],
) -> None:
    """XCube VAEs return grids plus jagged features, so they need their own snapshot test.

    There is no in-repo XCube ShapeNet fixture batch, so a fixed synthetic voxel cloud is used; this
    still pins the pretrained weights against regressions. The snapshot is the posterior mean `mu`: the
    decoded structure passes through argmax pruning whose topology can flip under fp32 noise, while the
    posterior is continuous and stable.
    """
    if not _FVDB_AVAILABLE:
        pytest.skip("fvdb is not installed")
    if not torch.cuda.is_available():
        pytest.skip("XCube requires CUDA, none available")

    from torch_pointcloud.models.xcube import XCubeVAE

    models_dir = models_dir_factory("*.safetensors")
    model, _ = create_model(model_name, task="base", pretrained=True, return_info=True)
    assert isinstance(model, XCubeVAE)
    model = model.to(DEVICE).eval()

    torch.manual_seed(0)
    num_points = 4000
    pos = (torch.rand(num_points, 3) - 0.5).to(DEVICE)
    pos = (torch.floor(pos / model.voxel_size) + 0.5) * model.voxel_size
    batch = torch.zeros(num_points, dtype=torch.long, device=DEVICE)
    normal = torch.nn.functional.normalize(torch.randn(num_points, 3), dim=1).to(DEVICE)

    with torch.no_grad():
        mu, _ = model.encode(pos, batch, normal=normal)
    _check_output(mu.reshape(-1), model_name, force_regen, models_dir)


@pytest.mark.pretrained
@pytest.mark.parametrize(
    "model_name",
    [
        "xcube-diffusion-coarse.shapenet-chair.nvidia",
        "xcube-diffusion-fine.shapenet-chair.nvidia",
        "xcube-diffusion-coarse.shapenet-car.nvidia",
        "xcube-diffusion-fine.shapenet-car.nvidia",
        "xcube-diffusion-coarse.shapenet-plane.nvidia",
        "xcube-diffusion-fine.shapenet-plane.nvidia",
    ],
)
def test_pretrained_xcube_diffusion(
    model_name: str,
    force_regen: bool,
    models_dir_factory: Callable[..., Path],
) -> None:
    """XCube diffusion models sample stochastically, so the snapshot pins a single denoising step.

    `denoise` on seeded latents at a fixed timestep is the continuous, deterministic quantity behind
    sampling (the DDIM loop only adds scheduler arithmetic and fresh noise around it); snapshotting it
    pins the converted EMA UNet weights, the conditioning channels and the frozen VAE's `scale_factor`
    against regressions. The dense 16^3 latent grid built for coarse sampling is a valid latent topology
    for the sparse fine UNets too.
    """
    if not _FVDB_AVAILABLE:
        pytest.skip("fvdb is not installed")
    if not torch.cuda.is_available():
        pytest.skip("XCube requires CUDA, none available")

    from torch_pointcloud.models.xcube import XCubeDiffusion

    models_dir = models_dir_factory("*.safetensors")
    model, _ = create_model(model_name, task="base", pretrained=True, return_info=True)
    assert isinstance(model, XCubeDiffusion)
    model = model.to(DEVICE).eval()

    torch.manual_seed(0)
    grid = model.latent_grid(1, DEVICE)
    latents = torch.randn(int(grid.total_voxels), model.vae.latent_channels, device=DEVICE)
    timesteps = torch.tensor([500], device=DEVICE)
    normal = None
    if model.normal_cond:
        normal = torch.nn.functional.normalize(torch.randn(int(grid.total_voxels), 3), dim=1).to(DEVICE)

    with torch.no_grad():
        pred = model.denoise(latents, grid, timesteps, normal=normal)

    del model
    torch.cuda.empty_cache()
    _check_output(pred.reshape(-1), model_name, force_regen, models_dir)
