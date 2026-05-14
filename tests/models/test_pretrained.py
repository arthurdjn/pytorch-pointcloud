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
from typing import Callable, Dict, List, Tuple

import pytest
import torch
from safetensors.torch import load_file, save_file
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from torch_pointcloud.datasets import (
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
    _TORCHSPARSE_AVAILABLE,
)

ATOL = 5e-3
RTOL = 5e-2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATASETS_DIR = DATA_DIR / "datasets"
SNAPSHOTS_DIR = DATA_DIR / "models"

DATASET_REGISTRY: Dict[str, Callable[..., Dataset]] = {
    "modelnet_resampled": partial(
        ModelNetNormalResampled,
        root=DATASETS_DIR,
        variant="40",
        train=False,
        show_progress=False,
    ),
    "s3dis": partial(
        S3DISHdf5,
        root=DATASETS_DIR,
        areas=("Area_5",),
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
    # S3DIS based models
    ("kpfcnn-base.s3dis", "segmentation", "s3dis"),
    ("kpfcnn-base-sm.s3dis", "segmentation", "s3dis"),
    ("kpfcnn-base-deform.s3dis", "segmentation", "s3dis"),
    ("kpfcnn-base-sm-deform.s3dis", "segmentation", "s3dis"),
    *[(f"pointnext-sm.s3dis-area{i}", "segmentation", "s3dis") for i in range(1, 7)],
    *[(f"pointnext-base.s3dis-area{i}", "segmentation", "s3dis") for i in range(1, 7)],
    *[(f"pointnext-lg.s3dis-area{i}", "segmentation", "s3dis") for i in range(1, 7)],
    *[(f"pointnext-xl.s3dis-area{i}", "segmentation", "s3dis") for i in range(1, 6)],
    *[(f"dgcnn-antao.s3dis.area{i}", "segmentation", "s3dis") for i in range(1, 7)],
    # ShapenetPart based models
    ("pointnext-sm.shapenetpart", "segmentation", "shapenetpart"),
    ("pointnext-sm-c64.shapenetpart", "segmentation", "shapenetpart"),
    ("pointnext-sm-c160.shapenetpart", "segmentation", "shapenetpart"),
    ("dgcnn-antao.shapenetpart", "segmentation", "shapenetpart"),
    # ScanObjectNN based models
    ("pointmlp-base.scanobjectnn", "classification", "scanobjectnn"),
    ("pointmlp-elite.scanobjectnn", "classification", "scanobjectnn"),
    ("point-mamba-base.scanobjectnn", "classification", "scanobjectnn"),
    ("point-mamba-base.scanobjectnn-nobg", "classification", "scanobjectnn"),
    ("point-mamba-base.scanobjectnn-augmentedrot-scale75", "classification", "scanobjectnn"),
    ("pointnext-sm.scanobjectnn", "classification", "scanobjectnn"),
    # ScanNet20 based models
    ("sonata-lp.scannet20", "segmentation", "scannet20"),
    ("concerto-large-lp.scannet20", "segmentation", "scannet20"),
    ("utonia-lp.scannet20", "segmentation", "scannet20"),
    ("octformer-base.scannet20", "segmentation", "scannet20"),
    ("octformer-base.scannet200", "segmentation", "scannet20"),
    ("dgcnn-antao.scannet20", "segmentation", "scannet20"),
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
