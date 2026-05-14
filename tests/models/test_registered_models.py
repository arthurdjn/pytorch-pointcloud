import inspect
import json
from pathlib import Path
from typing import Any, Callable, Dict, List

import pytest
import torch
import torch.nn as nn

from torch_pointcloud.models import create_model, list_models
from torch_pointcloud.utils.imports import (
    _DWCONV_AVAILABLE,
    _FLASH_ATTN_AVAILABLE,
    _MAMBA_SSM_AVAILABLE,
    _OCNN_AVAILABLE,
    _SPCONV_AVAILABLE,
    _TORCH_CLUSTER_AVAILABLE,
    _TORCH_SCATTER_AVAILABLE,
    _TORCHSPARSE_AVAILABLE,
)
from torch_pointcloud.utils.octree import build_octree

CLASSIFICATION_MODELS = [
    "dgcnn-antao.modelnet40.1024",
    "dgcnn-antao.modelnet40.2048",
    "kpfcnn.modelnet40",
    "octformer-base.modelnet40",
    "pointcnn-base",
    "pointconv-density-base",
    "pointconv-density-base.modelnet40",
    "point-mamba-base.modelnet40",
    "point-mamba-base.scanobjectnn",
    "point-mamba-base.scanobjectnn-nobg",
    "point-mamba-base.scanobjectnn-augmentedrot-scale75",
    "pointmlp-base",
    "pointmlp-base.modelnet40",
    "pointmlp-base.scanobjectnn",
    "pointmlp-elite",
    "pointmlp-elite.modelnet40",
    "pointmlp-elite.scanobjectnn",
    "pointnext-base",
    "pointnext-lg",
    "pointnext-sm",
    "pointnext-sm-c64.modelnet40",
    "pointnext-sm.scanobjectnn",
    "pointnext-xl",
]
BASE_MODELS = [
    "concerto-base",
    "concerto-large",
    "concerto-small",
    "concerto-tiny",
    "point-mamba-base.pretrain",
    "sonata-base",
    "utonia",
]
SEGMENTATION_MODELS = [
    "concerto-large-lp.scannet20",
    "utonia-lp.scannet20",
    "dgcnn-antao.shapenetpart",
    "dgcnn-antao.s3dis.area1",
    "dgcnn-antao.s3dis.area2",
    "dgcnn-antao.s3dis.area3",
    "dgcnn-antao.s3dis.area4",
    "dgcnn-antao.s3dis.area5",
    "dgcnn-antao.s3dis.area6",
    "dgcnn-antao.scannet20",
    "kpfcnn-base.s3dis",
    "kpfcnn-base-sm.s3dis",
    "kpfcnn-base-deform.s3dis",
    "kpfcnn-base-sm-deform.s3dis",
    "octformer-base.lg",
    "octformer-base.sm",
    "octformer-base.scannet20",
    "octformer-base.scannet200",
    "pointcnn-base",
    "pointmlp-base",
    "pointmlp-elite",
    "pointnext-base",
    "pointnext-base.s3dis-area1",
    "pointnext-base.s3dis-area2",
    "pointnext-base.s3dis-area3",
    "pointnext-base.s3dis-area4",
    "pointnext-base.s3dis-area5",
    "pointnext-base.s3dis-area6",
    "pointnext-lg",
    "pointnext-lg.s3dis-area1",
    "pointnext-lg.s3dis-area2",
    "pointnext-lg.s3dis-area3",
    "pointnext-lg.s3dis-area4",
    "pointnext-lg.s3dis-area5",
    "pointnext-lg.s3dis-area6",
    "pointnext-sm",
    "pointnext-sm.s3dis-area1",
    "pointnext-sm.s3dis-area2",
    "pointnext-sm.s3dis-area3",
    "pointnext-sm.s3dis-area4",
    "pointnext-sm.s3dis-area5",
    "pointnext-sm.s3dis-area6",
    "pointnext-sm.shapenetpart",
    "pointnext-sm-c64.shapenetpart",
    "pointnext-sm-c160.shapenetpart",
    "pointnext-xl",
    "pointnext-xl.s3dis-area1",
    "pointnext-xl.s3dis-area2",
    "pointnext-xl.s3dis-area3",
    "pointnext-xl.s3dis-area4",
    "pointnext-xl.s3dis-area5",
    "pointnext-xl.s3dis-area6",
    "randlanet-tsunghanwu.semantickitti",
    "sonata-lp.scannet20",
    "spunet-v1m1.scannet20",
    "spvcnn-30gmacs.semantickitti",
    "spvcnn-47gmacs.semantickitti",
    "spvcnn-119gmacs.semantickitti",
]


def _skip_if_model_deps_missing(model_name: str) -> None:
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


# Models whose `forward` cannot run on the synthetic `data_factory` input — they expect
# voxelized / serialized / octree-encoded inputs that the factory doesn't emit yet.
_UNSUPPORTED_BY_DATA_FACTORY = frozenset(
    {
        "octformer-base.modelnet40",
        "octformer-base.lg",
        "sonata-lp.scannet20",
        "concerto-large-lp.scannet20",
        "utonia-lp.scannet20",
        "spunet-v1m1.scannet20",
    }
)


def _skip_if_model_not_runnable(model_name: str) -> None:
    """Skip everything that can't be exercised in `test_model_forward` on this box."""
    _skip_if_model_deps_missing(model_name)
    if model_name.startswith(("point-mamba", "spvcnn", "spunet")) and not torch.cuda.is_available():
        pytest.skip(f"{model_name} requires CUDA, none available")
    if model_name in _UNSUPPORTED_BY_DATA_FACTORY:
        pytest.skip("Synthetic data factory doesn't support this model's input format yet")
    if model_name.startswith("randlanet."):
        # /4..16x decimation drops below K=16 on the synthetic 512+768 cloud; real clouds
        # are ~10^5+ points so this is purely a test-data artefact.
        pytest.skip("Synthetic test cloud is too small for /4 decimation with K=16.")


def _check_forward_output(
    output: object,
    *,
    model_name: str,
    task: str,
    expected_shape: tuple[int, int],
) -> None:
    """Assert that `output` is a structurally valid logits tensor.

    We deliberately don't snapshot values: random-weight numerics are fragile across
    torch / CUDA / scatter-op versions and would flap without catching real model bugs.
    """
    label = f"{model_name} ({task})"
    assert isinstance(output, torch.Tensor), f"{label}: expected Tensor, got {type(output).__name__}"
    assert torch.is_floating_point(output), f"{label}: non-float dtype {output.dtype}"
    # KDE-based density estimation in `pointconv-density` is numerically unstable with
    # random weights at small `in_channels`. Pretrained weights converge to finite output.
    if not model_name.startswith("pointconv-density"):
        assert torch.isfinite(output).all().item(), f"{label}: output contains NaN or Inf"
    assert tuple(output.shape) == expected_shape, (
        f"{label}: output shape {tuple(output.shape)} != expected {expected_shape}"
    )


def _check_architecture_or_regen(
    model: nn.Module,
    model_name: str,
    task: str,
    models_dir: Path,
    force_regen: bool,
) -> None:
    # Skip uninitialized lazy parameters since their shape is only known after a forward pass
    parameters = [p for p in model.parameters() if not isinstance(p, nn.parameter.UninitializedParameter)]
    state_dict = {
        k: list(v.shape)
        for k, v in model.state_dict().items()
        if not isinstance(v, nn.parameter.UninitializedParameter)
    }
    metadata = {
        "num_params": sum(p.numel() for p in parameters),
        "state_dict": state_dict,
    }

    expected_path = models_dir / f"{model_name}_{task}.json"
    if force_regen or not expected_path.exists():
        expected_path.parent.mkdir(parents=True, exist_ok=True)
        expected_path.write_text(json.dumps(metadata, indent=2) + "\n")
        pytest.fail(f"Regenerated {expected_path.as_posix()!r}")

    expected = json.loads(expected_path.read_text())

    assert metadata["num_params"] == expected["num_params"], (
        f"Parameter count changed for {model_name!r} ({task}): "
        f"expected {expected['num_params']}, got {metadata['num_params']}"
    )
    assert metadata["state_dict"] == expected["state_dict"], f"State-dict structure changed for {model_name!r} ({task})"


@pytest.mark.parametrize(
    "task,expected_models",
    [
        ("base", BASE_MODELS),
        ("classification", CLASSIFICATION_MODELS),
        ("segmentation", SEGMENTATION_MODELS),
    ],
)
def test_list_models(task: str, expected_models: List[str]) -> None:
    """Test that the list of models is correct."""
    models = list_models(task=task)  # type: ignore[arg-type]

    if set(models) != set(expected_models):
        missing_models = set(models) - set(expected_models)
        extra_models = set(expected_models) - set(models)
        err_msg = f"Expected {len(expected_models)} {task} models, got {len(models)}. "
        if missing_models:
            err_msg += f"\nMissing models: {', '.join(sorted(missing_models))}"
        if extra_models:
            err_msg += f"\nExtra models: {', '.join(sorted(extra_models))}"

        raise AssertionError(err_msg)


@pytest.mark.skipif(
    not _TORCH_CLUSTER_AVAILABLE and not _TORCH_SCATTER_AVAILABLE,
    reason="torch-cluster or torch-scatter is not installed",
)
@pytest.mark.parametrize("model_name", CLASSIFICATION_MODELS)
def test_classification_architecture(model_name: str, force_regen: bool, models_dir_factory: Any) -> None:
    """Test that the architecture of all registered segmentation models is correct.
    This test will only verify that the state-dict structure of the model matches the expected structure,
    but will not verify that the content of the weights are correct.

    This test is useful to catch accidental architecture changes in the models (e.g. renaming a parameter or module),
    and is faster than a full forward pass + weight loading.

    To regenerate the expected architecture as JSON files, run

    ```bash
    uv run pytest tests/models/test_registered_models.py -k test_classification_architecture --force-regen
    ```
    """
    # Only copy the models directory to the temporary directory
    models_dir = models_dir_factory("*.json")

    _skip_if_model_deps_missing(model_name)
    model = create_model(model_name, task="classification", in_channels=3, num_classes=10)
    _check_architecture_or_regen(
        model,
        model_name,
        task="classification",
        models_dir=models_dir,
        force_regen=force_regen,
    )


@pytest.mark.skipif(
    not _TORCH_CLUSTER_AVAILABLE and not _TORCH_SCATTER_AVAILABLE,
    reason="torch-cluster or torch-scatter is not installed",
)
@pytest.mark.parametrize("model_name", SEGMENTATION_MODELS)
def test_segmentation_architecture(model_name: str, force_regen: bool, models_dir_factory: Any) -> None:
    """Test that the architecture of all registered segmentation models is correct.
    This test will only verify that the state-dict structure of the model matches the expected structure,
    but will not verify that the content of the weights are correct.

    This test is useful to catch accidental architecture changes in the models (e.g. renaming a parameter or module),
    and is faster than a full forward pass + weight loading.

    To regenerate the expected architecture as JSON files, run

    ```bash
    uv run --no-sync pytest tests/models/test_registered_models.py -k test_segmentation_architecture --force-regen
    ```
    """
    _skip_if_model_deps_missing(model_name)
    # Only copy the models directory to the temporary directory
    models_dir = models_dir_factory("*.json")

    model = create_model(model_name, task="segmentation", in_channels=3, num_classes=10)
    _check_architecture_or_regen(
        model,
        model_name,
        task="segmentation",
        models_dir=models_dir,
        force_regen=force_regen,
    )


@pytest.fixture
def data_factory() -> Callable[[int, int], Dict[str, Any]]:
    def create_data(
        in_channels: int,
        spatial_dim: int = 3,
        num_categories: int = 0,
    ) -> Dict[str, Any]:
        torch.manual_seed(42)
        lengths = torch.tensor([512, 768])
        pos = torch.randn(int(lengths.sum()), spatial_dim)
        x = torch.randn(int(lengths.sum()), in_channels) if in_channels > 0 else None
        batch = torch.repeat_interleave(torch.arange(len(lengths)), lengths)
        cls_onehot = torch.zeros(len(lengths), num_categories) if num_categories > 0 else None

        octree = None
        if _OCNN_AVAILABLE:
            octree = build_octree(
                pos=pos,
                batch=batch,
                features=x,
                batch_size=len(lengths),
            )
            octree.construct_all_neigh()
            x = octree.features[octree.depth]
            pos = octree.points[octree.depth]

        return dict(
            x=x,
            pos=pos,
            octree=octree,
            depth=octree.depth if octree is not None else None,
            batch=batch,
            cls_onehot=cls_onehot,
            category=cls_onehot,
        )

    return create_data


@pytest.mark.skipif(
    not _TORCH_CLUSTER_AVAILABLE and not _TORCH_SCATTER_AVAILABLE,
    reason="torch-cluster or torch-scatter is not installed",
)
@pytest.mark.parametrize(
    "model_name,task",
    [
        *[(model, "classification") for model in CLASSIFICATION_MODELS],
        *[(model, "segmentation") for model in SEGMENTATION_MODELS],
    ],
)
def test_model_forward(model_name: str, task: str, data_factory: Callable) -> None:
    _skip_if_model_not_runnable(model_name)

    num_classes = 10
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = create_model(model_name, task=task, in_channels=3, num_classes=num_classes)  # type: ignore[call-overload]
    data = data_factory(
        in_channels=model.in_channels,
        spatial_dim=getattr(model, "spatial_dim", 3),
        num_categories=getattr(model, "num_categories", 0),
    )
    expected_rows = int(data["batch"].max().item()) + 1 if task == "classification" else data["pos"].shape[0]

    # Keep only the kwargs the model's forward actually accepts, then move to device.
    sig = inspect.signature(model.forward)
    kwargs = {a: data[a] for a in sig.parameters if a != "self"}
    kwargs = {k: v.to(device) if hasattr(v, "to") else v for k, v in kwargs.items()}
    model = model.to(device)

    output = model(**kwargs)
    _check_forward_output(output, model_name=model_name, task=task, expected_shape=(expected_rows, num_classes))

    if device == "cuda":
        torch.cuda.empty_cache()
