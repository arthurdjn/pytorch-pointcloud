import json
from pathlib import Path
from typing import Any, List

import pytest
import torch.nn as nn

from torch_pointcloud.models import create_model, list_models
from torch_pointcloud.models._base import ClassificationModel, SegmentationModel
from torch_pointcloud.utils.imports import (
    _DWCONV_AVAILABLE,
    _FLASH_ATTN_AVAILABLE,
    _MAMBA_SSM_AVAILABLE,
    _TORCH_CLUSTER_AVAILABLE,
    _TORCH_SCATTER_AVAILABLE,
)

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
    "pointmlp-elite",
    "pointnext-base",
    "pointnext-lg",
    "pointnext-sm",
    "pointnext-sm-c64.modelnet40",
    "pointnext-sm.scanobjectnn",
    "pointnext-xl",
]
SEGMENTATION_MODELS = [
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
    "sonata-lp.scannet20",
]


def _skip_if_model_optional_deps_missing(model_name: str) -> None:
    if model_name.startswith("point-mamba") and not _MAMBA_SSM_AVAILABLE:
        pytest.skip("mamba_ssm is not installed")
    if model_name.startswith("octformer") and not _DWCONV_AVAILABLE:
        pytest.skip("dwconv is not installed")
    if model_name.startswith("sonata") and not _FLASH_ATTN_AVAILABLE:
        pytest.skip("flash_attn is not installed")


def _check_architecture_or_regen(
    model: nn.Module,
    model_name: str,
    task: str,
    models_dir: Path,
    force_regen: bool,
) -> None:
    metadata = {
        "num_params": sum(p.numel() for p in model.parameters()),
        "state_dict": {k: list(v.shape) for k, v in model.state_dict().items()},
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
    [("classification", CLASSIFICATION_MODELS), ("segmentation", SEGMENTATION_MODELS)],
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
def test_classification_model_forward(model_name: str) -> None:
    """Test that all registered models can be created and work."""
    _skip_if_model_optional_deps_missing(model_name)

    model = create_model(model_name, task="classification", in_channels=3, num_classes=10)
    assert isinstance(model, ClassificationModel)


@pytest.mark.skipif(
    not _TORCH_CLUSTER_AVAILABLE and not _TORCH_SCATTER_AVAILABLE,
    reason="torch-cluster or torch-scatter is not installed",
)
@pytest.mark.parametrize("model_name", SEGMENTATION_MODELS)
def test_segmentation_model_forward(model_name: str) -> None:
    """Test that all registered models can be created and work."""
    _skip_if_model_optional_deps_missing(model_name)

    model = create_model(model_name, task="segmentation", in_channels=3, num_classes=10)
    assert isinstance(model, SegmentationModel)


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

    _skip_if_model_optional_deps_missing(model_name)
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
    # Only copy the models directory to the temporary directory
    models_dir = models_dir_factory("*.json")

    _skip_if_model_optional_deps_missing(model_name)
    model = create_model(model_name, task="segmentation", in_channels=3, num_classes=10)
    _check_architecture_or_regen(
        model,
        model_name,
        task="segmentation",
        models_dir=models_dir,
        force_regen=force_regen,
    )
