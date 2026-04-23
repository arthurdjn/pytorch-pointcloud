from typing import List

import pytest

from torch_pointcloud.models import create_model, list_models
from torch_pointcloud.models._base import ClassificationModel, SegmentationModel
from torch_pointcloud.utils.imports import (
    _DWCONV_AVAILABLE,
    _MAMBA_SSM_AVAILABLE,
    _TORCH_CLUSTER_AVAILABLE,
    _TORCH_SCATTER_AVAILABLE,
)

CLASSIFICATION_MODELS = [
    "dgcnn-base",
    "kpfcnn.modelnet40",
    "octformer-base.modelnet40",
    "pointcnn-base",
    "pointconv-original",
    "point-mamba-base.modelnet40",
    "point-mamba-base.scanobjectnn",
    "point-mamba-base.scanobjectnn-nobg",
    "point-mamba-base.scanobjectnn-augmentedrot-scale75",
    "pointmlp-base",
    "pointmlp-elite",
    "pointnext-base",
    "pointnext-base",
    "pointnext-lg",
    "pointnext-lg",
    "pointnext-sm",
    "pointnext-sm",
    "pointnext-xl",
    "pointnext-xl",
]
SEGMENTATION_MODELS = [
    "dgcnn-base",
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
    "pointnext-lg",
    "pointnext-sm",
    "pointnext-xl",
]


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
    if model_name.startswith("point-mamba") and not _MAMBA_SSM_AVAILABLE:
        pytest.skip("mamba_ssm is not installed")
    if model_name.startswith("octformer") and not _DWCONV_AVAILABLE:
        pytest.skip("dwconv is not installed")

    model = create_model(model_name, task="classification", in_channels=3, num_classes=10)
    assert isinstance(model, ClassificationModel)


@pytest.mark.skipif(
    not _TORCH_CLUSTER_AVAILABLE and not _TORCH_SCATTER_AVAILABLE,
    reason="torch-cluster or torch-scatter is not installed",
)
@pytest.mark.parametrize("model_name", SEGMENTATION_MODELS)
def test_segmentation_model_forward(model_name: str) -> None:
    """Test that all registered models can be created and work."""
    if model_name.startswith("point-mamba") and not _MAMBA_SSM_AVAILABLE:
        pytest.skip("mamba_ssm is not installed")
    if model_name.startswith("octformer") and not _DWCONV_AVAILABLE:
        pytest.skip("dwconv is not installed")

    model = create_model(model_name, task="segmentation", in_channels=3, num_classes=10)
    assert isinstance(model, SegmentationModel)
