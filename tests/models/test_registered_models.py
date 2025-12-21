from typing import List

import pytest
import torch

from torch_pointcloud.models import create_model, list_models
from torch_pointcloud.utils.imports import _TORCH_CLUSTER_AVAILABLE, _TORCH_SCATTER_AVAILABLE

CLASSIFICATION_MODELS = [
    "dgcnn-base",
    "kpconv-original.modelnet40",
    "kpconv-sm.modelnet40",
    "pointcnn-base",
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
    "pointcnn-base",
    "pointmlp-base",
    "pointnext-base",
    "pointnext-base",
    "pointnext-lg",
    "pointnext-lg",
    "pointnext-sm",
    "pointnext-sm",
    "pointnext-xl",
    "pointnext-xl",
]


@pytest.fixture
def data() -> dict:
    torch.manual_seed(42)
    lengths = torch.tensor([512, 768])
    pos = torch.randn(int(lengths.sum()), 3)
    features = torch.randn(int(lengths.sum()), 3)
    batch = torch.repeat_interleave(torch.arange(len(lengths)), lengths)

    return dict(features=features, pos=pos, batch=batch)


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
def test_classification_model_forward(model_name: str, data: dict) -> None:
    """Test that all registered models can be created and work."""
    model = create_model(model_name, task="classification", in_channels=3, num_classes=10)
    logits = model(data["features"], data["pos"], data["batch"])
    assert logits.shape == (data["batch"].max() + 1, 10)

    assert torch.isfinite(logits).all()


@pytest.mark.skipif(
    not _TORCH_CLUSTER_AVAILABLE and not _TORCH_SCATTER_AVAILABLE,
    reason="torch-cluster or torch-scatter is not installed",
)
@pytest.mark.parametrize("model_name", SEGMENTATION_MODELS)
def test_segmentation_model_forward(model_name: str, data: dict) -> None:
    """Test that all registered models can be created and work."""
    model = create_model(model_name, task="segmentation", in_channels=3, num_classes=10)
    logits = model(data["features"], data["pos"], data["batch"])
    assert logits.shape == (data["pos"].shape[0], 10)

    # Check that output is finite
    assert torch.isfinite(logits).all()
