import pytest
import torch

from torch_pointcloud.models import create_model
from torch_pointcloud.utils.imports import _TORCH_CLUSTER_AVAILABLE, _TORCH_SCATTER_AVAILABLE


@pytest.fixture
def data() -> dict:
    torch.manual_seed(42)
    lengths = torch.tensor([512, 768])
    pos = torch.randn(int(lengths.sum()), 3)
    features = torch.randn(int(lengths.sum()), 3)
    batch = torch.repeat_interleave(torch.arange(len(lengths)), lengths)

    return dict(features=features, pos=pos, batch=batch)


@pytest.mark.skipif(
    not _TORCH_CLUSTER_AVAILABLE and not _TORCH_SCATTER_AVAILABLE,
    reason="torch-cluster or torch-scatter is not installed",
)
@pytest.mark.parametrize(
    "model_name",
    [
        "dgcnn-base",
        "pointnext-sm",
        "pointnext-base",
        "pointnext-lg",
        "pointnext-xl",
        "pointnext-sm",
        "pointnext-base",
        "pointnext-lg",
        "pointnext-xl",
        "pointcnn-base",
    ],
)
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
@pytest.mark.parametrize(
    "model_name",
    [
        "dgcnn-base",
        "pointnext-sm",
        "pointnext-base",
        "pointnext-lg",
        "pointnext-xl",
        "pointnext-sm",
        "pointnext-base",
        "pointnext-lg",
        "pointnext-xl",
        "pointcnn-base",
    ],
)
def test_segmentation_model_forward(model_name: str, data: dict) -> None:
    """Test that all registered models can be created and work."""
    model = create_model(model_name, task="segmentation", in_channels=3, num_classes=10)
    logits = model(data["features"], data["pos"], data["batch"])
    assert logits.shape == (data["pos"].shape[0], 10)

    # Check that output is finite
    assert torch.isfinite(logits).all()
