from typing import Dict

import pytest
import torch
from torch import Tensor

from torch_pointcloud.models import model_info
from torch_pointcloud.models.pct import PCTClassification
from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.imports import _PYG_LIB_AVAILABLE

# See: https://docs.pytest.org/en/stable/how-to/skipping.html#summary
pytestmark = pytest.mark.skipif(
    not _PYG_LIB_AVAILABLE,
    reason="pyg-lib is not installed",
)


@pytest.fixture
def data() -> Dict[str, Tensor]:
    torch.manual_seed(42)
    lengths = torch.tensor([128, 96])
    pos = torch.randn(int(lengths.sum()), 3)
    x = torch.randn(int(lengths.sum()), 3)
    batch = torch.repeat_interleave(torch.arange(len(lengths)), lengths)
    return dict(x=x, pos=pos, batch=batch)


@pytest.fixture
def model() -> PCTClassification:
    return PCTClassification(
        in_channels=3,
        num_classes=10,
        spatial_dim=3,
        stem_channels=[16, 16],
        channels=[32, 64],
        num_points=[64, 32],
        num_neighbors=8,
        attention_channels=32,
        num_attention_blocks=2,
        proj_channels=64,
        head_channels=[32],
        dropout=0.0,
    ).eval()


def test_pct_classification_forward(model: PCTClassification, data: Dict[str, Tensor]) -> None:
    logits = model(data["x"], data["pos"], data["batch"])
    assert logits.shape == (2, 10)


def test_pct_classification_without_features(data: Dict[str, Tensor]) -> None:
    model = PCTClassification(
        in_channels=0,
        num_classes=4,
        stem_channels=[8, 8],
        channels=[16, 32],
        num_points=[32, 16],
        num_neighbors=4,
        attention_channels=16,
        num_attention_blocks=1,
        proj_channels=32,
    ).eval()
    assert model(None, data["pos"], data["batch"]).shape == (2, 4)


def test_pct_classification_reset_classifier(model: PCTClassification, data: Dict[str, Tensor]) -> None:
    model.reset_classifier(num_classes=42, global_pool=("max", "mean"))
    assert model.num_features == 128
    assert model(data["x"], data["pos"], data["batch"]).shape == (2, 42)


def test_pct_classification_num_classes_zero_returns_features(
    model: PCTClassification,
    data: Dict[str, Tensor],
) -> None:
    model.reset_classifier(num_classes=0)
    assert isinstance(model.head, torch.nn.Identity)
    assert model(data["x"], data["pos"], data["batch"]).shape == (2, model.num_features)


def test_pct_classification_forward_features_and_head(model: PCTClassification, data: Dict[str, Tensor]) -> None:
    x, pos, batch = model.forward_features(data["x"], data["pos"], data["batch"])
    assert x.shape == (64, 64) and pos.shape == (64, 3) and batch.shape == (64,)
    feats = model.forward_head(x, batch, pre_logits=True)
    assert feats.shape == (2, model.num_features)


def test_pct_classification_rejects_mismatched_stages() -> None:
    with pytest.raises(ValueError, match="same length"):
        PCTClassification(
            in_channels=0,
            num_classes=4,
            stem_channels=[8],
            channels=[16, 32],
            num_points=[32],
            num_neighbors=4,
            attention_channels=16,
            num_attention_blocks=1,
            proj_channels=32,
        )


@pytest.mark.parametrize(
    "name",
    [
        pytest.param("pct.modelnet40.strawberry-eat-mango", id="pct.modelnet40.strawberry-eat-mango"),
    ],
)
def test_registered_transform_runs_without_labels(name: str) -> None:
    info = model_info(name)
    sample = {DataKeys.POS: torch.rand(3000, 3)}
    if info["task"] == "part-segmentation":
        sample[DataKeys.CATEGORY] = torch.tensor(3)
    data = info["transform"](sample)
    assert DataKeys.SEGMENT not in data and DataKeys.NORMAL not in data
    assert data[DataKeys.POS].shape[1] == 3
