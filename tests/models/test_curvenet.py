from typing import Dict

import pytest
import torch
from torch import Tensor

from torch_pointcloud.models import model_info
from torch_pointcloud.models.curvenet import CurveNetClassification, CurveNetPartSegmentation
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
    pos = torch.rand(int(lengths.sum()), 3)
    batch = torch.repeat_interleave(torch.arange(len(lengths)), lengths)
    category = torch.nn.functional.one_hot(torch.tensor([0, 2]), num_classes=4).float()
    return dict(pos=pos, batch=batch, category=category)


@pytest.fixture
def model_clf() -> CurveNetClassification:
    return CurveNetClassification(
        in_channels=0,
        num_classes=10,
        embed_channels=16,
        channels=[16, 32, 64],
        blocks=[1, 2, 1],
        num_points=[None, 32, 16],
        radius=[None, 0.3, 0.5],
        num_neighbors=8,
        curves=[[10, 3], [5, 3], None],
        num_layers=1,
        proj_channels=64,
        head_channels=32,
        global_pool=["max", "mean"],
    ).eval()


@pytest.fixture
def model_partseg() -> CurveNetPartSegmentation:
    return CurveNetPartSegmentation(
        in_channels=0,
        num_classes=12,
        num_categories=4,
        embed_channels=16,
        channels=[16, 32, 64],
        blocks=[1, 1, 2],
        num_points=[None, 32, 8],
        radius=[None, 0.3, 0.6],
        num_neighbors=[8, 8, 4],
        curves=[[10, 3], None, None],
        num_layers=2,
        decoder_channels=[32, 16],
        gate_channels=[16, 8],
        global_channels=[8, 16],
        fuse_channels=[32, 32],
        head_channels=32,
        se_ratio=4,
    ).eval()


def test_curvenet_classification_forward(model_clf: CurveNetClassification, data: Dict[str, Tensor]) -> None:
    logits = model_clf(None, data["pos"], data["batch"])
    assert logits.shape == (2, 10)
    assert model_clf.num_features == 128


def test_curvenet_classification_with_features(data: Dict[str, Tensor]) -> None:
    model = CurveNetClassification(
        in_channels=3,
        num_classes=4,
        embed_channels=8,
        channels=[16],
        blocks=1,
        num_points=[None],
        radius=[None],
        num_neighbors=6,
        curves=[[4, 2]],
        proj_channels=16,
        head_channels=8,
    ).eval()
    assert model(data["pos"], data["pos"], data["batch"]).shape == (2, 4)


def test_curvenet_classification_reset_classifier(model_clf: CurveNetClassification, data: Dict[str, Tensor]) -> None:
    model_clf.reset_classifier(num_classes=0)
    assert isinstance(model_clf.head, torch.nn.Identity)
    assert model_clf(None, data["pos"], data["batch"]).shape == (2, model_clf.num_features)
    model_clf.reset_classifier(num_classes=5, global_pool="max")
    assert model_clf.num_features == 64
    assert model_clf(None, data["pos"], data["batch"]).shape == (2, 5)


def test_curvenet_classification_forward_features_and_head(
    model_clf: CurveNetClassification,
    data: Dict[str, Tensor],
) -> None:
    x, pos, batch = model_clf.forward_features(None, data["pos"], data["batch"])
    assert x.shape == (32, 64) and pos.shape == (32, 3) and batch.shape == (32,)
    assert model_clf.forward_head(x, batch, pre_logits=True).shape == (2, model_clf.num_features)


def test_curvenet_part_segmentation_forward(
    model_partseg: CurveNetPartSegmentation,
    data: Dict[str, Tensor],
) -> None:
    logits = model_partseg(None, data["pos"], data["batch"], data["category"])
    assert logits.shape == (data["pos"].shape[0], 12)


def test_curvenet_part_segmentation_intermediates(
    model_partseg: CurveNetPartSegmentation,
    data: Dict[str, Tensor],
) -> None:
    x, pos, batch, intermediates = model_partseg.forward_features(
        None, data["pos"], data["batch"], return_intermediates=True
    )
    assert x.shape == (16, 64)
    assert [stage["x"].shape for stage in intermediates] == [(224, 16), (64, 32)]
    x, pos, batch = model_partseg.forward_decoder(x, pos, batch, data["category"], intermediates)
    assert x.shape == (224, 32) and torch.equal(batch, data["batch"])


def test_curvenet_part_segmentation_reset_classifier(
    model_partseg: CurveNetPartSegmentation,
    data: Dict[str, Tensor],
) -> None:
    model_partseg.reset_classifier(num_classes=0)
    feats = model_partseg(None, data["pos"], data["batch"], data["category"])
    assert feats.shape == (data["pos"].shape[0], model_partseg.num_features)
    model_partseg.reset_classifier(num_classes=7)
    assert model_partseg(None, data["pos"], data["batch"], data["category"]).shape == (data["pos"].shape[0], 7)


def test_curvenet_part_segmentation_rejects_mismatched_decoder() -> None:
    with pytest.raises(ValueError, match="decoder_channels"):
        CurveNetPartSegmentation(
            in_channels=0,
            num_classes=4,
            num_categories=2,
            embed_channels=8,
            channels=[8, 16],
            blocks=1,
            num_points=[None, 8],
            radius=[None, 0.5],
            num_neighbors=4,
            curves=[None, None],
            decoder_channels=[8, 8],
            gate_channels=[4, 4],
            global_channels=[4, 8],
            fuse_channels=[8],
            head_channels=8,
            se_ratio=2,
        )


@pytest.mark.parametrize(
    "name",
    [
        pytest.param("curvenet.modelnet40.tiange-xiang", id="curvenet.modelnet40.tiange-xiang"),
        pytest.param("curvenet.shapenetpart.tiange-xiang", id="curvenet.shapenetpart.tiange-xiang"),
    ],
)
def test_registered_transform_runs_without_labels(name: str) -> None:
    info = model_info(name)
    sample = {DataKeys.POS: torch.rand(3000, 3)}
    if info["task"] == "part-segmentation":
        sample[DataKeys.CATEGORY] = torch.tensor(3)
    transform = info["transform"]
    assert transform is not None
    data = transform(sample)
    assert DataKeys.SEGMENT not in data and DataKeys.NORMAL not in data
    assert data[DataKeys.POS].shape[1] == 3
