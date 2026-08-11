from typing import Dict

import pytest
import torch
from torch import Tensor
from torch_geometric.nn import MLP

from torch_pointcloud.models import create_model
from torch_pointcloud.models.pvcnn2 import PVCNN2Classification, PVCNN2Segmentation
from torch_pointcloud.utils.imports import _TORCH_CLUSTER_AVAILABLE

# See: https://docs.pytest.org/en/stable/how-to/skipping.html#summary
pytestmark = pytest.mark.skipif(
    not _TORCH_CLUSTER_AVAILABLE,
    reason="torch-cluster is not installed",
)


@pytest.fixture
def data() -> Dict[str, Tensor]:
    torch.manual_seed(42)
    lengths = torch.tensor([256, 512])
    pos = torch.randn(int(lengths.sum()), 3)
    x = torch.randn(int(lengths.sum()), 6)
    batch = torch.repeat_interleave(torch.arange(len(lengths)), lengths)
    return dict(x=x, pos=pos, batch=batch)


@pytest.fixture
def model_clf() -> PVCNN2Classification:
    return PVCNN2Classification(
        in_channels=6,
        num_classes=10,
        ratios=[0.5, 0.5],
        radii=[0.2, 0.4],
        num_neighbors=[16, 16],
        sa_channels=[[32, 64], [64, 64]],
        encoder_channels=[6, 32, 64],
        encoder_depths=[1, 1],
        encoder_resolutions=[8, 8],
        encoder_kernel_sizes=[3, 3],
        use_se=False,
        normalize=True,
        act="relu",
        act_kwargs=None,
        act_first=False,
        norm="batch_norm",
        norm_kwargs=None,
        dropout=0.0,
        global_pool="max",
    )


@pytest.fixture
def model_seg() -> PVCNN2Segmentation:
    return PVCNN2Segmentation(
        in_channels=6,
        num_classes=10,
        ratios=[0.5, 0.5],
        radii=[0.2, 0.4],
        num_neighbors=[16, 16],
        sa_channels=[[32, 64], [64, 64]],
        encoder_channels=[6, 32, 64],
        encoder_depths=[1, 1],
        encoder_resolutions=[8, 8],
        encoder_kernel_sizes=[3, 3],
        fp_channels=[[64, 32], [32, 32]],
        decoder_channels=[32, 32],
        decoder_depths=[1, 1],
        decoder_resolutions=[8, 8],
        decoder_kernel_sizes=[3, 3],
        use_se=False,
        normalize=True,
        act="relu",
        act_kwargs=None,
        act_first=False,
        norm="batch_norm",
        norm_kwargs=None,
        dropout=0.0,
    )


def test_pvcnn2_classification_forward(model_clf: PVCNN2Classification, data: Dict[str, Tensor]) -> None:
    logits = model_clf(data["x"], data["pos"], data["batch"])
    assert logits.shape == (int(data["batch"].max()) + 1, model_clf.num_classes)
    assert logits.dtype == data["x"].dtype


def test_pvcnn2_classification_reset_classifier(model_clf: PVCNN2Classification, data: Dict[str, Tensor]) -> None:
    model_clf.reset_classifier(num_classes=42)
    logits = model_clf(data["x"], data["pos"], data["batch"])
    assert logits.shape == (int(data["batch"].max()) + 1, 42)


def test_pvcnn2_classification_reset_classifier_zero(model_clf: PVCNN2Classification) -> None:
    model_clf.reset_classifier(num_classes=0)
    assert isinstance(model_clf.head, torch.nn.Identity)


def test_pvcnn2_segmentation_forward(model_seg: PVCNN2Segmentation, data: Dict[str, Tensor]) -> None:
    logits = model_seg(data["x"], data["pos"], data["batch"])
    assert logits.shape == (data["pos"].shape[0], model_seg.num_classes)
    assert logits.dtype == data["x"].dtype


def test_pvcnn2_classification_forward_features_and_head(
    model_clf: PVCNN2Classification, data: Dict[str, Tensor]
) -> None:
    x, _, batch = model_clf.forward_features(data["x"], data["pos"], data["batch"])
    assert x.shape[0] == batch.shape[0]
    logits = model_clf.forward_head(x, batch)
    assert logits.shape == (int(data["batch"].max()) + 1, model_clf.num_classes)


def test_pvcnn2_segmentation_forward_features_decoder_head(
    model_seg: PVCNN2Segmentation, data: Dict[str, Tensor]
) -> None:
    x, pos, batch, intermediates = model_seg.forward_features(
        data["x"], data["pos"], data["batch"], return_intermediates=True
    )
    assert len(intermediates) > 0
    x, _, _ = model_seg.forward_decoder(x, pos, batch, intermediates)
    assert x.shape[0] == data["pos"].shape[0]
    logits = model_seg.forward_head(x)
    assert logits.shape == (data["pos"].shape[0], model_seg.num_classes)


def test_pvcnn2_segmentation_reset_classifier(model_seg: PVCNN2Segmentation, data: Dict[str, Tensor]) -> None:
    model_seg.reset_classifier(num_classes=42)
    logits = model_seg(data["x"], data["pos"], data["batch"])
    assert logits.shape == (data["pos"].shape[0], 42)


def test_pvcnn2_segmentation_reset_classifier_zero(model_seg: PVCNN2Segmentation) -> None:
    model_seg.reset_classifier(num_classes=0)
    assert isinstance(model_seg.head, torch.nn.Identity)


def test_pvcnn2_segmentation_asymmetric_decoder(data: Dict[str, Tensor]) -> None:
    model = PVCNN2Segmentation(
        in_channels=6,
        num_classes=10,
        ratios=[0.5, 0.5, 0.5],
        radii=[0.2, 0.4, 0.4],
        num_neighbors=[8, 8, 8],
        sa_channels=[[16, 32], [32, 64], [32, 64]],
        encoder_channels=[6, 16, 32, 64],
        encoder_depths=[1, 1, 0],
        encoder_resolutions=[4, 4, 0],
        encoder_kernel_sizes=[3, 3, 0],
        fp_channels=[[32, 32], [32, 16]],
        decoder_channels=[32, 16],
        decoder_depths=[1, 1],
        decoder_resolutions=[4, 4],
        decoder_kernel_sizes=[3, 3],
    )
    assert model.decoder.skip_channels == (32, 3)
    logits = model(data["x"], data["pos"], data["batch"])
    assert logits.shape == (data["pos"].shape[0], 10)


def test_pvcnn2_segmentation_too_many_decoder_blocks() -> None:
    with pytest.raises(ValueError, match="decoder blocks"):
        PVCNN2Segmentation(
            in_channels=6,
            num_classes=10,
            ratios=[0.5, 0.5],
            radii=[0.2, 0.4],
            num_neighbors=[8, 8],
            sa_channels=[[16, 32], [32, 64]],
            encoder_channels=[6, 16, 32],
            encoder_depths=[1, 1],
            encoder_resolutions=[4, 4],
            encoder_kernel_sizes=[3, 3],
            fp_channels=[[32, 32], [32, 16], [16, 16]],
            decoder_channels=[32, 16, 16],
            decoder_depths=[1, 1, 1],
            decoder_resolutions=[4, 4, 4],
            decoder_kernel_sizes=[3, 3, 3],
        )


def test_pvcnn2_segmentation_reference_hparams() -> None:
    torch.manual_seed(42)
    model = create_model("pvcnn2.s3dis-area5", task="segmentation")
    assert isinstance(model, PVCNN2Segmentation)
    assert isinstance(model.head, MLP)
    assert model.decoder.skip_channels == (256, 128, 64, 6)

    lengths = torch.tensor([512, 768])
    num_points = int(lengths.sum())
    x = torch.randn(num_points, 9)
    pos = torch.randn(num_points, 3)
    batch = torch.repeat_interleave(torch.arange(len(lengths)), lengths)

    model.eval()
    with torch.no_grad():
        logits = model(x, pos, batch)
    assert logits.shape == (num_points, 13)
