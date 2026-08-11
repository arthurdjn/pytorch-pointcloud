from typing import Dict

import pytest
import torch
from torch import Tensor

from torch_pointcloud.models import list_models
from torch_pointcloud.models.pointnext import (
    PointNeXtClassification,
    PointNeXtDecoder,
    PointNeXtEncoder,
    PointNeXtEncoderBlock,
    PointNeXtPartSegmentation,
    PointNeXtSegmentation,
)
from torch_pointcloud.utils.imports import _TORCH_CLUSTER_AVAILABLE, _TORCH_SCATTER_AVAILABLE

# See: https://docs.pytest.org/en/stable/how-to/skipping.html#summary
pytestmark = pytest.mark.skipif(
    not _TORCH_CLUSTER_AVAILABLE and not _TORCH_SCATTER_AVAILABLE,
    reason="torch-cluster or torch-scatter is not installed",
)


@pytest.fixture
def data() -> Dict[str, Tensor]:
    torch.manual_seed(42)
    lengths = torch.tensor([256, 512])
    pos = torch.randn(int(lengths.sum()), 3)
    features = torch.randn(int(lengths.sum()), 6)
    batch = torch.repeat_interleave(torch.arange(len(lengths)), lengths)

    return dict(
        features=features,
        pos=pos,
        batch=batch,
    )


@pytest.fixture
def model_clf() -> PointNeXtClassification:
    return PointNeXtClassification(
        in_channels=6,
        num_classes=10,
        stem_channels=32,
        encoder_channels=[32, 64, 128],
        encoder_depths=[2, 2, 2],
        ratios=[0.5, 0.5, 0.5],
        radiuses=[0.1, 0.2, 0.4, 0.8],
        num_neighbors=[16, 16, 16, 16],
    )


@pytest.fixture
def model_seg() -> PointNeXtSegmentation:
    return PointNeXtSegmentation(
        in_channels=6,
        num_classes=10,
        stem_channels=32,
        encoder_channels=[32, 64, 128],
        encoder_depths=[2, 2, 2],
        decoder_channels=[128, 64, 32],
        decoder_depths=[2, 2, 2],
        ratios=[0.5, 0.5, 0.5],
        radiuses=[0.1, 0.2, 0.4, 0.8],
        num_neighbors=[16, 16, 16, 16],
    )


def test_pointnext_encoder_block_basic(data: Dict[str, Tensor]) -> None:
    """Test basic PointNeXtEncoderBlock functionality."""
    block = PointNeXtEncoderBlock(
        spatial_dim=3,
        channels=6,
        depth=2,
        expansion=4,
        radius=0.1,
        num_neighbors=16,
    )

    out_features, out_pos, out_batch = block(data["features"], data["pos"], data["batch"])

    assert out_features.shape[0] == out_pos.shape[0] == out_batch.shape[0]
    assert out_features.shape[1] == 6
    assert out_pos.shape[1] == 3
    assert out_batch.shape[0] <= data["batch"].shape[0]  # May be downsampled


def test_pointnext_encoder_basic(data: Dict[str, Tensor]) -> None:
    """Test basic PointNeXtEncoder functionality."""
    encoder = PointNeXtEncoder(
        channels=[6, 32, 64, 128],
        depths=[2, 2, 2],
        ratios=[0.5, 0.5, 0.5],
        radiuses=[0.1, 0.2, 0.4, 0.8],
        num_neighbors=[16, 16, 16, 16],
    )

    out_features, out_pos, out_batch = encoder(data["features"], data["pos"], data["batch"])

    assert out_features.shape[0] == out_pos.shape[0] == out_batch.shape[0]
    assert out_features.shape[1] == 128  # Last channel
    assert out_pos.shape[1] == 3


def test_pointnext_encoder_with_intermediates(data: Dict[str, Tensor]) -> None:
    """Test PointNeXtEncoder with intermediate outputs."""
    encoder = PointNeXtEncoder(
        channels=[6, 32, 64, 128],
        depths=[2, 2, 2],
        ratios=[0.5, 0.5, 0.5],
        radiuses=[0.1, 0.2, 0.4, 0.8],
        num_neighbors=[16, 16, 16, 16],
    )

    _, _, _, intermediates = encoder(
        data["features"],
        data["pos"],
        data["batch"],
        return_intermediates=True,
    )

    assert len(intermediates) == 3  # Number of blocks
    for intermediate in intermediates:
        assert hasattr(intermediate, "x")
        assert hasattr(intermediate, "pos")
        assert hasattr(intermediate, "batch")
        assert intermediate.x.shape[0] == intermediate.pos.shape[0] == intermediate.batch.shape[0]


def test_pointnext_encoder_decoder_basic(data: Dict[str, Tensor]) -> None:
    """Test basic PointNeXtDecoder functionality."""
    encoder = PointNeXtEncoder(
        channels=[6, 32, 64, 128],
        depths=[2, 2, 2],
        ratios=[0.5, 0.5, 0.5],
        radiuses=[0.1, 0.2, 0.4, 0.8],
        num_neighbors=[16, 16, 16, 16],
    )

    decoder = PointNeXtDecoder(
        channels=[128, 128, 64, 32],
        skip_channels=[64, 32, 6],
        depths=[2, 2, 2],
    )

    x, pos, batch, intermediates = encoder(data["features"], data["pos"], data["batch"], return_intermediates=True)
    x, pos, batch = decoder(x, pos, batch, intermediates)

    assert x.shape[0] == pos.shape[0] == batch.shape[0]
    assert x.shape[1] == 32  # Final channel
    assert pos.shape[1] == 3


def test_pointnext_classification_forward(model_clf: PointNeXtClassification, data: Dict[str, Tensor]) -> None:
    """Test PointNeXtClassification forward pass."""
    logits = model_clf(data["features"], data["pos"], data["batch"])
    assert logits.shape == (data["batch"].max() + 1, model_clf.num_classes)
    assert logits.dtype == data["features"].dtype


def test_pointnext_segmentation_forward(model_seg: PointNeXtSegmentation, data: Dict[str, Tensor]) -> None:
    """Test PointNeXtSegmentation forward pass."""
    logits = model_seg(data["features"], data["pos"], data["batch"])
    assert logits.shape == (data["pos"].shape[0], model_seg.num_classes)
    assert logits.dtype == data["features"].dtype


@pytest.fixture
def model_partseg() -> PointNeXtPartSegmentation:
    return PointNeXtPartSegmentation(
        in_channels=6,
        num_classes=10,
        num_categories=4,
        stem_channels=32,
        stem_plain_last=False,
        encoder_channels=[32, 64, 128],
        encoder_depths=[2, 2, 2],
        encoder_expansion=4,
        sa_layers=1,
        sa_use_res=True,
        decoder_channels=[128, 64, 32],
        decoder_depths=[2, 2, 2],
        decoder_plain_last=True,
        ratios=[0.5, 0.5, 0.5],
        radiuses=[0.1, 0.2, 0.4, 0.8],
        num_neighbors=[16, 16, 16, 16],
        add_self_loops=False,
        spatial_dim=3,
        act="relu",
        act_kwargs=None,
        act_first=False,
        norm="batch_norm",
        norm_kwargs=None,
        bias=True,
        dropout=0.0,
        head_channels=None,
    )


@pytest.fixture
def partseg_category(data: Dict[str, Tensor]) -> Tensor:
    num_batches = int(data["batch"].max()) + 1
    return torch.nn.functional.one_hot(torch.arange(num_batches) % 4, num_classes=4).float()


def test_pointnext_part_segmentation_forward(
    model_partseg: PointNeXtPartSegmentation, data: Dict[str, Tensor], partseg_category: Tensor
) -> None:
    logits = model_partseg(data["features"], data["pos"], data["batch"], category=partseg_category)
    assert logits.shape == (data["pos"].shape[0], model_partseg.num_classes)
    assert logits.dtype == data["features"].dtype


def test_pointnext_part_segmentation_reset_classifier(
    model_partseg: PointNeXtPartSegmentation, data: Dict[str, Tensor], partseg_category: Tensor
) -> None:
    model_partseg.reset_classifier(num_classes=42)
    logits = model_partseg(data["features"], data["pos"], data["batch"], partseg_category)
    assert logits.shape == (data["pos"].shape[0], 42)


def test_pointnext_segmentation_single_dropout_with_mlp_head() -> None:
    model = PointNeXtSegmentation(
        in_channels=6,
        num_classes=10,
        stem_channels=32,
        encoder_channels=[32, 64, 128],
        encoder_depths=[2, 2, 2],
        decoder_channels=[128, 64, 32],
        decoder_depths=[2, 2, 2],
        ratios=[0.5, 0.5, 0.5],
        radiuses=[0.1, 0.2, 0.4, 0.8],
        num_neighbors=[16, 16, 16, 16],
        head_channels=[16],
        dropout=0.9,
    )
    model.train()
    x = torch.randn(64, model.embedding_dim)
    assert torch.equal(model.forward_head(x, pre_logits=True), x)

    linear_head = PointNeXtSegmentation(
        in_channels=6,
        num_classes=10,
        stem_channels=32,
        encoder_channels=[32, 64, 128],
        encoder_depths=[2, 2, 2],
        decoder_channels=[128, 64, 32],
        decoder_depths=[2, 2, 2],
        ratios=[0.5, 0.5, 0.5],
        radiuses=[0.1, 0.2, 0.4, 0.8],
        num_neighbors=[16, 16, 16, 16],
        dropout=0.9,
    )
    linear_head.train()
    torch.manual_seed(0)
    assert not torch.equal(linear_head.forward_head(x, pre_logits=True), x)


def test_pointnext_part_segmentation_single_dropout_with_mlp_head(
    data: Dict[str, Tensor], partseg_category: Tensor, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = PointNeXtPartSegmentation(
        in_channels=6,
        num_classes=10,
        num_categories=4,
        stem_channels=32,
        encoder_channels=[32, 64, 128],
        encoder_depths=[2, 2, 2],
        sa_layers=1,
        decoder_channels=[128, 64, 32],
        decoder_depths=[2, 2, 2],
        ratios=[0.5, 0.5, 0.5],
        radiuses=[0.1, 0.2, 0.4, 0.8],
        num_neighbors=[16, 16, 16, 16],
        head_channels=[16],
        dropout=0.5,
    )
    model.train()

    original = torch.nn.functional.dropout
    widths: list[int] = []

    def recording_dropout(x: Tensor, p: float = 0.5, training: bool = True, inplace: bool = False) -> Tensor:
        if p > 0:
            widths.append(x.shape[-1])
        return original(x, p=p, training=training, inplace=inplace)

    monkeypatch.setattr(torch.nn.functional, "dropout", recording_dropout)
    model(data["features"], data["pos"], data["batch"], partseg_category)
    assert widths
    assert all(width != model.embedding_dim for width in widths)


def test_pointnext_classification_num_classes_zero_returns_features(data: Dict[str, Tensor]) -> None:
    model = PointNeXtClassification(
        in_channels=6,
        num_classes=0,
        stem_channels=32,
        encoder_channels=[32, 64, 128],
        encoder_depths=[2, 2, 2],
        ratios=[0.5, 0.5, 0.5],
        radiuses=[0.1, 0.2, 0.4, 0.8],
        num_neighbors=[16, 16, 16, 16],
        head_channels=[16],
    )
    assert isinstance(model.head, torch.nn.Identity)
    out = model(data["features"], data["pos"], data["batch"])
    assert out.shape == (int(data["batch"].max()) + 1, model.embedding_dim)


def test_pointnext_segmentation_reset_classifier_keeps_head_channels() -> None:
    model = PointNeXtSegmentation(
        in_channels=6,
        num_classes=10,
        stem_channels=32,
        encoder_channels=[32, 64, 128],
        encoder_depths=[2, 2, 2],
        decoder_channels=[128, 64, 32],
        decoder_depths=[2, 2, 2],
        ratios=[0.5, 0.5, 0.5],
        radiuses=[0.1, 0.2, 0.4, 0.8],
        num_neighbors=[16, 16, 16, 16],
        head_channels=[16],
    )
    model.reset_classifier(num_classes=7)
    assert model.head.channel_list == [model.embedding_dim, 16, 7]


def test_pointnext_xl_s3dis_area6_registered_without_weights() -> None:
    pretrained = list_models("pointnext-xl.s3dis*", task="segmentation", pretrained=True)
    assert "pointnext-xl.s3dis-area6.openpoints" not in pretrained
    assert "pointnext-xl.s3dis-area5.openpoints" in pretrained
    assert "pointnext-xl.s3dis-area6.openpoints" in list_models("pointnext-xl*", task="segmentation")
