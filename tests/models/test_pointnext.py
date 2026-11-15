from typing import Dict

import pytest
import torch
from torch import Tensor

from torch_pointcloud.datasets.s3dis import S3DIS_CLASSES, S3DIS_HDF5_CLASSES
from torch_pointcloud.models import create_model, model_info
from torch_pointcloud.models.pointnext import (
    PointNeXtClassification,
    PointNeXtDecoder,
    PointNeXtEncoder,
    PointNeXtEncoderBlock,
    PointNeXtPartSegmentation,
    PointNeXtSegmentation,
)
from torch_pointcloud.ops.cluster import radius
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
    lengths = torch.tensor([256, 512])
    pos = torch.rand(int(lengths.sum()), 3) * 0.5
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
        assert intermediate["x"].shape[0] == intermediate["pos"].shape[0] == intermediate["batch"].shape[0]


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
    model_partseg: PointNeXtPartSegmentation,
    data: Dict[str, Tensor],
    partseg_category: Tensor,
) -> None:
    logits = model_partseg(data["features"], data["pos"], data["batch"], category=partseg_category)
    assert logits.shape == (data["pos"].shape[0], model_partseg.num_classes)
    assert logits.dtype == data["features"].dtype


def test_pointnext_part_segmentation_reset_classifier(
    model_partseg: PointNeXtPartSegmentation,
    data: Dict[str, Tensor],
    partseg_category: Tensor,
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
    x = torch.randn(64, model.num_features)
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
    data: Dict[str, Tensor],
    partseg_category: Tensor,
    monkeypatch: pytest.MonkeyPatch,
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
    assert all(width != model.decoder.channels[-1] for width in widths)


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
    assert out.shape == (int(data["batch"].max()) + 1, model.num_features)


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
    assert model.head.channel_list == [model.num_features, 16, 7]


def test_pointnext_classification_headless_keeps_features_in_train_mode() -> None:
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
        dropout=0.9,
    )
    model.train()
    x = torch.ones(64, model.num_features)
    pos = torch.zeros(64, 3)
    batch = torch.arange(64)
    assert torch.equal(model.forward_head(x, pos, batch, pre_logits=True), x)


def test_pointnext_segmentation_headless_keeps_features_in_train_mode() -> None:
    model = PointNeXtSegmentation(
        in_channels=6,
        num_classes=0,
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
    x = torch.ones(64, model.num_features)
    assert torch.equal(model.forward_head(x, pre_logits=True), x)


def test_pointnext_part_segmentation_headless_keeps_features_in_train_mode() -> None:
    model = PointNeXtPartSegmentation(
        in_channels=6,
        num_classes=0,
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
        dropout=0.9,
    )
    model.train()
    x = torch.ones(64, model.num_features // 3)
    batch = torch.zeros(64, dtype=torch.long)
    assert torch.equal(model.forward_head(x, batch, pre_logits=True), torch.ones(64, model.num_features))


def test_pointnext_fixture_points_have_neighbors_within_the_first_radius(data: Dict[str, Tensor]) -> None:
    row, _ = radius(data["pos"], data["pos"], 0.1, batch_x=data["batch"], batch_y=data["batch"], max_num_neighbors=16)
    degree = torch.bincount(row, minlength=data["pos"].size(0))
    assert degree.float().mean() > 4


def test_pointnext_segmentation_logits_depend_on_the_neighbors(
    model_seg: PointNeXtSegmentation, data: Dict[str, Tensor]
) -> None:
    model_seg.eval()
    pos, batch = data["pos"], data["batch"]
    features = data["features"].clone()
    with torch.no_grad():
        logits = model_seg(features, pos, batch)
        features[0] += 10.0
        perturbed = model_seg(features, pos, batch)
    changed = (logits - perturbed).abs().amax(dim=1) > 1e-6
    assert changed[batch == 0].sum() > 1
    assert not changed[batch == 1].any()


@pytest.mark.parametrize(
    "size",
    [
        pytest.param("sm", id="sm"),
        pytest.param("base", id="base"),
        pytest.param("lg", id="lg"),
        pytest.param("xl", id="xl"),
    ],
)
def test_pointnext_segmentation_config_builds_the_released_architecture(size: str) -> None:
    released_hparams = model_info(f"pointnext-{size}.s3dis-area5.openpoints", "semantic-segmentation")["hparams"]
    released = create_model(f"pointnext-{size}.s3dis-area5.openpoints", task="semantic-segmentation")
    config = create_model(
        f"pointnext-{size}",
        task="semantic-segmentation",
        in_channels=released_hparams["in_channels"],
        num_classes=released_hparams["num_classes"],
    )
    released_shapes = {key: tuple(value.shape) for key, value in released.state_dict().items()}
    config_shapes = {key: tuple(value.shape) for key, value in config.state_dict().items()}
    assert config_shapes == released_shapes


def test_pointnext_s3dis_checkpoint_head_rows_follow_the_dataset_classes() -> None:
    info = model_info("pointnext-sm.s3dis-area5.openpoints", "semantic-segmentation")
    weights = info["weights"]
    assert weights is not None
    filter_fn = weights["pretrained_filter_fn"]
    rows = torch.arange(13.0)
    state_dict = {"head.lins.1.weight": rows.unsqueeze(1).repeat(1, 4), "head.lins.1.bias": rows.clone()}
    filtered = filter_fn(state_dict, torch.nn.Identity())
    checkpoint_classes = [S3DIS_HDF5_CLASSES[int(row)] for row in filtered["head.lins.1.bias"]]
    assert tuple(checkpoint_classes) == tuple(S3DIS_CLASSES)
    assert torch.equal(filtered["head.lins.1.weight"][:, 0], filtered["head.lins.1.bias"])
    assert torch.equal(state_dict["head.lins.1.bias"], rows)


@pytest.mark.parametrize(
    "size, area",
    [
        pytest.param(size, area, id=f"{size}-area{area}")
        for size in ("sm", "base", "lg", "xl")
        for area in range(1, 7)
        if (size, area) != ("xl", 6)
    ],
)
def test_pointnext_s3dis_entries_keep_the_dataset_labels(size: str, area: int) -> None:
    info = model_info(f"pointnext-{size}.s3dis-area{area}.openpoints", "semantic-segmentation")
    weights = info["weights"]
    assert weights is not None
    assert "pretrained_filter_fn" in weights
    segment = torch.arange(13).repeat(8)
    sample = {
        DataKeys.POS: torch.rand(104, 3) * 4.0,
        DataKeys.COLOR: torch.randint(0, 256, (104, 3)).float(),
        DataKeys.SEGMENT: segment.clone(),
    }
    transform = info["transform"]
    assert transform is not None
    data = transform(sample)
    assert torch.equal(data[DataKeys.ORIGIN_SEGMENT], segment)
