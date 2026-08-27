from typing import Dict

import pytest
import torch
from torch import Tensor

from torch_pointcloud.models.spvcnn import (
    PointTensor,
    ResidualBlock,
    SparseTensor,
    SPVCNNClassification,
    SPVCNNDecoder,
    SPVCNNDecoderBlock,
    SPVCNNSegmentation,
    _spvcnn_semantickitti_transforms,
    point_to_voxel,
)
from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.imports import _CUDA_AVAILABLE, _TORCHSPARSE_AVAILABLE

# See: https://docs.pytest.org/en/stable/how-to/skipping.html#summary
pytestmark = [
    pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA is not available"),
    pytest.mark.skipif(not _TORCHSPARSE_AVAILABLE, reason="torchsparse is not installed"),
]


@pytest.fixture(autouse=True)
def _torchsparse_kmap_mode() -> None:
    """torch >= 2.10 rejects torchsparse's default `hashmap_on_the_fly` downsample kmap builder
    (its legacy `make_variable` call hits `set_stride` on a detached coords tensor). The `hashmap`
    builder takes a different C++ path and is unaffected."""
    import torchsparse.nn.functional as spF

    config = spF.conv_config.get_default_conv_config()
    config.kmap_mode = "hashmap"
    spF.conv_config.set_global_conv_config(config)


@pytest.fixture
def data() -> Dict[str, Tensor]:
    torch.manual_seed(42)
    lengths = torch.tensor([512, 768])
    pos = torch.randn(int(lengths.sum()), 3) * 20.0
    x = torch.randn(int(lengths.sum()), 6)
    batch = torch.repeat_interleave(torch.arange(len(lengths)), lengths)
    return dict(
        x=x.cuda(),
        pos=pos.cuda(),
        batch=batch.cuda(),
    )


@pytest.fixture
def model_seg() -> SPVCNNSegmentation:
    return SPVCNNSegmentation(
        in_channels=6,
        num_classes=10,
        spatial_dim=3,
        stem_channels=16,
        encoder_channels=(16, 32, 64),
        encoder_depths=(1, 1, 1),
        encoder_fusion_stages=(False, False, True),
        decoder_channels=(32, 16, 16),
        decoder_depths=(1, 1, 1),
        decoder_fusion_stages=(False, True, False),
        kernel_size=3,
        stride=1,
        dilation=1,
        drop_path=0.3,
        act="relu",
        act_kwargs=None,
        norm="batch_norm",
        norm_kwargs=None,
    ).cuda()


def test_spvcnn_segmentation_forward(model_seg: SPVCNNSegmentation, data: Dict[str, Tensor]) -> None:
    logits = model_seg(data["x"], data["pos"], data["batch"])
    assert logits.shape == (data["pos"].shape[0], model_seg.num_classes)


def test_spvcnn_segmentation_forward_features_decoder_head(
    model_seg: SPVCNNSegmentation, data: Dict[str, Tensor]
) -> None:
    x_voxels, x_points, intermediates = model_seg.forward_features(data["x"], data["pos"], data["batch"])
    assert len(intermediates) > 0
    x_voxels, x_points = model_seg.forward_decoder(x_voxels, x_points, intermediates)
    assert x_points.F.shape[0] == data["pos"].shape[0]
    logits = model_seg.forward_head(x_points.F)
    assert logits.shape == (data["pos"].shape[0], model_seg.num_classes)


def test_spvcnn_segmentation_reset_classifier(model_seg: SPVCNNSegmentation, data: Dict[str, Tensor]) -> None:
    model_seg.reset_classifier(num_classes=42)
    model_seg.cuda()
    logits = model_seg(data["x"], data["pos"], data["batch"])
    assert logits.shape == (data["pos"].shape[0], 42)


def test_spvcnn_segmentation_reset_classifier_zero(model_seg: SPVCNNSegmentation) -> None:
    model_seg.reset_classifier(num_classes=0)
    assert isinstance(model_seg.head, torch.nn.Identity)


@pytest.fixture
def model_clf() -> SPVCNNClassification:
    return SPVCNNClassification(
        in_channels=6,
        num_classes=10,
        spatial_dim=3,
        stem_channels=16,
        encoder_channels=(16, 32, 64),
        encoder_depths=(1, 1, 1),
        encoder_fusion_stages=(False, False, True),
        kernel_size=3,
        stride=1,
        dilation=1,
        drop_path=0.3,
        global_pool="max",
        dropout=0.0,
        act="relu",
        act_kwargs=None,
        norm="batch_norm",
        norm_kwargs=None,
    ).cuda()


def test_spvcnn_classification_forward(model_clf: SPVCNNClassification, data: Dict[str, Tensor]) -> None:
    logits = model_clf(data["x"], data["pos"], data["batch"])
    assert logits.shape == (int(data["batch"].max()) + 1, model_clf.num_classes)


def test_spvcnn_classification_forward_features_and_head(
    model_clf: SPVCNNClassification, data: Dict[str, Tensor]
) -> None:
    x = model_clf.forward_features(data["x"], data["pos"], data["batch"])
    assert x.shape[0] == data["pos"].shape[0]
    logits = model_clf.forward_head(x, data["batch"])
    assert logits.shape == (int(data["batch"].max()) + 1, model_clf.num_classes)


def test_spvcnn_classification_reset_classifier(model_clf: SPVCNNClassification, data: Dict[str, Tensor]) -> None:
    model_clf.reset_classifier(num_classes=42)
    model_clf.cuda()
    logits = model_clf(data["x"], data["pos"], data["batch"])
    assert logits.shape == (int(data["batch"].max()) + 1, 42)


def test_spvcnn_classification_reset_classifier_zero(model_clf: SPVCNNClassification) -> None:
    model_clf.reset_classifier(num_classes=0)
    assert isinstance(model_clf.head, torch.nn.Identity)


def test_spvcnn_classification_reset_classifier_keeps_current_pooling(model_clf: SPVCNNClassification) -> None:
    model_clf.reset_classifier(10, global_pool="mean")
    pool = model_clf.global_pool
    model_clf.reset_classifier(7)
    assert model_clf.global_pool is pool
    assert type(pool).__name__ == "MeanPool"


def test_spvcnn_decoder_block_threads_act_norm_dropout() -> None:
    decoder = SPVCNNDecoder(
        depths=(1,),
        channels=(8, 8),
        skip_channels=(8,),
        fusion_stages=(False,),
        dropout=0.3,
        act="leaky_relu",
        act_kwargs={"negative_slope": 0.1},
        norm=None,
    )
    block = decoder.get_submodule("block0")
    assert isinstance(block, SPVCNNDecoderBlock)
    assert block.dropout == 0.3
    residual = block.blocks[0]
    assert isinstance(residual, ResidualBlock)
    assert isinstance(residual.act, torch.nn.LeakyReLU)
    assert residual.act.negative_slope == 0.1
    assert isinstance(residual.norm1, torch.nn.Identity)
    assert isinstance(residual.norm2, torch.nn.Identity)


def test_spvcnn_semantickitti_transform_back_projects_to_full_resolution() -> None:
    transform = _spvcnn_semantickitti_transforms()
    data = transform(
        {
            DataKeys.POS: torch.tensor([[0.01, 0.01, 0.01], [0.02, 0.02, 0.02], [1.0, 1.0, 1.0]]),
            DataKeys.INTENSITY: torch.tensor([[0.5], [0.7], [0.9]]),
            DataKeys.SEGMENT: torch.tensor([10, 40, 50]),  # car, road, building
        }
    )

    assert torch.equal(data[DataKeys.ORIGIN_SEGMENT], torch.tensor([0, 8, 12]))
    assert data[DataKeys.POS].shape[0] == 2  # points 0 and 1 share a 5 cm voxel
    assert data[DataKeys.SEGMENT].shape[0] == 2
    inverse = data[DataKeys.INVERSE]
    assert inverse.shape == (3,)
    assert int(inverse[0]) == int(inverse[1])
    assert int(inverse[0]) != int(inverse[2])

    preds_sub = data[DataKeys.SEGMENT]  # per-voxel labels stand in for per-voxel predictions
    preds_full = preds_sub[inverse]
    assert torch.equal(preds_full, torch.tensor([0, 0, 12]))


def test_spvcnn_point_to_voxel_masks_out_of_grid_points() -> None:
    x_voxels = SparseTensor(
        feats=torch.zeros(2, 1).cuda(),
        coords=torch.tensor([[0, 0, 0, 0], [0, 1, 1, 1]], dtype=torch.int32).cuda(),
        stride=1,
    )
    x_points = PointTensor(
        feats=torch.tensor([[2.0], [4.0], [100.0]]).cuda(),
        coords=torch.tensor([[0, 0, 0, 0], [0, 0, 0, 0], [0, 5, 5, 5]], dtype=torch.float32).cuda(),
    )

    out = point_to_voxel(x_voxels, x_points)
    assert out.F.shape == (2, 1)
    assert out.F[0].item() == 3.0
    assert out.F[1].item() == 0.0
    assert int(x_points._caches.idx_query[x_voxels.s].min()) == -1
