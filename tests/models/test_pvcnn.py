from typing import Dict

import pytest
import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import MLP
from torch_geometric.nn.dense.linear import Linear as PyGLinear

from torch_pointcloud.layers.pvcnn_blocks import PVConv
from torch_pointcloud.models.pvcnn import PVCNNClassification, PVCNNSegmentation
from torch_pointcloud.utils.imports import _TORCH_SCATTER_AVAILABLE

# See: https://docs.pytest.org/en/stable/how-to/skipping.html#summary
pytestmark = pytest.mark.skipif(
    not _TORCH_SCATTER_AVAILABLE,
    reason="torch-scatter is not installed",
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
def model_clf() -> PVCNNClassification:
    return PVCNNClassification(
        in_channels=6,
        num_classes=10,
        channels=[32, 64, 128],
        global_channels=[64, 32],
        depths=[1, 1, 1],
        kernel_sizes=[3, 3, 3],
        resolutions=[8, 8, 8],
        use_se=False,
        normalize=True,
        dropout=0.0,
        global_pool="max",
        act="relu",
        act_kwargs=None,
        act_first=False,
        norm="batch_norm",
        norm_kwargs=None,
    )


@pytest.fixture
def model_seg() -> PVCNNSegmentation:
    return PVCNNSegmentation(
        in_channels=6,
        num_classes=10,
        channels=[32, 64, 128],
        global_channels=[64, 32],
        depths=[1, 1, 1],
        kernel_sizes=[3, 3, 3],
        resolutions=[8, 8, 8],
        spatial_dim=3,
        use_se=False,
        normalize=True,
        dropout=0.0,
        global_pool="max",
        act="relu",
        act_kwargs=None,
        act_first=False,
        norm="batch_norm",
        norm_kwargs=None,
    )


def test_pvcnn_classification_forward(model_clf: PVCNNClassification, data: Dict[str, Tensor]) -> None:
    logits = model_clf(data["x"], data["pos"], data["batch"])
    assert logits.shape == (int(data["batch"].max()) + 1, model_clf.num_classes)
    assert logits.dtype == data["x"].dtype


def test_pvcnn_segmentation_forward(model_seg: PVCNNSegmentation, data: Dict[str, Tensor]) -> None:
    logits = model_seg(data["x"], data["pos"], data["batch"])
    assert logits.shape == (data["pos"].shape[0], model_seg.num_classes)
    assert logits.dtype == data["x"].dtype


def test_pvcnn_classification_forward_features_and_head(
    model_clf: PVCNNClassification, data: Dict[str, Tensor]
) -> None:
    x = model_clf.forward_features(data["x"], data["pos"], data["batch"])
    assert x.dim() == 2
    logits = model_clf.forward_head(x, data["batch"])
    assert logits.shape == (int(data["batch"].max()) + 1, model_clf.num_classes)


def test_pvcnn_segmentation_forward_features_decoder_head(
    model_seg: PVCNNSegmentation, data: Dict[str, Tensor]
) -> None:
    x, intermediates = model_seg.forward_features(data["x"], data["pos"], data["batch"], return_intermediates=True)
    assert len(intermediates) > 0
    x = model_seg.forward_decoder(x, data["batch"], intermediates)
    assert x.shape[0] == data["pos"].shape[0]
    logits = model_seg.forward_head(x)
    assert logits.shape == (data["pos"].shape[0], model_seg.num_classes)


def test_pvcnn_classification_reset_classifier(model_clf: PVCNNClassification, data: Dict[str, Tensor]) -> None:
    model_clf.reset_classifier(num_classes=42)
    logits = model_clf(data["x"], data["pos"], data["batch"])
    assert logits.shape == (int(data["batch"].max()) + 1, 42)


def test_pvcnn_classification_reset_classifier_zero(model_clf: PVCNNClassification) -> None:
    model_clf.reset_classifier(num_classes=0)
    assert isinstance(model_clf.head, nn.Identity)


def test_pvcnn_segmentation_reset_classifier(model_seg: PVCNNSegmentation, data: Dict[str, Tensor]) -> None:
    model_seg.reset_classifier(num_classes=42)
    logits = model_seg(data["x"], data["pos"], data["batch"])
    assert logits.shape == (data["pos"].shape[0], 42)


def test_pvcnn_segmentation_reset_classifier_zero(model_seg: PVCNNSegmentation) -> None:
    model_seg.reset_classifier(num_classes=0)
    assert isinstance(model_seg.head, nn.Identity)


def test_pvconv_voxel_branch_uses_generic_act_and_norm() -> None:
    """Default PVConv builds its voxel branch from the generic `act` / `norm` kwargs.

    No paper-specific defaults are baked into the block — `act="relu"` yields
    `nn.ReLU` and `norm="batch_norm"` (with `dim=3` inside Conv3dBlock) yields
    `nn.BatchNorm3d` with PyTorch's default eps.
    """
    from torch_pointcloud.layers.conv3d_blocks import Conv3dBlock

    pv = PVConv(in_channels=4, out_channels=8, kernel_size=3, resolution=4, act="relu")
    first = pv.voxel_layers[0]
    assert isinstance(first, Conv3dBlock)
    assert isinstance(first.norm, nn.BatchNorm3d)
    assert first.norm.eps == 1e-5
    assert isinstance(first.act, nn.ReLU)


def test_pvcnn_mit_han_lab_factory_patches_voxel_branch_to_paper_recipe() -> None:
    """The registered S3DIS Area-5 factory should patch every PVConv voxel branch
    to upstream's LeakyReLU(0.1) + BN3d(eps=1e-4) recipe after construction,
    while keeping the point branch on ReLU."""
    from torch_pointcloud.layers.conv3d_blocks import Conv3dBlock
    from torch_pointcloud.models._registry import create_model

    model = create_model("pvcnn.s3dis-area5.mit-han-lab", task="segmentation", pretrained=False)
    pvconvs = [m for m in model.modules() if isinstance(m, PVConv)]
    assert pvconvs, "Expected at least one PVConv in the factory-built model."
    for pv in pvconvs:
        for block in pv.voxel_layers:
            if not isinstance(block, Conv3dBlock):
                continue
            assert isinstance(block.norm, nn.BatchNorm3d)
            assert block.norm.eps == 1e-4
            assert isinstance(block.act, nn.LeakyReLU)
            assert block.act.negative_slope == 0.1
        assert pv.mlp.act.__class__.__name__ == "ReLU"


def test_pvcnn_segmentation_with_head_channels_creates_mlp_head() -> None:
    """`head_channels` should swap the default Linear head for a multi-layer MLP that
    matches our paper-faithful registered model."""
    model = PVCNNSegmentation(
        in_channels=9,
        num_classes=13,
        channels=[32, 64],
        depths=[1, 1],
        kernel_sizes=[3, 3],
        resolutions=[8, 0],
        global_channels=[32, 16],
        head_channels=[24, 16],
        head_dropout=0.3,
    )
    head = model.head
    assert isinstance(head, MLP)
    # PyG MLP: 3 lins, 2 norms (last lin is plain).
    assert len(head.lins) == 3
    for lin, expected_out in zip(head.lins, (24, 16, 13)):
        assert isinstance(lin, PyGLinear)
        assert lin.out_channels == expected_out
