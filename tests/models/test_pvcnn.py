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


def test_pvconv_voxel_branch_uses_generic_act_and_norm() -> None:
    """Default PVConv builds its voxel branch from the generic `act` / `norm` kwargs.

    No paper-specific defaults are baked into the block — `act="relu"` yields
    `nn.ReLU` and `norm="batch_norm"` (with `dim=3` inside Conv3dBlock) yields
    `nn.BatchNorm3d` with PyTorch's default eps.
    """
    pv = PVConv(in_channels=4, out_channels=8, kernel_size=3, resolution=4, act="relu")
    assert isinstance(pv.voxel_layers[1], nn.BatchNorm3d)
    assert pv.voxel_layers[1].eps == 1e-5
    assert isinstance(pv.voxel_layers[2], nn.ReLU)


def test_pvcnn_mit_han_lab_factory_patches_voxel_branch_to_paper_recipe() -> None:
    """The registered S3DIS Area-5 factory should patch every PVConv voxel branch
    to upstream's LeakyReLU(0.1) + BN3d(eps=1e-4) recipe after construction,
    while keeping the point branch on ReLU."""
    from torch_pointcloud.models._registry import create_model

    model = create_model("pvcnn-mit-han-lab.s3dis-area5", task="segmentation", pretrained=False)
    pvconvs = [m for m in model.modules() if isinstance(m, PVConv)]
    assert pvconvs, "Expected at least one PVConv in the factory-built model."
    for pv in pvconvs:
        for layer in pv.voxel_layers:
            if isinstance(layer, nn.BatchNorm3d):
                assert layer.eps == 1e-4
            elif isinstance(layer, (nn.ReLU, nn.LeakyReLU)):
                assert isinstance(layer, nn.LeakyReLU)
                assert layer.negative_slope == 0.1
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
