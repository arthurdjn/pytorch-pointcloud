from typing import Dict, Tuple

import pytest
import torch
from torch import Tensor, nn
from torch_geometric.nn import MLP

from torch_pointcloud.models import create_model, model_info
from torch_pointcloud.models._registry import Task
from torch_pointcloud.models.pointnet import PointNetClassification, PointNetEncoder, PointNetSegmentation
from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.imports import _PYG_LIB_AVAILABLE


@pytest.fixture
def data() -> Dict[str, Tensor]:
    torch.manual_seed(42)
    lengths = torch.tensor([256, 512])
    pos = torch.randn(int(lengths.sum()), 3)
    x = torch.randn(int(lengths.sum()), 6)
    batch = torch.repeat_interleave(torch.arange(len(lengths)), lengths)
    return dict(x=x, pos=pos, batch=batch)


@pytest.fixture
def model_clf() -> PointNetClassification:
    return PointNetClassification(
        num_classes=10,
        in_channels=6,
        spatial_dim=3,
        dropout=0.0,
        global_pool="max",
        mlp1_channels=(32,),
        mlp2_channels=(64, 128),
        act="relu",
        norm="batch_norm",
        use_features_transform=True,
        tnet_mlp1_dims=(32, 64, 128),
        tnet_mlp2_dims=(64, 32),
        tnet_act="relu",
        tnet_norm="batch_norm",
        head_channels=(64, 32),
    )


@pytest.fixture
def model_seg() -> PointNetSegmentation:
    return PointNetSegmentation(
        num_classes=10,
        spatial_dim=3,
        in_channels=6,
        dropout=0.3,
        mlp1_channels=(32,),
        mlp2_channels=(64, 128),
        act="relu",
        norm="batch_norm",
        global_pool="max",
        use_features_transform=True,
        tnet_mlp1_dims=(32, 64, 128),
        tnet_mlp2_dims=(64, 32),
        tnet_act="relu",
        tnet_norm="batch_norm",
        seg_head_dims=(64, 32),
    )


def test_pointnet_classification_forward(model_clf: PointNetClassification, data: Dict[str, Tensor]) -> None:
    logits = model_clf(data["x"], data["pos"], data["batch"])
    assert logits.shape == (int(data["batch"].max()) + 1, model_clf.num_classes)
    assert logits.dtype == data["x"].dtype


def test_pointnet_classification_reset_classifier(model_clf: PointNetClassification, data: Dict[str, Tensor]) -> None:
    model_clf.reset_classifier(num_classes=42)
    logits = model_clf(data["x"], data["pos"], data["batch"])
    assert logits.shape == (int(data["batch"].max()) + 1, 42)


def test_pointnet_segmentation_forward(model_seg: PointNetSegmentation, data: Dict[str, Tensor]) -> None:
    logits = model_seg(data["x"], data["pos"], data["batch"])
    assert logits.shape == (data["pos"].shape[0], model_seg.num_classes)
    assert logits.dtype == data["x"].dtype


def test_pointnet_classification_forward_features_and_head(
    model_clf: PointNetClassification,
    data: Dict[str, Tensor],
) -> None:
    x = model_clf.forward_features(data["x"], data["pos"], data["batch"])
    assert x.dim() == 2
    logits = model_clf.forward_head(x, data["batch"])
    assert logits.shape == (int(data["batch"].max()) + 1, model_clf.num_classes)


def test_pointnet_segmentation_forward_features_and_head(
    model_seg: PointNetSegmentation,
    data: Dict[str, Tensor],
) -> None:
    x, point_features = model_seg.forward_features(data["x"], data["pos"], data["batch"])
    assert x.shape[0] == point_features.shape[0] == data["pos"].shape[0]
    logits = model_seg.forward_head(x, point_features, data["batch"])
    assert logits.shape == (data["pos"].shape[0], model_seg.num_classes)


def test_pointnet_classification_head_channels(model_clf: PointNetClassification) -> None:
    assert model_clf.head.channel_list == [model_clf.num_features, 64, 32, model_clf.num_classes]


def test_pointnet_segmentation_head_keeps_full_chain(model_seg: PointNetSegmentation) -> None:
    assert model_seg.head.channel_list == [model_seg.num_features, 64, 32, model_seg.num_classes]


@pytest.mark.parametrize(
    ("act_last", "nonnegative"),
    [
        pytest.param(True, True, id="act-last"),
        pytest.param(False, False, id="no-act-last"),
    ],
)
def test_pointnet_encoder_act_last(act_last: bool, nonnegative: bool, data: Dict[str, Tensor]) -> None:
    encoder = PointNetEncoder(
        in_channels=6,
        mlp1_channels=(32,),
        mlp2_channels=(64, 128),
        tnet_mlp1_dims=(32, 64),
        tnet_mlp2_dims=(32,),
        act_first=False,
        act_last=act_last,
    )
    out = encoder(data["x"], data["pos"], data["batch"])
    assert out.shape == (data["pos"].shape[0], 128)
    assert bool((out >= 0).all()) is nonnegative


@pytest.mark.parametrize("act_first", [pytest.param(True, id="act-first"), pytest.param(False, id="norm-first")])
def test_pointnet_encoder_act_first(act_first: bool) -> None:
    encoder = PointNetEncoder(in_channels=6, act_first=act_first)
    assert encoder.stnet.local_nn.act_first is act_first
    assert encoder.ftnet is not None and encoder.ftnet.local_nn.act_first is act_first
    assert encoder.mlp1.act_first is act_first
    assert encoder.mlp2.act_first is act_first


@pytest.mark.parametrize(
    ("input_transform_uses_features", "input_transform_in_channels"),
    [
        pytest.param(False, 3, id="coordinates"),
        pytest.param(True, 9, id="coordinates-and-features"),
    ],
)
def test_pointnet_encoder_input_transform_uses_features(
    input_transform_uses_features: bool,
    input_transform_in_channels: int,
    data: Dict[str, Tensor],
) -> None:
    encoder = PointNetEncoder(in_channels=6, input_transform_uses_features=input_transform_uses_features)
    out = encoder(data["x"], data["pos"], data["batch"])
    assert encoder.stnet.local_nn.channel_list[0] == input_transform_in_channels
    assert out.shape == (data["pos"].shape[0], 1024)


@pytest.mark.parametrize("act_first", [pytest.param(True, id="act-first"), pytest.param(False, id="norm-first")])
def test_pointnet_encoder_second_mlp_matches_pyg_mlp(act_first: bool, data: Dict[str, Tensor]) -> None:
    torch.manual_seed(0)
    encoder = PointNetEncoder(mlp1_channels=(32,), mlp2_channels=(64, 128), act_first=act_first).eval()
    for module in encoder.modules():
        if isinstance(module, nn.BatchNorm1d):
            assert module.running_mean is not None and module.running_var is not None
            nn.init.uniform_(module.weight, 0.5, 1.5)
            nn.init.uniform_(module.bias, -0.5, 0.5)
            module.running_mean.uniform_(-0.5, 0.5)
            module.running_var.uniform_(0.5, 1.5)
    reference = MLP([32, 64, 128], act="relu", norm="batch_norm", act_first=act_first, plain_last=False).eval()
    state = dict(encoder.mlp2.state_dict())
    state.update({f"norms.1.{key}": value for key, value in encoder.out_norm.state_dict().items()})
    reference.load_state_dict(state)

    inputs: Dict[str, Tensor] = {}
    encoder.mlp2.register_forward_hook(lambda module, args, output: inputs.update(x=args[0]))
    with torch.no_grad():
        out = encoder(None, data["pos"], data["batch"])
        expected = reference(inputs["x"])
    torch.testing.assert_close(out, expected)


@pytest.mark.parametrize(
    ("name", "task", "sample", "logits_shape"),
    [
        pytest.param(
            "pointnet.modelnet40.jiawei-ren",
            "classification",
            {DataKeys.POS: torch.randn(2048, 3)},
            (1, 40),
            id="modelnet40",
        ),
        pytest.param(
            "pointnet.scanobjectnn-hardest.openpoints",
            "classification",
            {DataKeys.POS: torch.randn(2048, 3)},
            (1, 15),
            marks=pytest.mark.skipif(not _PYG_LIB_AVAILABLE, reason="pyg-lib is not installed"),
            id="scanobjectnn",
        ),
        pytest.param(
            "pointnet.s3dis-area5.xu-yan",
            "semantic-segmentation",
            {
                DataKeys.POS: torch.randn(4096, 3),
                DataKeys.COLOR: torch.rand(4096, 3),
                DataKeys.NORM_POS: torch.rand(4096, 3),
            },
            (4096, 13),
            id="s3dis",
        ),
    ],
)
def test_pointnet_registered_transform_runs_without_labels(
    name: str,
    task: Task,
    sample: Dict[str, Tensor],
    logits_shape: Tuple[int, ...],
) -> None:
    transform = model_info(name, task=task)["transform"]
    assert transform is not None
    out = transform(sample)
    model = create_model(name, task=task, pretrained=False).eval()
    batch = torch.zeros(out[DataKeys.POS].shape[0], dtype=torch.long)
    with torch.no_grad():
        logits = model(out.get(DataKeys.X), out[DataKeys.POS], batch)
    assert tuple(logits.shape) == logits_shape
