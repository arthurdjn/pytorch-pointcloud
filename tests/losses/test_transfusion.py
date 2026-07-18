from typing import Any, Dict, Tuple

import torch
from torch import Tensor

from torch_pointcloud.losses import TransFusionLoss
from torch_pointcloud.utils.data import DataKeys

_POINT_CLOUD_RANGE = (-12.0, -12.0, -2.0, 12.0, 12.0, 4.0)
_VOXEL_SIZE = (1.0, 1.0, 6.0)
_STRIDE = 1
_NUM_CLASSES = 3
_CODE_WEIGHTS = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0)


def _data(batch_size: int = 2, num_queries: int = 20, size: int = 24) -> Tuple[Dict[str, Tensor], Dict[str, Any]]:
    torch.manual_seed(0)
    output: Dict[str, Tensor] = {
        "center": torch.rand(batch_size, 2, num_queries) * size,
        "height": torch.randn(batch_size, 1, num_queries),
        "dim": torch.randn(batch_size, 3, num_queries) * 0.2 + 1.0,
        "rot": torch.randn(batch_size, 2, num_queries),
        "vel": torch.randn(batch_size, 2, num_queries),
        "iou": torch.randn(batch_size, 1, num_queries),
        "heatmap": torch.randn(batch_size, _NUM_CLASSES, num_queries),
        "dense_heatmap": torch.randn(batch_size, _NUM_CLASSES, size, size),
    }
    box = torch.tensor(
        [
            [2.0, 3.0, 0.2, 3.5, 2.0, 1.5, 0.4],
            [-5.0, 4.0, 0.1, 4.0, 1.8, 1.6, -0.6],
            [6.0, -3.0, 0.0, 3.2, 1.7, 1.5, 1.1],
        ]
    )
    batch: Dict[str, Any] = {
        DataKeys.BOX: box,
        DataKeys.LABEL: torch.tensor([0, 1, 2]),
        DataKeys.BATCH_BOX: torch.tensor([0, 0, 1]),
    }
    return output, batch


def _loss() -> TransFusionLoss:
    return TransFusionLoss(_NUM_CLASSES, _POINT_CLOUD_RANGE, _VOXEL_SIZE, _STRIDE, code_weights=_CODE_WEIGHTS)


def _perfect_output(boxes: Tensor, labels: Tensor, num_queries: int = 4) -> Dict[str, Tensor]:
    """Queries 0..K-1 decode exactly to the GT boxes with confident class logits; the rest are background."""
    k = boxes.shape[0]
    center = torch.full((1, 2, num_queries), 1.0)
    height = torch.zeros(1, 1, num_queries)
    dim = torch.zeros(1, 3, num_queries)
    rot = torch.zeros(1, 2, num_queries)
    rot[0, 1] = 1.0
    heatmap = torch.full((1, _NUM_CLASSES, num_queries), -10.0)
    iou = torch.ones(1, 1, num_queries)

    center[0, 0, :k] = (boxes[:, 0] + 12.0) / 1.0
    center[0, 1, :k] = (boxes[:, 1] + 12.0) / 1.0
    height[0, 0, :k] = boxes[:, 2]
    dim[0, :, :k] = boxes[:, 3:6].log().t()
    rot[0, 0, :k] = torch.sin(boxes[:, 6])
    rot[0, 1, :k] = torch.cos(boxes[:, 6])
    heatmap[0, labels, torch.arange(k)] = 10.0
    return {
        "center": center,
        "height": height,
        "dim": dim,
        "rot": rot,
        "vel": torch.zeros(1, 2, num_queries),
        "iou": iou,
        "heatmap": heatmap,
        "dense_heatmap": torch.zeros(1, _NUM_CLASSES, 24, 24),
    }


def test_transfusion_loss_perfect_queries_regression_terms_zero() -> None:
    boxes = torch.tensor([[2.0, 3.0, 0.2, 3.5, 2.0, 1.5, 0.4], [-5.0, 4.0, 0.1, 4.0, 1.8, 1.6, -0.6]])
    labels = torch.tensor([0, 2])
    batch: Dict[str, Any] = {
        DataKeys.BOX: boxes,
        DataKeys.LABEL: labels,
        DataKeys.BATCH_BOX: torch.zeros(2, dtype=torch.long),
    }
    output = _perfect_output(boxes, labels)
    out = _loss()(output, batch)
    assert out["bbox_loss"] < 1e-5
    assert out["iou_loss"] < 1e-5
    assert out["cls_loss"] < 1e-3


def test_transfusion_loss_perturbed_queries_are_larger() -> None:
    boxes = torch.tensor([[2.0, 3.0, 0.2, 3.5, 2.0, 1.5, 0.4], [-5.0, 4.0, 0.1, 4.0, 1.8, 1.6, -0.6]])
    labels = torch.tensor([0, 2])
    batch: Dict[str, Any] = {
        DataKeys.BOX: boxes,
        DataKeys.LABEL: labels,
        DataKeys.BATCH_BOX: torch.zeros(2, dtype=torch.long),
    }
    output = _perfect_output(boxes, labels)
    perfect = _loss()(output, batch)
    output["center"][0, 0, 0] += 0.5
    out = _loss()(output, batch)
    assert out["bbox_loss"] > perfect["bbox_loss"] + 1e-3
    assert out["loss"] > perfect["loss"]


def test_transfusion_loss_returns_scalar_dict() -> None:
    output, batch = _data()
    out = _loss()(output, batch)
    for key in ("loss", "heatmap_loss", "cls_loss", "bbox_loss", "iou_loss"):
        assert key in out, key
        assert out[key].ndim == 0
        assert torch.isfinite(out[key])


def test_transfusion_loss_backward() -> None:
    output, batch = _data()
    for value in output.values():
        value.requires_grad_(True)
    _loss()(output, batch)["loss"].backward()
    assert output["dense_heatmap"].grad is not None
    assert output["heatmap"].grad is not None
    assert output["iou"].grad is not None


def test_transfusion_loss_no_boxes_is_finite() -> None:
    output, batch = _data()
    batch[DataKeys.BOX] = batch[DataKeys.BOX][:0]
    batch[DataKeys.LABEL] = batch[DataKeys.LABEL][:0]
    batch[DataKeys.BATCH_BOX] = batch[DataKeys.BATCH_BOX][:0]
    assert torch.isfinite(_loss()(output, batch)["loss"])
