from typing import Any, Dict

import pytest
import torch

pytest.importorskip("torch_scatter")
pytest.importorskip("torch_cluster")

from torch_pointcloud.lightning import LitDetectionModel
from torch_pointcloud.losses import VoteNetLoss
from torch_pointcloud.models import create_model


def _optimizer(params: Any) -> torch.optim.Optimizer:
    return torch.optim.Adam(params, lr=1e-3)


def _criterion(mean_size_arr: Any) -> VoteNetLoss:
    return VoteNetLoss(num_heading_bin=12, num_size_cluster=10, num_class=10, mean_size_arr=mean_size_arr)


def _synthetic_batch(num_scenes: int, points_per_scene: int, max_boxes: int = 4) -> Dict[str, Any]:
    n = num_scenes * points_per_scene
    return {
        "x": torch.rand(n, 1),
        "pos": torch.rand(n, 3),
        "batch": torch.arange(num_scenes).repeat_interleave(points_per_scene),
        "center_label": torch.rand(num_scenes, max_boxes, 3),
        "heading_class_label": torch.randint(0, 12, (num_scenes, max_boxes)),
        "heading_residual_label": torch.rand(num_scenes, max_boxes),
        "size_class_label": torch.randint(0, 10, (num_scenes, max_boxes)),
        "size_residual_label": torch.rand(num_scenes, max_boxes, 3),
        "sem_cls_label": torch.randint(0, 10, (num_scenes, max_boxes)),
        "box_label_mask": torch.ones(num_scenes, max_boxes),
        "vote_label": torch.rand(num_scenes, points_per_scene, 9),
        "vote_label_mask": torch.randint(0, 2, (num_scenes, points_per_scene)).float(),
    }


def _module() -> LitDetectionModel:
    torch.manual_seed(0)
    model = create_model("votenet-fair-base.sunrgbd", task="detection")
    return LitDetectionModel(model, optimizer=_optimizer, criterion=_criterion)


def test_detection_training_step_returns_finite_scalar_loss() -> None:
    module = _module()
    batch = _synthetic_batch(num_scenes=2, points_per_scene=1024)
    loss = module.training_step(batch, 0)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert loss.requires_grad
    assert torch.isfinite(loss)


def test_detection_validation_logs_map() -> None:
    module = _module()
    module.eval()
    logged: Dict[str, Any] = {}
    module.log = lambda name, value, **kwargs: logged.__setitem__(name, value)  # type: ignore[method-assign]
    batch = _synthetic_batch(num_scenes=2, points_per_scene=1024)
    module.on_validation_epoch_start()
    with torch.no_grad():
        module.validation_step(batch, 0)
    module.on_validation_epoch_end()
    assert "val/mAP@0.25" in logged
    assert "val/mAP@0.5" in logged
    assert torch.isfinite(torch.as_tensor(float(logged["val/mAP@0.25"])))
