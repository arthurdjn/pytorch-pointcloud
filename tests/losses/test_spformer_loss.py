from typing import Any, Dict

import torch
import torch.nn.functional as F

from torch_pointcloud.losses import SPFormerLoss
from torch_pointcloud.utils.data import DataKeys


def _output(mask_logits: torch.Tensor, class_logits: torch.Tensor) -> Dict[str, Any]:
    """Two scenes of 3 and 2 superpoints (one point each), 3 queries, 2 classes."""
    prediction = {
        "class_logits": class_logits,
        "scores": torch.zeros(2, 3, requires_grad=True),
        "mask_logits": mask_logits,
    }
    return {
        **prediction,
        "superpoint": torch.arange(5),
        "superpoint_batch": torch.tensor([0, 0, 0, 1, 1]),
        "aux_outputs": [prediction],
    }


def _data() -> Dict[str, Any]:
    # Scene 0: instance 4 (class 1) on superpoints 0-1; scene 1: instance 0 (class 0) on its superpoint 1.
    return {
        DataKeys.INSTANCE: torch.tensor([4, 4, -1, -1, 0]),
        DataKeys.SEGMENT: torch.tensor([1, 1, -1, 0, 0]),
    }


def test_spformer_loss_terms_and_gradients() -> None:
    torch.manual_seed(0)
    mask_logits = torch.randn(3, 5, requires_grad=True)
    class_logits = torch.randn(2, 3, 3, requires_grad=True)
    output = _output(mask_logits, class_logits)
    losses = SPFormerLoss(num_classes=2)(output, _data())
    assert set(losses) == {"loss", "class_loss", "mask_loss", "dice_loss", "score_loss"}
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()
    assert mask_logits.grad is not None and mask_logits.grad.abs().sum() > 0
    assert class_logits.grad is not None and class_logits.grad.abs().sum() > 0


def test_spformer_loss_perfect_prediction() -> None:
    """Query 1 predicts scene 0's instance and query 0 scene 1's: the mask terms vanish and the score term
    regresses the IoU of 1 from the zero scores."""
    mask_logits = torch.full((3, 5), -20.0)
    mask_logits[1, :2] = 20.0
    mask_logits[0, 4] = 20.0
    class_logits = torch.full((2, 3, 3), -20.0)
    class_logits[:, :, 2] = 20.0
    class_logits[0, 1] = torch.tensor([-20.0, 20.0, -20.0])
    class_logits[1, 0] = torch.tensor([20.0, -20.0, -20.0])
    losses = SPFormerLoss(num_classes=2)(_output(mask_logits, class_logits), _data())
    assert losses["class_loss"] < 1e-6
    assert losses["mask_loss"] < 1e-6
    assert losses["dice_loss"] < 1e-6
    torch.testing.assert_close(losses["score_loss"], torch.tensor(0.5))


def test_spformer_loss_int32_labels() -> None:
    torch.manual_seed(0)
    output = _output(torch.randn(3, 5), torch.randn(2, 3, 3))
    data = {key: value.int() for key, value in _data().items()}
    expected = SPFormerLoss(num_classes=2)(output, _data())
    torch.testing.assert_close(SPFormerLoss(num_classes=2)(output, data), expected)


def test_spformer_loss_scene_without_instance_supervises_its_classes_only() -> None:
    """Scene 1 has no instance: the mask terms are scene 0's, averaged over both scenes (the dice term summed)."""
    torch.manual_seed(0)
    mask_logits = torch.randn(3, 5, requires_grad=True)
    class_logits = torch.randn(2, 3, 3, requires_grad=True)
    data = {DataKeys.INSTANCE: torch.tensor([4, 4, -1, -1, -1]), DataKeys.SEGMENT: torch.tensor([1, 1, -1, 0, 0])}
    losses = SPFormerLoss(num_classes=2)(_output(mask_logits, class_logits), data)
    losses["loss"].backward()
    assert mask_logits.grad is not None and class_logits.grad is not None
    assert mask_logits.grad[:, :3].abs().sum() > 0
    assert mask_logits.grad[:, 3:].abs().sum() == 0
    assert class_logits.grad[1].abs().sum() > 0

    alone = _output(mask_logits.detach()[:, :3], class_logits.detach()[:1])
    alone.update(superpoint=torch.arange(3), superpoint_batch=torch.zeros(3, dtype=torch.long))
    alone["scores"] = alone["aux_outputs"][0]["scores"] = torch.zeros(1, 3)
    expected = SPFormerLoss(num_classes=2)(alone, {key: value[:3] for key, value in data.items()})
    torch.testing.assert_close(losses["mask_loss"], expected["mask_loss"] / 2)
    torch.testing.assert_close(losses["dice_loss"], expected["dice_loss"])
    torch.testing.assert_close(losses["score_loss"], expected["score_loss"] / 2)


def test_spformer_loss_batch_without_instances_is_the_no_object_class_loss() -> None:
    """Without any instance every query is assigned the no-object slot, in both decoder layers."""
    torch.manual_seed(0)
    mask_logits = torch.randn(3, 5, requires_grad=True)
    class_logits = torch.randn(2, 3, 3, requires_grad=True)
    data = {DataKeys.INSTANCE: torch.full((5,), -1), DataKeys.SEGMENT: torch.tensor([1, 1, -1, 0, 0])}
    criterion = SPFormerLoss(num_classes=2)
    losses = criterion(_output(mask_logits, class_logits), data)
    no_object = torch.full((2, 3), 2)
    class_loss = 0.5 * F.cross_entropy(class_logits.transpose(1, 2), no_object, criterion.class_weights)
    torch.testing.assert_close(losses["class_loss"], class_loss)
    torch.testing.assert_close(losses["loss"], 2 * class_loss)
    assert (losses["mask_loss"], losses["dice_loss"], losses["score_loss"]) == (0, 0, 0)
    losses["loss"].backward()
    assert mask_logits.grad is None or mask_logits.grad.abs().sum() == 0
