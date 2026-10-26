from typing import Any, Dict

import torch

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
