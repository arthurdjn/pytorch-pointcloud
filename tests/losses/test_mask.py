import torch
import torch.nn.functional as F

from torch_pointcloud.losses import DiceLoss, dice_loss, pairwise_binary_cross_entropy_with_logits, pairwise_dice_loss


def test_dice_loss_hand_checked() -> None:
    """Sigmoid $p = 0.5$ everywhere against two of four positives: $1 - (2 + 1) / (2 + 2 + 1)$."""
    loss = dice_loss(torch.zeros(1, 4), torch.tensor([[1.0, 1.0, 0.0, 0.0]]))
    torch.testing.assert_close(loss, torch.tensor([0.4]))


def test_dice_loss_module_averages_masks() -> None:
    logits = torch.randn(3, 10)
    targets = (torch.rand(3, 10) > 0.5).float()
    torch.testing.assert_close(DiceLoss(smooth=2.0)(logits, targets), dice_loss(logits, targets, smooth=2.0).mean())


def test_pairwise_losses_match_the_elementwise_losses() -> None:
    torch.manual_seed(0)
    logits = torch.randn(5, 30)
    targets = (torch.rand(3, 30) > 0.5).float()
    dice = pairwise_dice_loss(logits, targets)
    bce = pairwise_binary_cross_entropy_with_logits(logits, targets)
    assert dice.shape == bce.shape == (5, 3)
    for q in range(5):
        for i in range(3):
            torch.testing.assert_close(dice[q, i], dice_loss(logits[q], targets[i]))
            torch.testing.assert_close(bce[q, i], F.binary_cross_entropy_with_logits(logits[q], targets[i]))
