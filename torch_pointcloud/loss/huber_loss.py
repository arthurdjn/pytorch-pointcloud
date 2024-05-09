import torch
from torch import Tensor, nn


def huber_loss(error: Tensor, delta: float = 1.0) -> Tensor:
    abs_error = torch.abs(error)
    quadratic = torch.clamp(abs_error, max=delta)
    linear = abs_error - quadratic
    loss = 0.5 * quadratic**2 + delta * linear
    return loss


class HuberLoss(nn.Module):
    """
    Huber loss function, also known as smooth L1 loss.
    It is less sensitive to outliers than the mean square error loss and in some cases prevents exploding gradients.

    Example:

    ```python
    criterion = HuberLoss(delta=1.0)

    pred = torch.randn(32, 6)
    target = torch.randn(32, 6)
    error = pred - target
    loss = criterion(error)
    ```

    """

    def __init__(self, delta: float = 1.0):
        super(HuberLoss, self).__init__()
        self.delta = delta

    def forward(self, error: Tensor) -> Tensor:
        return huber_loss(error, self.delta)
