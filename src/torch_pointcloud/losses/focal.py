r"""Focal losses: the sigmoid focal loss of the anchor / point / query heads, the Gaussian focal loss of the heatmap heads
and the Poly-1 focal loss."""

from typing import Literal, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = ["GaussianFocalLoss", "Poly1FocalLoss", "SigmoidFocalLoss"]


def sigmoid_focal_loss(
    logits: Tensor,
    targets: Tensor,
    weights: Optional[Tensor] = None,
    *,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> Tensor:
    r"""Sigmoid focal loss between per-class logits and binary targets (no reduction).

    The classification term of :arxiv: [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002) (Lin et al., 2017), shared by the
    anchor, point and query heads: binary cross-entropy on the sigmoid of each logit, scaled by
    $\alpha_t (1 - p_t)^\gamma$ so confident predictions contribute little. Same math as
    `torchvision.ops.sigmoid_focal_loss`, plus an optional per-row weight.

    Args:
        logits: Per-class logits, shape $(\ldots, C)$.
        targets: Binary (one-hot) targets of the same shape.
        weights: Optional weight of every row, shape $(\ldots)$, broadcast over the class dimension.
        alpha: Positive/negative balance $\alpha$.
        gamma: Focusing exponent $\gamma$.

    Returns:
        The weighted per-element loss, shape $(\ldots, C)$.

    Shape:
        - logits: $(\ldots, C)$
        - targets: $(\ldots, C)$
        - weights: $(\ldots)$
        - output: $(\ldots, C)$

    Example:
        ```pycon
        >>> logits = torch.zeros(1, 2, 3)
        >>> targets = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
        >>> sigmoid_focal_loss(logits, targets, torch.ones(1, 2)).shape
        torch.Size([1, 2, 3])

        ```
    """
    prob = logits.sigmoid()
    alpha_weight = targets * alpha + (1.0 - targets) * (1.0 - alpha)
    pt = targets * (1.0 - prob) + (1.0 - targets) * prob
    focal_weight = alpha_weight * pt.pow(gamma)
    bce = logits.clamp(min=0) - logits * targets + torch.log1p(torch.exp(-logits.abs()))
    loss = focal_weight * bce
    if weights is not None:
        loss = loss * weights.unsqueeze(-1)
    return loss


def gaussian_focal_loss(
    pred: Tensor,
    target: Tensor,
    *,
    alpha: float = 2.0,
    gamma: float = 4.0,
    eps: float = 1e-4,
) -> Tensor:
    r"""Penalty-reduced focal loss over a Gaussian heatmap, normalized by the number of peaks.

    The heatmap objective of :arxiv: [Objects as Points](https://arxiv.org/abs/1904.07850) (Zhou et al., 2019), used by the
    center-based detection heads (CenterPoint, VoxelNeXt, TransFusion). Cells whose target is exactly $1$
    are positives with the $(1 - p)^\alpha \log p$ term; every other cell is a soft negative whose
    $p^\alpha \log(1 - p)$ term is down-weighted by $(1 - y)^\gamma$, so cells near a peak barely count.
    The sum is divided by the number of positives (or left as is when there is none).

    Args:
        pred: Predicted probabilities (post-sigmoid) of any shape; clamped to $[\varepsilon, 1 - \varepsilon]$.
        target: Gaussian heatmap target of the same shape, values in $[0, 1]$.
        alpha: Focusing exponent on the prediction.
        gamma: Down-weighting exponent on the soft negatives.
        eps: Clamp keeping the logarithms finite.

    Returns:
        The scalar loss.

    Shape:
        - pred: $(\ldots)$
        - target: $(\ldots)$
        - output: scalar

    Example:
        ```pycon
        >>> target = torch.tensor([[1.0, 0.5], [0.0, 0.0]])
        >>> gaussian_focal_loss(torch.full((2, 2), 0.5), target).shape
        torch.Size([])

        ```
    """
    pred = pred.clamp(min=eps, max=1 - eps)
    pos = target.eq(1).to(pred.dtype)
    neg_weights = (1 - target).pow(gamma)
    pos_loss = -(pred.log() * (1 - pred).pow(alpha) * pos).sum()
    neg_loss = -((1 - pred).log() * pred.pow(alpha) * neg_weights * (1 - pos)).sum()
    return (pos_loss + neg_loss) / pos.sum().clamp(min=1.0)


def one_hot_foreground(labels: Tensor, num_classes: int) -> Tensor:
    r"""One-hot encode per-row class labels, dropping the background column.

    Ignored ($-1$) and background ($0$) rows map to an all-zero row; a foreground row with label
    $\ell \ge 1$ maps to a one-hot row on class $\ell - 1$. The target format of
    [`sigmoid_focal_loss`][torch_pointcloud.losses.focal.sigmoid_focal_loss] for the anchor and
    point heads.

    Args:
        labels: Class labels ($-1$ ignore, $0$ background, $\ge 1$ foreground), shape $(\ldots)$.
        num_classes: Number of foreground classes $C$.

    Returns:
        One-hot foreground targets, shape $(\ldots, C)$.

    Shape:
        - labels: $(\ldots)$
        - output: $(\ldots, C)$

    Example:
        ```pycon
        >>> one_hot_foreground(torch.tensor([[2, 0, -1]]), 3).tolist()
        [[[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]

        ```
    """
    cared = labels >= 0
    cls_targets = labels * cared.to(labels.dtype)
    one_hot = torch.zeros(*labels.shape, num_classes + 1, dtype=torch.float32, device=labels.device)
    one_hot.scatter_(-1, cls_targets.unsqueeze(-1), 1.0)
    return one_hot[..., 1:]


class SigmoidFocalLoss(nn.Module):
    r"""Module form of [`sigmoid_focal_loss`][torch_pointcloud.losses.focal.sigmoid_focal_loss].

    Args:
        alpha: Positive/negative balance $\alpha$.
        gamma: Focusing exponent $\gamma$.
        reduction: `"none"` returns the per-element loss, `"sum"` its sum, `"mean"` its mean.

    Example:
        ```python
        criterion = SigmoidFocalLoss(reduction="sum")
        criterion(torch.zeros(4, 3), torch.zeros(4, 3)).shape  # torch.Size([])
        ```
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: Literal["none", "sum", "mean"] = "mean",
    ) -> None:
        super().__init__()
        if reduction not in ("none", "sum", "mean"):
            raise ValueError(f"`reduction` must be 'none', 'sum' or 'mean', got {reduction!r}.")

        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: Tensor, targets: Tensor, weights: Optional[Tensor] = None) -> Tensor:
        r"""Compute the focal loss from logits $(\ldots, C)$, binary targets $(\ldots, C)$ and row weights $(\ldots)$."""
        loss = sigmoid_focal_loss(logits, targets, weights, alpha=self.alpha, gamma=self.gamma)
        if self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "mean":
            loss = loss.mean()
        return loss

    def extra_repr(self) -> str:
        return f"alpha={self.alpha}, gamma={self.gamma}, reduction={self.reduction!r}"


class GaussianFocalLoss(nn.Module):
    r"""Module form of [`gaussian_focal_loss`][torch_pointcloud.losses.focal.gaussian_focal_loss].

    Takes heatmap **logits** and applies the sigmoid itself.

    Args:
        alpha: Focusing exponent on the prediction.
        gamma: Down-weighting exponent on the soft negatives.

    Example:
        ```python
        criterion = GaussianFocalLoss()
        criterion(torch.zeros(1, 2, 4, 4), torch.zeros(1, 2, 4, 4)).shape  # torch.Size([])
        ```
    """

    def __init__(self, alpha: float = 2.0, gamma: float = 4.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        r"""Compute the loss from heatmap logits and the Gaussian target of the same shape."""
        return gaussian_focal_loss(logits.sigmoid(), target, alpha=self.alpha, gamma=self.gamma)

    def extra_repr(self) -> str:
        return f"alpha={self.alpha}, gamma={self.gamma}"


def poly1_focal_loss(
    logits: Tensor,
    targets: Tensor,
    *,
    epsilon: float = 1.0,
    alpha: Optional[float] = 0.25,
    gamma: float = 2.0,
) -> Tensor:
    r"""Poly-1 focal loss between per-class logits and binary targets (no reduction).

    The focal loss plus its first Poly-1 correction term of :arxiv: [PolyLoss: A Polynomial Expansion Perspective of Classification Loss Functions](https://arxiv.org/abs/2204.12511) (Leng et al., 2022),
    with $p$ the sigmoid of the logit,
    $p_t = p$ on the positive entries and $1 - p$ elsewhere,

    $$
    \ell = \alpha_t\, (1 - p_t)^\gamma\, \mathrm{BCE}(\text{logit}, \text{target}) + \epsilon\, (1 - p_t)^{\gamma + 1}
    $$

    Args:
        logits: Per-class logits, shape $(\ldots, C)$.
        targets: Binary (one-hot) targets of the same shape.
        epsilon: Weight $\epsilon$ of the Poly-1 term.
        alpha: Positive/negative balance $\alpha$; `None` disables it.
        gamma: Focusing exponent $\gamma$.

    Returns:
        The per-element loss, shape $(\ldots, C)$.

    Shape:
        - logits: $(\ldots, C)$
        - targets: $(\ldots, C)$
        - output: $(\ldots, C)$

    Example:
        ```pycon
        >>> logits = torch.zeros(2, 3)
        >>> targets = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        >>> poly1_focal_loss(logits, targets).shape
        torch.Size([2, 3])

        ```
    """
    p = logits.sigmoid()
    p_t = p * targets + (1 - p) * (1 - targets)
    focal = F.binary_cross_entropy_with_logits(logits, targets, reduction="none") * (1 - p_t) ** gamma
    if alpha is not None:
        focal = focal * (alpha * targets + (1 - alpha) * (1 - targets))
    return focal + epsilon * (1 - p_t) ** (gamma + 1)


class Poly1FocalLoss(nn.Module):
    r"""Module form of [`poly1_focal_loss`][torch_pointcloud.losses.focal.poly1_focal_loss], taking class indices.

    The targets are either class indices $(\ldots)$, one-hot encoded over the $C$ logits, or binary targets of the
    logits' shape. `reduction="mean"` averages over every element of the $(\ldots, C)$ loss.

    Args:
        epsilon: Weight $\epsilon$ of the Poly-1 term.
        alpha: Positive/negative balance $\alpha$; `None` disables it.
        gamma: Focusing exponent $\gamma$.
        reduction: `"none"` returns the per-element loss, `"sum"` its sum, `"mean"` its mean.

    Example:
        ```python
        criterion = Poly1FocalLoss()
        criterion(torch.zeros(4, 3), torch.tensor([0, 1, 2, 0])).shape  # torch.Size([])
        ```
    """

    def __init__(
        self,
        epsilon: float = 1.0,
        alpha: Optional[float] = 0.25,
        gamma: float = 2.0,
        reduction: Literal["none", "sum", "mean"] = "mean",
    ) -> None:
        super().__init__()
        if reduction not in ("none", "sum", "mean"):
            raise ValueError(f"`reduction` must be 'none', 'sum' or 'mean', got {reduction!r}.")

        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        r"""Compute the loss from logits $(\ldots, C)$ and class indices $(\ldots)$ or binary targets $(\ldots, C)$."""
        if targets.dim() == logits.dim() - 1:
            targets = F.one_hot(targets.long(), num_classes=logits.shape[-1])

        loss = poly1_focal_loss(
            logits,
            targets.to(logits.dtype),
            epsilon=self.epsilon,
            alpha=self.alpha,
            gamma=self.gamma,
        )

        if self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "mean":
            loss = loss.mean()
        return loss

    def extra_repr(self) -> str:
        return f"epsilon={self.epsilon}, alpha={self.alpha}, gamma={self.gamma}, reduction={self.reduction!r}"
