r"""Prediction reducers for ensemble / TTA / voting workflows.

Pure callables on a sequence of per-point prediction tensors. They aggregate
without knowing how the predictions were produced (live inference, saved files,
multi-model fold ensemble, multi-seed voting).

Two reductions:

- **`mean_ensemble`** / **`MeanEnsemble`**: stack outputs along a new leading
  dim and average. Matches nnUNet's softmax fold-ensembling and MONAI's
  `MeanEnsemble`.
- **`vote_ensemble`** / **`VoteEnsemble`**: argmax each output, one-hot, sum.
  The result's `argmax` is the majority-vote label. Matches MONAI's
  `VoteEnsemble`.

Both functional and class forms ship side-by-side, mirroring
`transforms.functional` ↔ `transforms.transforms`. Call the function inline
when the choice is fixed; instantiate the class when the reducer is selected
by config.
"""

from abc import ABCMeta, abstractmethod
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor


def _stack_outputs(outputs: Sequence[Tensor]) -> Tensor:
    if len(outputs) == 0:
        raise ValueError("`outputs` must contain at least one tensor.")
    return torch.stack(list(outputs), dim=0)


def mean_ensemble(outputs: Sequence[Tensor]) -> Tensor:
    r"""Per-point mean across a list of prediction tensors.

    Args:
        outputs: Sequence of $(N, C)$ tensors (typically softmax probabilities
            or logits). All must have the same shape.

    Returns:
        Mean along the stacking dim, shape $(N, C)$.
    """
    return _stack_outputs(outputs).mean(dim=0)


def vote_ensemble(outputs: Sequence[Tensor], num_classes: int) -> Tensor:
    r"""Per-point majority-vote counts across a list of prediction tensors.

    Each output is argmax'd along its last dim, one-hot encoded with
    `num_classes` channels, and summed across the ensemble. Take `argmax` of
    the result to get the majority-vote labels per point; the raw counts are
    useful for tie inspection.

    Args:
        outputs: Sequence of $(N, C)$ tensors. The argmax of each is taken, so
            either logits or probabilities work.
        num_classes: Channel count $C$ used for the one-hot encoding. Must be
            at least `max(argmax(output)) + 1`; pass the model's `num_classes`.

    Returns:
        Per-point class-vote counts, shape $(N, \text{num\_classes})$, dtype
        matching the inputs.
    """
    stacked = _stack_outputs(outputs)
    preds = stacked.argmax(dim=-1)
    one_hot = F.one_hot(preds, num_classes=num_classes).to(stacked.dtype)
    return one_hot.sum(dim=0)


class Ensemble(metaclass=ABCMeta):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, outputs: Sequence[Tensor]) -> Tensor:
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")

    def __call__(self, outputs: Sequence[Tensor]) -> Tensor:
        return self.forward(outputs)


class MeanEnsemble(Ensemble):
    r"""Class form of `mean_ensemble`.

    Stateless wrapper for use when the reducer is chosen by config or stored
    as part of a pipeline.

    Example:
        ```python
        from torch_pointcloud.utils.ensemble import MeanEnsemble

        reducer = MeanEnsemble()
        probs = reducer(outputs)
        ```
    """

    def forward(self, outputs: Sequence[Tensor]) -> Tensor:
        return mean_ensemble(outputs)


class VoteEnsemble(Ensemble):
    r"""Class form of `vote_ensemble`.

    Args:
        num_classes: Channel count for the one-hot encoding. Stored on the
            instance so `forward` matches the `Callable[[Sequence[Tensor]],
            Tensor]` signature of `MeanEnsemble`.

    Example:
        ```python
        from torch_pointcloud.utils.ensemble import VoteEnsemble

        reducer = VoteEnsemble(num_classes=13)
        votes = reducer(outputs)
        labels = votes.argmax(dim=-1)
        ```
    """

    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes

    def forward(self, outputs: Sequence[Tensor]) -> Tensor:
        return vote_ensemble(outputs, self.num_classes)
