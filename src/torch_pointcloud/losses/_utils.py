r"""Numerical helpers shared by the detection losses."""

import torch
from torch import Tensor

_EPS = 1e-4


def _clamp_sigmoid(x: Tensor) -> Tensor:
    r"""Sigmoid clamped to $[\varepsilon, 1 - \varepsilon]$ so the focal $\log$ terms stay finite."""
    return torch.clamp(x.sigmoid(), min=_EPS, max=1 - _EPS)
