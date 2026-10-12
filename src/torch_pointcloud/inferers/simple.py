"""Single forward-pass inference over the whole scene."""

from typing import Any, Callable, Dict

import torch
from torch import Tensor

from .inferer import Inferer


class SimpleInferer(Inferer):
    """Direct call to the predictor on the whole scene.

    The lightest possible `Inferer`. Use when the model can consume the entire
    point cloud in one forward pass (object classification, small scenes).

    Args:
        softmax: If `True`, softmax the predictor output over the last dim before
            returning it. The default returns the predictor output unchanged.

    Example:
        ```python
        import torch

        import torch_pointcloud as tp
        from torch_pointcloud.inferers import SimpleInferer

        model = tp.create_model("pointnet.s3dis-area5").eval()
        data = {
            "pos": torch.rand(4096, 3) * 10,
            "x": torch.rand(4096, 9),
            "batch": torch.zeros(4096, dtype=torch.long),
        }

        inferer = SimpleInferer()
        logits = inferer(data, predictor=lambda d: model(d["x"], d["pos"], d["batch"]))
        ```
    """

    def __init__(self, softmax: bool = False) -> None:
        self.softmax = softmax

    def forward(
        self,
        data: Dict[str, Any],
        predictor: Callable[[Dict[str, Any]], Tensor],
    ) -> Tensor:
        out = predictor(data)
        if self.softmax:
            out = torch.softmax(out, dim=-1)
        return out
