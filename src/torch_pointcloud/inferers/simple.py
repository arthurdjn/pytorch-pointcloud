from typing import Any, Callable, Dict

from torch import Tensor

from .inferer import Inferer


class SimpleInferer(Inferer):
    """Direct call to the predictor on the whole scene.

    The lightest possible `Inferer`. Use when the model can consume the entire
    point cloud in one forward pass (object classification, small scenes).

    Example:
        ```python
        from torch_pointcloud.inferers import SimpleInferer

        inferer = SimpleInferer()
        logits = inferer(data, predictor=lambda d: model(d["pos"], d["pos"], d["batch"]))
        ```
    """

    def forward(
        self,
        data: Dict[str, Any],
        predictor: Callable[[Dict[str, Any]], Tensor],
    ) -> Tensor:
        return predictor(data)
