from abc import ABCMeta, abstractmethod
from typing import Any, Callable, Dict

import torch
from torch import Tensor


class Inferer(metaclass=ABCMeta):
    r"""Base class for test-time inference strategies.

    An `Inferer` decouples *how* a model is run at test time from the model
    itself. The model only knows how to map a batch of points to per-point
    logits; the `Inferer` decides whether that happens in a single forward
    pass, over cropped windows, tiled blocks, or repeated under augmentation,
    and how partial predictions are stitched back into one per-point output.
    This keeps evaluation code identical regardless of scene size or protocol.

    Subclasses implement `forward`; `__call__` delegates to it, mirroring
    `torch.nn.Module`. To use an inferer, call the instance directly:

    ```{.python notest}
    inferer = SomeInferer(...)
    logits = inferer(data, predictor=lambda d: model(d["pos"], d["pos"], d["batch"]))
    ```

    `data` is a packed-batch dict (at minimum containing position and batch indices).
    `predictor` is any callable taking such a dict and returning per-point logits of shape
    $(N, C_\text{out})$. Inferers are stateless with respect to the scene, so
    one instance can be reused across scenes and wrapped by another inferer.

    Every concrete inferer exposes a knob controlling whether partial predictions are
    converted to softmax probabilities before aggregation. The defaults differ:

    | Inferer                  | Parameter     | Default | Aggregated quantity                                          |
    | ------------------------ | ------------- | ------- | ------------------------------------------------------------ |
    | `SimpleInferer`          | `softmax`     | `False` | predictor output as-is                                       |
    | `SlidingWindowInferer`   | `softmax`     | `True`  | softmax probabilities per block (`"max"` / `"vote"` always)  |
    | `KNNWindowInferer`       | `softmax`     | `False` | raw logits (`"weighted_mean"`); always probabilities (`"ema"`) |
    | `VoxelPartitionInferer`  | `softmax`     | `False` | raw logits per pass                                          |
    | `PotentialSphereInferer` | none          |         | always an EMA of softmax probabilities                        |
    | `TTAInferer`             | `ema_softmax` | `True`  | base output as-is (`"mean"`); softmax of base output (`"ema"`) |
    | `PartRefinementInferer`  | none          |         | one-hot refined labels of the base output's argmax           |

    When the input scene is empty ($N = 0$), inferers that never call the predictor
    return a $(0, 0)$ tensor (the channel count cannot be inferred without a predictor
    call); `SimpleInferer` and `TTAInferer` return whatever the predictor / base
    inferer produces for the empty input.

    To add a custom strategy, subclass `Inferer` and implement `forward`:

    ```python
    from torch_pointcloud.inferers import Inferer

    class MyInferer(Inferer):
        def forward(self, data, predictor):
            return predictor(data)
    ```
    """

    @abstractmethod
    def forward(
        self,
        data: Dict[str, Any],
        predictor: Callable[[Dict[str, Any]], Tensor],
    ) -> Tensor:
        r"""Run the inference strategy.

        Args:
            data: Packed-batch dict. Must contain `pos` and `batch` keys (the
                exact names are configurable on subclasses that expose
                `pos_key` / `batch_key`).
            predictor: Callable taking a packed dict and returning per-point
                logits of shape $(N, C_\text{out})$.

        Returns:
            Per-point output tensor of shape $(N, C_\text{out})$.
        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")

    @torch.no_grad()
    def __call__(
        self,
        data: Dict[str, Any],
        predictor: Callable[[Dict[str, Any]], Tensor],
    ) -> Tensor:
        return self.forward(data, predictor)
