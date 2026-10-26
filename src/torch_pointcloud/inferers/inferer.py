"""Abstract base classes for test-time inference strategies, one per task family."""

from abc import ABCMeta, abstractmethod
from typing import Any, Callable, Dict, Generic, TypeVar

import torch
from torch import Tensor

from torch_pointcloud.utils.types import Detection3D

OutputT = TypeVar("OutputT")


class Inferer(Generic[OutputT], metaclass=ABCMeta):
    r"""Base class for test-time inference strategies.

    An `Inferer` decouples *how* a model is run at test time from the model
    itself. The model only knows how to map a batch of points to an output;
    the `Inferer` decides whether that happens in a single forward pass, over
    cropped windows, tiled blocks, or repeated under augmentation, and how the
    partial outputs are merged back into one. This keeps evaluation code
    identical regardless of scene size or protocol.

    How partial outputs merge depends on the task, so the strategies come in
    families: a `SegmentationInferer` merges per-point scores, a
    `DetectionInferer` merges sets of boxes. `SimpleInferer` merges nothing
    and serves every task.

    Subclasses implement `forward`; `__call__` delegates to it, mirroring
    `torch.nn.Module`. To use an inferer, call the instance directly:

    ```python
    import torch

    import torch_pointcloud as tp
    from torch_pointcloud.inferers import SimpleInferer

    model = tp.create_model("pointnet.s3dis-area5").eval()
    data = {"pos": torch.rand(4096, 3) * 10, "x": torch.rand(4096, 9), "batch": torch.zeros(4096, dtype=torch.long)}

    inferer = SimpleInferer()
    logits = inferer(data, predictor=lambda d: model(d["x"], d["pos"], d["batch"]))
    ```

    `data` is a packed-batch dict (at minimum containing position and batch indices).
    `predictor` is any callable taking such a dict and returning the model's output.
    Inferers keep nothing of the scene, so one instance can be reused across scenes
    and wrapped by another inferer of the same family.

    Inferers that draw random numbers take a `seed`. `None` follows the global generator,
    like the transforms do, so one `torch.manual_seed` (or `seed_everything`) seeds the
    whole evaluation. An int gives the inferer its own stream, offset by the number of
    calls the instance has made: successive calls (e.g. `TTASegmentationInferer` votes) draw
    different numbers, and a fresh instance replays the same sequence.

    To add a custom strategy, subclass the family of its output and implement `forward`:

    ```python
    from torch_pointcloud.inferers import SegmentationInferer

    class MyInferer(SegmentationInferer):
        def forward(self, data, predictor):
            return predictor(data)
    ```
    """

    @abstractmethod
    def forward(
        self,
        data: Dict[str, Any],
        predictor: Callable[[Dict[str, Any]], OutputT],
    ) -> OutputT:
        r"""Run the inference strategy.

        Args:
            data: Packed-batch dict. Must contain `pos` and `batch` keys (the
                exact names are configurable on subclasses that expose
                `pos_key` / `batch_key`).
            predictor: Callable taking a packed dict and returning the model's output.

        Returns:
            The merged output, of the same type as the predictor's.
        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")

    @torch.no_grad()
    def __call__(
        self,
        data: Dict[str, Any],
        predictor: Callable[[Dict[str, Any]], OutputT],
    ) -> OutputT:
        return self.forward(data, predictor)


class SegmentationInferer(Inferer[Tensor]):
    r"""Base class for inference strategies merging per-point scores.

    The predictor returns per-point scores of shape $(N, C_\text{out})$ and the inferer returns one row per
    point of the input. Rows are indexed by point, not by position, so predictions of the same point from
    overlapping fragments or rotated views line up and are merged by weighted averaging.

    The `softmax` parameter controls whether partial predictions are converted to softmax
    probabilities before aggregation, and `aggregate` how they are combined:

    | Inferer                              | `softmax` | Aggregated quantity                                             |
    | ------------------------------------ | --------- | --------------------------------------------------------------- |
    | `SlidingWindowSegmentationInferer`   | `False`   | raw logits per block; always probabilities (`"max"` / `"vote"`) |
    | `KNNWindowSegmentationInferer`       | `False`   | raw logits (`"mean"`); always probabilities (`"ema"`)           |
    | `VoxelPartitionSegmentationInferer`  | `False`   | raw logits per pass                                             |
    | `PotentialSphereSegmentationInferer` | none      | always an EMA of softmax probabilities                          |
    | `TTASegmentationInferer`             | `False`   | base output as-is                                               |
    | `PartRefinementInferer`              | none      | one-hot refined labels of the base output's argmax              |

    When the input scene is empty ($N = 0$), inferers that never call the predictor
    return a $(0, 0)$ tensor (the channel count cannot be inferred without a predictor
    call); `TTASegmentationInferer` returns whatever its base inferer produces for the
    empty input.
    """


class DetectionInferer(Inferer[Detection3D]):
    r"""Base class for inference strategies merging sets of 3D boxes.

    The predictor returns the raw detections of a model (its `decode` output, see `Detection3D`) in the
    coordinates the model was fed. The inferer maps every partial set back to the coordinates of the input
    scene and concatenates them, with per-box `batch` indices of the input scenes. It never filters: the
    duplicates that overlapping views or tiles produce are left to the evaluation postprocess (score
    threshold and `nms3d`), like the boxes of a single pass.
    """
