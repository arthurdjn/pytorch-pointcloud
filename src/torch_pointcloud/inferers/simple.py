"""Single forward-pass inference over the whole scene."""

from typing import Any, Callable, Dict, Optional

import torch
from torch import Tensor

from torch_pointcloud.utils.data import DataKeys

from ._utils import apply_transform
from .inferer import Inferer


class SimpleInferer(Inferer):
    r"""Direct call to the predictor on the whole scene.

    The lightest possible `Inferer`. Use when the model can consume the entire
    point cloud in one forward pass (object classification, small scenes). An
    optional `transform` prepares the scene for the predictor (voxelization, grid
    coordinates); when it changes the row count, the map it records under
    `inverse_key` gathers the predictions back to the input points, so a
    `TTAInferer` over this inferer voxelizes every augmented view anew.

    Args:
        softmax: If `True`, softmax the predictor output over the last dim before
            returning it. The default returns the predictor output unchanged.
        transform: Optional callable applied to `data` before the predictor. If it
            changes the row count (voxelize, pad, ...) it must record a
            source-to-predictor index map under `inverse_key`.
        pos_key: Dict key for the position tensor, whose row count is the number of
            input points. Read only when `transform` is set.
        inverse_key: Dict key under which a row-altering `transform` records a
            source-to-predictor long index map of shape $(N,)$ with values in
            $[0, N_\text{predictor})$. Any value already at this key is dropped before
            `transform` runs, and the map is popped before the predictor is called.
            Leave `None` when the transform preserves the row count.

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

        One voxelization of the scene, predictions read back at every input point:

        ```python
        import torch
        import torch_pointcloud.transforms as T
        from torch_pointcloud.inferers import SimpleInferer

        data = {"pos": torch.rand(4096, 3) * 10, "batch": torch.zeros(4096, dtype=torch.long)}
        voxelize = T.Voxelize(
            pos_key="pos",
            pos_reduce="first",
            size=0.5,
            keys="batch",
            dst_pos_grid_key="pos_grid",
            dst_inverse_key="inverse",
        )
        inferer = SimpleInferer(transform=voxelize, inverse_key="inverse")
        out = inferer(data, predictor=lambda d: d["pos_grid"].float())  # (4096, 3), the voxel of each point
        ```
    """

    def __init__(
        self,
        softmax: bool = False,
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        pos_key: str = DataKeys.POS,
        inverse_key: Optional[str] = None,
    ) -> None:
        self.softmax = softmax
        self.transform = transform
        self.pos_key = pos_key
        self.inverse_key = inverse_key

    def forward(
        self,
        data: Dict[str, Any],
        predictor: Callable[[Dict[str, Any]], Tensor],
    ) -> Tensor:
        inverse_map = None
        if self.transform is not None:
            data, inverse_map = apply_transform(
                data,
                self.transform,
                pos_key=self.pos_key,
                inverse_key=self.inverse_key,
            )

        out = predictor(data)
        if inverse_map is not None:
            out = out[inverse_map]
        if self.softmax:
            out = torch.softmax(out, dim=-1)
        return out
