"""Single forward-pass inference over the whole scene."""

from typing import Any, Callable, Dict, List, Optional

import torch
from torch import Tensor

from torch_pointcloud.utils.data import DataKeys, collate

from ._utils import apply_transform, check_batch_alignment, index_select_dict
from .inferer import Inferer


class SimpleInferer(Inferer):
    r"""Direct call to the predictor on the whole scene.

    The lightest possible `Inferer`. Use when the model can consume the entire
    point cloud in one forward pass (object classification, small scenes).

    An optional `transform` prepares each point cloud of the batch for the predictor (voxelization, grid coordinates):
    it runs on every cloud alone, the results are collated into one predictor call, and when it changes the row count
    the map it records under `inverse_key` reads the predictions back at every input point. A `TTAInferer` over this
    inferer thus voxelizes every augmented view anew. With a `transform`, an empty batch ($N = 0$) returns a $(0, 0)$
    tensor: the predictor is never called, so the channel count cannot be inferred.

    Args:
        softmax: If `True`, softmax the predictor output over the last dim before
            returning it. The default returns the predictor output unchanged.
        transform: Optional per-cloud callable applied before the predictor. If it changes the row count (voxelize,
            pad, ...) it must record a source-to-predictor index map under `inverse_key`. Each cloud receives the rows
            of the per-point tensors (leading dimension $N$); any other entry, a scene-level `label` of shape $(B,)$
            included, reaches every cloud whole and is concatenated by `collate`.
        pos_key: Dict key for the position tensor. Read only when `transform` is set.
        batch_key: Dict key for the per-point batch index. Read only when `transform` is set.
        inverse_key: Dict key under which a row-altering `transform` records a source-to-predictor long index map of
            shape $(N_\text{cloud},)$ with values in $[0, N_\text{predictor})$. Any value already at this key is
            dropped before `transform` runs, and the map is popped before the predictor is called. Leave `None` when
            the transform preserves the row count.

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

        One voxelization per cloud, the predictions read back at every input point:

        ```python
        import torch

        import torch_pointcloud.transforms as T
        from torch_pointcloud.inferers import SimpleInferer

        data = {"pos": torch.rand(4096, 3) * 10, "batch": torch.arange(2).repeat_interleave(2048)}
        voxelize = T.Voxelize(
            pos_key="pos",
            pos_reduce="first",
            size=0.5,
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
        batch_key: str = DataKeys.BATCH,
        inverse_key: Optional[str] = None,
    ) -> None:
        self.softmax = softmax
        self.transform = transform
        self.pos_key = pos_key
        self.batch_key = batch_key
        self.inverse_key = inverse_key

    def forward(
        self,
        data: Dict[str, Any],
        predictor: Callable[[Dict[str, Any]], Tensor],
    ) -> Tensor:
        if self.transform is None:
            out = predictor(data)
        else:
            if self.pos_key not in data:
                raise KeyError(f"`data` is missing the required key {self.pos_key!r}.")

            if self.batch_key not in data:
                raise KeyError(f"`data` is missing the required key {self.batch_key!r}.")

            pos = data[self.pos_key]
            batch = data[self.batch_key]
            check_batch_alignment(pos, batch, self.pos_key, self.batch_key)
            n = pos.size(0)
            if n == 0:
                return pos.new_zeros((0, 0))

            indices: List[Tensor] = []
            samples: List[Dict[str, Any]] = []
            inverse_maps: List[Optional[Tensor]] = []
            for b in torch.unique(batch).tolist():
                idx = torch.where(batch == b)[0]
                sample, inverse_map = apply_transform(
                    index_select_dict(data, idx, n),
                    self.transform,
                    pos_key=self.pos_key,
                    inverse_key=self.inverse_key,
                )
                indices.append(idx)
                samples.append(sample)
                inverse_maps.append(inverse_map)

            packed = collate(samples, batch_from=self.pos_key, batch_key=self.batch_key)
            logits = predictor(packed)
            sizes = [int(sample[self.pos_key].size(0)) for sample in samples]
            parts = [
                part if inverse_map is None else part[inverse_map.to(part.device)]
                for part, inverse_map in zip(logits.split(sizes), inverse_maps)
            ]
            out = logits.new_empty((n, *logits.shape[1:]))
            out[torch.cat(indices).to(out.device)] = torch.cat(parts)
        if self.softmax:
            out = torch.softmax(out, dim=-1)
        return out
