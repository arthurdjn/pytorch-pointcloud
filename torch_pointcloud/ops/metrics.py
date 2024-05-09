from typing import Any, List, Literal, Tuple, Union

import torch
import torch.autograd
from torch import Tensor

from torch_pointcloud import _C  # type: ignore[attr-defined]
from torch_pointcloud.utils import default_vector


class SidedDistance(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, pc1: Tensor, pc2: Tensor, lengths1: Tensor, lengths2: Tensor) -> Tuple[Tensor, Tensor]:
        dists, idxs = _C.sided_distance(pc1, pc2, lengths1, lengths2)
        ctx.save_for_backward(pc1, pc2, idxs, lengths1, lengths2)
        ctx.mark_non_differentiable(idxs)
        return dists, idxs

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Tensor) -> Tuple[Tensor, Tensor, None, None]:
        grad_dists, grad_idxs, *_ = grad_outputs
        pc1, pc2, idxs, lengths1, lengths2 = ctx.saved_tensors
        grad_pc1, grad_pc2 = _C.sided_distance_backward(grad_dists, pc1, pc2, idxs, lengths1, lengths2)
        return grad_pc1, grad_pc2, None, None


def sided_distance(
    pc1: Tensor,
    pc2: Tensor,
    *,
    lengths1: Union[int, Tensor, None] = None,
    lengths2: Union[int, Tensor, None] = None,
) -> Tuple[Tensor, Tensor]:
    B, N1, _ = pc1.shape
    B, N2, _ = pc2.shape

    lengths1 = default_vector(lengths1, size=B, default_value=N1).long()
    lengths2 = default_vector(lengths2, size=B, default_value=N2).long()
    lengths1 = lengths1.clamp(0, N1).to(pc1.device)
    lengths2 = lengths2.clamp(0, N2).to(pc2.device)

    return SidedDistance.apply(pc1, pc2, lengths1, lengths2)  # type: ignore[no-untyped-call]


def chamfer_distance(
    pc1: Tensor,
    pc2: Tensor,
    *,
    lengths1: Union[int, List[int], Tensor, None] = None,
    lengths2: Union[int, List[int], Tensor, None] = None,
    reduction: Literal["mean", "sum"] = "mean",
) -> Tensor:
    assert reduction in ["mean", "sum"], f"Reduction must be 'mean' or 'sum', got {reduction}."
    B, N1, _ = pc1.shape
    B, N2, _ = pc2.shape

    lengths1 = default_vector(pc1, size=B, default_value=N1).long()
    lengths2 = default_vector(pc2, size=B, default_value=N2).long()
    lengths1 = lengths1.clamp(0, N1).to(pc1.device)
    lengths2 = lengths2.clamp(0, N2).to(pc2.device)

    dists1, _ = sided_distance(pc1, pc2, lengths1=lengths1, lengths2=lengths2)
    dists2, _ = sided_distance(pc2, pc1, lengths1=lengths2, lengths2=lengths1)
    reduce_fn = torch.mean if reduction == "mean" else torch.sum
    return torch.stack([reduce_fn(dists1[i, : lengths1[i]]) + reduce_fn(dists2[i, : lengths2[i]]) for i in range(B)])
