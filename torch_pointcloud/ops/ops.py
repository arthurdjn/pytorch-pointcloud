from typing import Any, Optional, Sequence, Tuple, Union

import torch
import torch.autograd
from torch import Tensor

from torch_pointcloud import _C  # type: ignore[attr-defined]
from torch_pointcloud.utils import default_vector


class ThreeInterpolate(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        points: Tensor,
        idxs: Tensor,
        weights: Tensor,
        lengths: Tensor,
        out_lengths: Tensor,
    ) -> Tensor:
        ctx.save_for_backward(idxs, weights, lengths, out_lengths)
        ctx.num_points = points.size(1)
        out = _C.three_interpolate(points, idxs, weights, lengths, out_lengths)
        return out

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Tensor) -> Tuple[Tensor, None, None, None, None]:
        # There is a typing error if not using *args as arguments in the function signature:
        # -> Signature of "backward" incompatible with supertype "_SingleLevelFunction" mypy[override]
        # Using *args is a workaround to avoid the error, and it is expected to be fixed in future versions,
        # to have a cleaner function signature and easier to understand.
        grad_out, *_ = grad_outputs
        grad_out = grad_out.contiguous()
        idxs, weights, lengths, out_lengths = ctx.saved_tensors
        M = ctx.num_points
        grad_points = _C.three_interpolate_backward(grad_out, idxs, weights, lengths, out_lengths, M)
        return grad_points, None, None, None, None


class KInterpolate(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        points: Tensor,
        idxs: Tensor,
        weights: Tensor,
        K: Tensor,
        lengths: Tensor,
        out_lengths: Tensor,
    ) -> Tensor:
        ctx.save_for_backward(idxs, weights, K, lengths, out_lengths)
        ctx.num_points = points.size(1)
        out = _C.k_interpolate(points, idxs, weights, K, lengths, out_lengths)
        return out

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Tensor) -> Tuple[Tensor, None, None, None, None, None]:
        # There is a typing error if not using *args as arguments in the function signature:
        # -> Signature of "backward" incompatible with supertype "_SingleLevelFunction" mypy[override]
        # Using *args is a workaround to avoid the error, and it is expected to be fixed in future versions,
        # to have a cleaner function signature and easier to understand.
        grad_out, *_ = grad_outputs
        grad_out = grad_out.contiguous()
        idxs, weights, K, lengths, out_lengths = ctx.saved_tensors
        M = ctx.num_points
        grad_points = _C.k_interpolate_backward(grad_out, idxs, weights, K, lengths, out_lengths, M)
        return grad_points, None, None, None, None, None


def k_interpolate(
    points: Tensor,
    idxs: Tensor,
    weights: Tensor,
    k: Union[int, Sequence[int], Tensor] = 3,
    lengths: Optional[Union[int, Sequence[int], Tensor]] = None,
    out_lengths: Optional[Union[int, Sequence[int], Tensor]] = None,
) -> Tensor:
    B, M, _ = points.shape
    N = idxs.size(1)

    # Convert the input parameters to as 1D tensors os size B
    lengths = default_vector(lengths, size=B, default_value=M).long()  # (B,)
    out_lengths = default_vector(out_lengths, size=B, default_value=N).long()  # (B,)
    k = default_vector(k, size=B, default_value=3).long()  # (B,)

    # Make sure the values are within the correct range
    lengths = lengths.clamp(0, M).contiguous().to(points.device)
    out_lengths = out_lengths.clamp(0, N).contiguous().to(points.device)
    k = k.clamp(0, N).contiguous().to(points.device)

    return KInterpolate.apply(points, idxs, weights, k, lengths, out_lengths)


def three_interpolate(
    points: Tensor,
    idxs: Tensor,
    weights: Tensor,
    lengths: Optional[Union[int, Sequence[int], Tensor]] = None,
    out_lengths: Optional[Union[int, Sequence[int], Tensor]] = None,
) -> Tensor:
    B, M, _ = points.shape
    N = idxs.size(1)

    # Convert the input parameters to as 1D tensors os size B
    lengths = default_vector(lengths, size=B, default_value=M).long()  # (B,)
    out_lengths = default_vector(out_lengths, size=B, default_value=N).long()  # (B,)

    # Make sure the values are within the correct range
    lengths = lengths.clamp(0, M).contiguous().to(points.device)
    out_lengths = out_lengths.clamp(0, N).contiguous().to(points.device)

    return ThreeInterpolate.apply(points, idxs, weights, lengths, out_lengths)  # type: ignore[no-untyped-call]
    # return k_interpolate(points, idxs, weights, k=3, lengths=lengths, out_lengths=out_lengths)


def knn(
    pc1: Tensor,
    pc2: Tensor,
    k: Optional[Union[int, Sequence[int], Tensor]] = 3,
    lengths1: Optional[Union[int, Sequence[int], Tensor]] = None,
    lengths2: Optional[Union[int, Sequence[int], Tensor]] = None,
) -> Tuple[Tensor, Tensor]:
    B, N1, _ = pc1.shape
    B, N2, _ = pc2.shape

    # Convert the input parameters to as 1D tensors os size B
    k = default_vector(k, size=B, default_value=3).long()  # (B,)
    lengths1 = default_vector(lengths1, size=B, default_value=N1).long()  # (B,)
    lengths2 = default_vector(lengths2, size=B, default_value=N2).long()  # (B,)

    # Make sure the values are within the correct range
    k = k.clamp(0, N2).to(pc1.device)
    lengths1 = lengths1.clamp(0, N1).to(pc1.device)
    lengths2 = lengths2.clamp(0, N2).to(pc1.device)

    return _C.knn(pc1, pc2, k, lengths1, lengths2)


def three_nn(
    pc1: Tensor,
    pc2: Tensor,
    lengths1: Optional[Union[int, Tensor]] = None,
    lengths2: Optional[Union[int, Tensor]] = None,
) -> Tuple[Tensor, Tensor]:
    B, N1, _ = pc1.shape
    B, N2, _ = pc2.shape

    # Convert the input parameters to as 1D tensors os size B
    lengths1 = default_vector(lengths1, size=B, default_value=N1).long()
    lengths2 = default_vector(lengths2, size=B, default_value=N2).long()

    # Make sure the values are within the correct range
    lengths1 = lengths1.clamp(0, N1).to(pc1.device)
    lengths2 = lengths2.clamp(0, N2).to(pc1.device)

    return _C.three_nn(pc1.contiguous(), pc2.contiguous(), lengths1, lengths2)


def knn_interpolate(
    features: Tensor,
    pos: Tensor,
    pos_skip: Tensor,
    k: int = 3,
    lengths1: Optional[Tensor] = None,
    lengths2: Optional[Tensor] = None,
) -> Tensor:
    dist, idx = knn(pos_skip, pos, k=k)
    dist_inv = 1.0 / (dist + 1e-8)
    norm = torch.sum(dist_inv, dim=2, keepdim=True)
    weight = dist_inv / norm
    return k_interpolate(features.contiguous(), idx.contiguous(), weight.contiguous(), k=k)


def three_nn_interpolate(features: Tensor, pos: Tensor, pos_skip: Tensor) -> Tensor:
    dists, idxs = three_nn(pos_skip, pos)
    dists_inv = 1.0 / (dists + 1e-8)
    norms = torch.sum(dists_inv, dim=2, keepdim=True)
    weights = dists_inv / norms
    return three_interpolate(features, idxs, weights)


def fps(
    points: Tensor,
    num_samples: Optional[Union[int, Tensor]] = None,
    ratio: Union[float, Tensor, None] = None,
    lengths: Optional[Union[int, Tensor]] = None,
    start_idxs: Union[bool, int, Tensor, None] = None,
) -> Tensor:
    B, N, _ = points.shape
    if num_samples is not None and ratio is None:
        num_samples = default_vector(num_samples, size=B).long()
    elif num_samples is None and ratio is not None:
        ratio = default_vector(ratio, size=B)
        num_samples = (N * ratio).round().long()
    else:
        raise ValueError("Invalid combination of num_samples and ratio. Expected only one of them to be specified.")

    if start_idxs is None or start_idxs is True:
        start_idxs = torch.randint(0, N, (B,), dtype=torch.int64)
    elif start_idxs is False:
        start_idxs = torch.zeros(B, dtype=torch.int64)

    start_idxs = default_vector(start_idxs, size=B).long()  # (B,)
    lengths = default_vector(lengths, size=B, default_value=N).long()  # (B,)

    num_samples = num_samples.to(points.device)
    start_idxs = start_idxs.clamp(0, N - 1).to(points.device)
    lengths = lengths.clamp(0, N).to(points.device)
    return _C.fps(points, lengths, num_samples, start_idxs)


def ball_query(
    pc1: Tensor,
    pc2: Tensor,
    radius: Union[float, Tensor],
    max_neighbors: Union[int, Sequence[int], Tensor],
    lengths1: Optional[Union[int, Sequence[int], Tensor]] = None,
    lengths2: Optional[Union[int, Sequence[int], Tensor]] = None,
) -> Tuple[Tensor, Tensor]:
    B, N1, _ = pc1.shape
    B, N2, _ = pc2.shape

    lengths1 = default_vector(lengths1, size=B, default_value=N1).long()  # (B,)
    lengths2 = default_vector(lengths2, size=B, default_value=N2).long()  # (B,)
    max_neighbors = default_vector(max_neighbors, size=B).long()  # (B,)
    radius = default_vector(radius, size=B)  # (B,)

    lengths1 = lengths1.to(pc1.device)
    lengths2 = lengths2.to(pc1.device)
    max_neighbors = max_neighbors.to(pc1.device)
    radius = radius.clamp(0).to(pc1.device)

    return _C.ball_query(pc1, pc2, lengths1, lengths2, max_neighbors, radius)


def ball_grouping(features: Tensor, idx: Tensor) -> Tensor:
    # TODO: Handle the -1 index directly in the ball_query, and add a fill_mode and fill_value parameters
    # TODO: to handle the -1, maybe pass the lengths of the features too...
    # fille_mode: str = 'closest', 'furthest', 'random', 'mirror', 'constant', 'pad' (pad: fill_value=-1)

    # all_idx = idx.reshape(idx.shape[0], -1)
    # all_idx = all_idx.unsqueeze(1).repeat(1, features.shape[1], 1)
    # grouped_features = features.gather(2, all_idx)
    # return grouped_features.reshape(idx.shape[0], features.shape[1], idx.shape[1], idx.shape[2])

    B, C, N = features.size()
    idx = idx.detach().clone()
    npoint = idx.size(1)
    nsample = idx.size(2)

    # Handle -1 in idx for padding
    mask = idx == -1  # Create a mask where idx is -1
    idx[mask] = 0  # Temporarily replace -1 with 0 to avoid out-of-bounds errors

    # Proceed with grouping
    all_idx = idx.view(B, -1).unsqueeze(1).expand(-1, C, -1)
    grouped_features = features.gather(2, all_idx).view(B, C, npoint, nsample)

    # Set features corresponding to -1 indices to zero
    mask = mask.unsqueeze(1).expand(-1, C, -1, -1)  # Expand mask to match grouped_features dimensions
    grouped_features[mask] = 0

    return grouped_features
