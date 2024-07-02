from typing import Any, Tuple

from torch import Tensor
from torch.autograd import Function
from torch_pvcnn import _C  # type: ignore[attr-defined]


class TrilinearDevoxelize(Function):
    @staticmethod
    def forward(ctx: Any, coords: Tensor, features: Tensor, resolution: int) -> Tensor:
        out, idxs, weights = _C.trilinear_devoxelize(coords.contiguous(), features.contiguous(), resolution)
        ctx.save_for_backward(idxs, weights)
        ctx.resolution = resolution
        return out

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Tensor) -> Tuple[None, Tensor, None]:
        grad_out, *_ = grad_outputs
        idxs, weights = ctx.saved_tensors
        grad_features = _C.trilinear_devoxelize_backward(
            grad_out.contiguous(), idxs.contiguous(), weights.contiguous(), ctx.resolution
        )
        return None, grad_features, None


class AvgVoxelize(Function):
    @staticmethod
    def forward(ctx: Any, coords: Tensor, features: Tensor, resolution: int) -> Tensor:
        out, idxs, counts = _C.avg_voxelize(coords.contiguous(), features.contiguous(), resolution)
        ctx.save_for_backward(idxs, counts)
        return out

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Tensor) -> Tuple[None, Tensor, None]:
        grad_out, *_ = grad_outputs
        idxs, counts = ctx.saved_tensors
        grad_features = _C.avg_voxelize_backward(grad_out.contiguous(), idxs.contiguous(), counts.contiguous())
        return None, grad_features, None


def trilinear_devoxelize(coords: Tensor, features: Tensor, resolution: int) -> Tensor:
    """Perform trilinear devoxelization.

    Args:
        coords: Coordinates of the voxels, of shape (B, N, 3)
        features: Features of the voxels, of shape (B, C, R, R, R)
        resolution: Resolution of the voxel grid R

    Returns:
        The devoxelized features of the points, of shape (B, C, N)
    """
    return TrilinearDevoxelize.apply(coords, features, resolution)


def avg_voxelize(coords: Tensor, features: Tensor, resolution: int) -> Tensor:
    """Perform average voxelization.

    Args:
        coords: Coordinates of the points, of shape (B, N, 3)
        features: Features of the points, of shape (B, C, N)
        resolution: Resolution of the voxel grid

    Returns:
        The voxelized features, of shape (B, C, R, R, R)
    """
    return AvgVoxelize.apply(coords, features, resolution)
