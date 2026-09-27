"""Fitting and repulsive regularizer of deformable kernel point convolutions."""

import torch
import torch.nn as nn
from torch import Tensor

from torch_pointcloud.layers.kpconv_blocks import KPConv


def kpconv_deform_regularizer(module: nn.Module, fitting_power: float = 1.0, repulse_extent: float = 1.0) -> Tensor:
    r"""Regularize the deformed kernel points of every deformable `KPConv` in `module`.

    The point-to-point regularizer of :arxiv: [KPConv](https://arxiv.org/abs/1904.08889): a fitting term pulls
    every deformed kernel point towards its closest input point, a repulsive term keeps the kernel points of one
    location further apart than `repulse_extent`. Both read the statistics the layers keep from their last
    forward pass (`running_min_d2`, `running_deformed_kernel`), so call it after the forward and before the
    backward of the same batch. Distances are expressed in units of each layer's `kp_sigma`.

    Args:
        module: Model holding the `KPConv` layers.
        fitting_power: Weight of the whole term.
        repulse_extent: Distance, in units of `kp_sigma`, below which kernel points repel each other.

    Returns:
        A scalar, `fitting_power * (2 * fitting + repulsive)` summed over the deformable layers, zero when
        `module` has none.

    Example:
        ```python
        >>> import torch
        >>> from torch_pointcloud.layers import KPConv
        >>> from torch_pointcloud.losses import kpconv_deform_regularizer
        >>> conv = KPConv(spatial_dim=3, in_channels=4, out_channels=8, kernel_size=15, kp_radius=1.0, kp_sigma=1.0, deformable=True)
        >>> pos = torch.rand(32, 3)
        >>> edge_index = torch.stack([torch.arange(32).repeat_interleave(4), torch.randint(0, 32, (128,))])
        >>> out = conv(torch.randn(32, 4), pos, edge_index)
        >>> kpconv_deform_regularizer(conv).shape
        torch.Size([])

        ```
    """
    loss = torch.zeros((), device=next(module.parameters()).device)
    for layer in module.modules():
        if not isinstance(layer, KPConv) or not layer.deformable:
            continue
        if layer.running_min_d2 is None or layer.running_deformed_kernel is None:
            raise RuntimeError(
                "`kpconv_deform_regularizer` needs a forward pass with `track_running_stats=True` first."
            )

        fitting = (layer.running_min_d2 / layer.kp_sigma**2).mean()
        kernel_points = layer.running_deformed_kernel / layer.kp_sigma  # (N, K, D)
        # Only the point being pushed receives a gradient, the others are held fixed.
        distances = torch.cdist(kernel_points, kernel_points.detach())  # (N, K, K)
        overlap = torch.clamp_max(distances - repulse_extent, 0.0) ** 2
        overlap = overlap - torch.diag_embed(torch.diagonal(overlap, dim1=-2, dim2=-1))
        repulsive = overlap.sum(-1).mean()
        loss = loss + fitting_power * (2 * fitting + repulsive)
    return loss


class KPConvDeformRegularizer(nn.Module):
    r"""Module form of `kpconv_deform_regularizer`, to sit next to the task loss in a training loop.

    Args:
        fitting_power: Weight of the whole term.
        repulse_extent: Distance, in units of `kp_sigma`, below which kernel points repel each other.

    Example:
        ```python
        >>> import torch
        >>> from torch_pointcloud.layers import KPConv
        >>> from torch_pointcloud.losses import KPConvDeformRegularizer
        >>> conv = KPConv(spatial_dim=3, in_channels=4, out_channels=8, kernel_size=15, kp_radius=1.0, kp_sigma=1.0, deformable=True)
        >>> pos = torch.rand(32, 3)
        >>> edge_index = torch.stack([torch.arange(32).repeat_interleave(4), torch.randint(0, 32, (128,))])
        >>> out = conv(torch.randn(32, 4), pos, edge_index)
        >>> KPConvDeformRegularizer(repulse_extent=1.2)(conv).shape
        torch.Size([])

        ```
    """

    def __init__(self, fitting_power: float = 1.0, repulse_extent: float = 1.0) -> None:
        super().__init__()
        self.fitting_power = fitting_power
        self.repulse_extent = repulse_extent

    def forward(self, module: nn.Module) -> Tensor:
        return kpconv_deform_regularizer(module, fitting_power=self.fitting_power, repulse_extent=self.repulse_extent)

    def extra_repr(self) -> str:
        return f"fitting_power={self.fitting_power}, repulse_extent={self.repulse_extent}"
