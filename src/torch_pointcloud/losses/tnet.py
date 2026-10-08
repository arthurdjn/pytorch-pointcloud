"""Orthogonality regularizer of the T-Net transforms (PointNet's feature-transform regularization)."""

import torch
import torch.nn as nn
from torch import Tensor

from torch_pointcloud.layers.tnet import DynamicTNet, TNet

__all__ = ["TNetOrthogonalityRegularizer"]


def tnet_orthogonality_regularizer(module: nn.Module) -> Tensor:
    r"""Pull the transforms predicted by every T-Net in `module` towards an orthogonal matrix.

    The feature-transform regularization of :arxiv: [PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](https://arxiv.org/abs/1612.00593) (Qi et al., 2017),
    $\lVert I - A A^\top \rVert_F$ averaged over the samples of the batch and summed over the T-Nets found in
    `module`. Pass the T-Net (or the sub-module) to regularize, e.g. a PointNet's feature transform `model.ftnet`
    to leave its input transform free, as the PointNet paper does.

    The matrices are read from the `running_transform` the layers keep from their last forward pass, so
    call it after the forward and before the backward of the same batch.

    Args:
        module: A `TNet` / `DynamicTNet`, or a module containing some.

    Returns:
        The scalar regularizer (zero when `module` holds no T-Net).

    Raises:
        RuntimeError: If a T-Net has no stored transform (no forward pass yet, or `track_running_stats=False`).

    Example:
        ```pycon
        >>> from torch_pointcloud.layers.tnet import TNet
        >>> tnet = TNet(local_channels=[32, 64], global_channels=[32], k=8)
        >>> _ = tnet(torch.randn(6, 8), torch.tensor([0, 0, 0, 1, 1, 1]))
        >>> tnet_orthogonality_regularizer(tnet).shape
        torch.Size([])

        ```
    """
    loss = torch.zeros((), device=next(module.parameters()).device)

    for layer in module.modules():
        if not isinstance(layer, (TNet, DynamicTNet)):
            continue

        if layer.running_transform is None:
            raise RuntimeError(
                "`tnet_orthogonality_regularizer` needs a forward pass with `track_running_stats=True` first."
            )

        transform = layer.running_transform
        identity = torch.eye(transform.shape[-1], dtype=transform.dtype, device=transform.device)
        loss = loss + torch.norm(identity - transform @ transform.transpose(1, 2), dim=(1, 2)).mean()

    return loss


class TNetOrthogonalityRegularizer(nn.Module):
    r"""Module form of [`tnet_orthogonality_regularizer`][torch_pointcloud.losses.tnet.tnet_orthogonality_regularizer].

    Example:
        ```python
        import torch
        from torch_pointcloud.layers import TNet
        from torch_pointcloud.losses import TNetOrthogonalityRegularizer
        ftnet = TNet(local_channels=[32, 64], global_channels=[32], k=16)
        out = ftnet(torch.randn(10, 16), torch.arange(2).repeat_interleave(5))
        TNetOrthogonalityRegularizer()(ftnet).shape  # torch.Size([])
        ```
    """

    def forward(self, module: nn.Module) -> Tensor:
        return tnet_orthogonality_regularizer(module)
