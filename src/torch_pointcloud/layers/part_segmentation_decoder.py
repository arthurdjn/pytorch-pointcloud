"""Category-conditioned decoder from patch tokens back to per-point features."""

from typing import Sequence

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import MLP

from .pointnet2_blocks import PointNet2FeaturePropagation


class PartSegmentationDecoder(nn.Module):
    r"""Category-conditioned decoder from patch tokens to per-point features.

    Interpolates the patch features back to every point with a PointNet++-style feature propagation
    (`PointNet2FeaturePropagation`, the point coordinates as skip features) and appends a per-shape global
    descriptor: the max- and mean-pooled patch features and an embedding of the category one-hot, broadcast to the
    points of the shape. This is the decoder the patch-token transformers put in front of their per-point
    classifier for part segmentation. The points stay packed, so the shapes may have different numbers of points.

    Args:
        in_channels: The patch feature width $C$.
        num_categories: The number of object categories $K$ of the one-hot.
        propagation_channels: Widths of the feature-propagation MLP, the last one $C_p$ being its output width.
        category_channels: The width $E$ of the category embedding.
        spatial_dim: The spatial dimension $D$ of the points and patch centers.

    Shape:
        - x: $(B, G, C)$ patch features.
        - center: $(B, G, D)$ patch centers.
        - pos: $(N, D)$ point coordinates.
        - batch: $(N,)$ shape index of every point, sorted (as `collate` builds it): the $k$-NN interpolation
          pairs points and patches across shapes on an unsorted `batch`.
        - category: $(B, K)$ category one-hot.
        - Output: $(N, C_p + 2 C + E)$.

    Example:
        ```python
        import torch
        from torch_pointcloud.layers import PartSegmentationDecoder

        decoder = PartSegmentationDecoder(in_channels=96, num_categories=16, propagation_channels=(64, 32))
        x, center = torch.randn(2, 8, 96), torch.randn(2, 8, 3)
        pos = torch.randn(300, 3)
        batch = torch.cat([torch.zeros(200), torch.ones(100)]).long()
        category = torch.nn.functional.one_hot(torch.tensor([3, 7]), 16).float()
        out = decoder(x, center, pos, batch, category)
        print(out.shape)
        ```
    """

    def __init__(
        self,
        in_channels: int,
        num_categories: int,
        propagation_channels: Sequence[int],
        category_channels: int = 64,
        spatial_dim: int = 3,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_categories = num_categories
        self.propagation_channels = propagation_channels
        self.category_channels = category_channels
        self.category_embed = MLP(
            [num_categories, category_channels],
            act="leaky_relu",
            act_kwargs=dict(negative_slope=0.2),
            norm="batch_norm",
            bias=False,
            plain_last=False,
        )
        self.propagation = PointNet2FeaturePropagation(
            channels=[in_channels + spatial_dim, *propagation_channels],
            k=3,
            act="relu",
            norm="batch_norm",
            bias=True,
            plain_last=False,
            weighting="squared",
            eps=1e-8,
        )

    @property
    def out_channels(self) -> int:
        """Channel count of the per-point features."""
        return self.propagation_channels[-1] + 2 * self.in_channels + self.category_channels

    def forward(self, x: Tensor, center: Tensor, pos: Tensor, batch: Tensor, category: Tensor) -> Tensor:
        B, G, C = x.shape
        if category.shape != (B, self.num_categories):
            raise ValueError(
                f"`category` must be a one-hot of shape ({B}, {self.num_categories}), one row per shape of `x`; "
                f"got {tuple(category.shape)}."
            )

        x_global = torch.cat([x.amax(dim=1), x.mean(dim=1), self.category_embed(category)], dim=1)

        patch_batch = torch.arange(B, device=x.device).repeat_interleave(G)
        x_patch = x.reshape(B * G, C)
        center = center.reshape(B * G, -1)
        x_point, _, _ = self.propagation(x_patch, center, patch_batch, pos, pos, batch)
        return torch.cat([x_point, x_global[batch]], dim=1)
