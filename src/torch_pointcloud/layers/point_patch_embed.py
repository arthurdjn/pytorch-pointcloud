"""Mini-PointNet patch embedding turning local point groups into tokens."""

from typing import Any, Callable, Dict, Optional, Sequence, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import MLP


class PointPatchEmbed(nn.Module):
    r"""Mini-PointNet patch (token) embedding for grouped point clouds.

    Embeds each local group of points into a single token via a two-stage shared MLP with an intermediate
    per-group max-pool and global-feature concatenation, as used by the masked / autoregressive point
    self-supervised models (Point-MAE, Point-BERT, PointGPT, Point-M2AE). A shared $1 \times 1$ convolution
    over $(B, C, M)$ is equivalent to a `MLP` over the feature dim, so both stages are plain PyG `MLP`s.
    `local_mlp` maps `in_channels` $\to \text{local\_channels}$; the per-group max-pool is concatenated to
    give $2 \cdot \text{local\_channels}[-1]$; `global_mlp` maps that to `embed_dim` through `global_channels`.
    The module is permutation-invariant over the points within a group.

    Args:
        embed_dim: Output token dimension.
        in_channels: Channels per input point ($3$ for coordinates only, plus any concatenated features).
        local_channels: Hidden widths of the per-point MLP (`in_channels` is prepended).
        global_channels: Hidden widths of the per-group MLP ($2 \cdot \text{local\_channels}[-1]$ input and `embed_dim` output are added).
        act: Activation of the MLPs.
        act_kwargs: Extra arguments for the activation.
        norm: Normalization of the MLPs.
        norm_kwargs: Extra arguments for the normalization.

    Shape:
        - Input: $(B, G, M, C)$ where $B$ is the batch size, $G$ the number of groups, $M$ the group size, and $C$ = `in_channels`.
        - Output: $(B, G, D)$ where $D$ = `embed_dim`.

    Example:
        ```python
        import torch
        from torch_pointcloud.layers import PointPatchEmbed

        embed = PointPatchEmbed(embed_dim=384)
        tokens = embed(torch.randn(2, 64, 32, 3))
        print(tokens.shape)
        ```
    """

    def __init__(
        self,
        embed_dim: int,
        in_channels: int = 3,
        local_channels: Sequence[int] = (128, 256),
        global_channels: Sequence[int] = (512,),
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.mid_channels = local_channels[-1]
        factory_kwargs: Dict[str, Any] = dict(
            act=act, act_kwargs=act_kwargs, norm=norm, norm_kwargs=norm_kwargs, plain_last=True
        )
        self.local_mlp = MLP([in_channels, *local_channels], **factory_kwargs)
        self.global_mlp = MLP([2 * self.mid_channels, *global_channels, embed_dim], **factory_kwargs)

    def forward(self, neighborhood: Tensor) -> Tensor:
        B, G, M, C = neighborhood.shape
        points = neighborhood.reshape(B * G * M, C)
        feature = self.local_mlp(points).reshape(B * G, M, self.mid_channels)
        feature_global = feature.max(dim=1, keepdim=True)[0]
        feature = torch.cat([feature_global.expand(-1, M, -1), feature], dim=-1).reshape(
            B * G * M, 2 * self.mid_channels
        )
        feature = self.global_mlp(feature).reshape(B * G, M, self.embed_dim)
        feature_global = feature.max(dim=1)[0]
        return feature_global.reshape(B, G, self.embed_dim)
