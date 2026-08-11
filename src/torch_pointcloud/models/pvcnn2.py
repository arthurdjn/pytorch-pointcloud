from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union, overload

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import MLP
from torch_geometric.typing import OptTensor

import torch_pointcloud.transforms as T
from torch_pointcloud.layers import PoolLike, create_pool
from torch_pointcloud.layers.conv3d_blocks import Conv3dBlock
from torch_pointcloud.layers.pointnet2_blocks import FPModule, SAModule, ensure_msg_list, ensure_msg_list_size
from torch_pointcloud.layers.pvcnn_blocks import PVConv
from torch_pointcloud.utils.conversion import ensure_tuple, ensure_tuple_size
from torch_pointcloud.utils.data import DataKeys

from ._base import ClassificationModel, SegmentationModel
from ._registry import register_model


class PVCNN2EncoderBlock(nn.Module):
    r"""One PVCNN++ encoder stage: optional set-abstraction downsampling followed by point-voxel convs.

    When `sa_module` is given, the block first downsamples the cloud (farthest point sampling +
    ball grouping), then refines the abstracted features with `depth` point-voxel conv layers.
    A `resolution` of $0$ or `None` swaps each point-voxel conv for a plain MLP layer.

    Args:
        in_channels: Number of input feature channels of the conv stack.
        out_channels: Number of output feature channels of the conv stack.
        depth: Number of point-voxel conv (or MLP) layers.
        resolution: Voxel grid resolution of the point-voxel convs; $0$ or `None` uses MLP layers.
        kernel_size: Kernel size of the voxel branch convolutions.
        use_se: Add a squeeze-and-excitation gate to every voxel branch.
        normalize: Normalize coordinates into the unit voxel grid before voxelization.
        act: Activation function.
        act_kwargs: Keyword arguments for the activation function.
        act_first: Apply the activation before the normalization.
        norm: Normalization function.
        norm_kwargs: Keyword arguments for the normalization function.
        sa_module: Optional set-abstraction module applied before the conv stack.

    Shape:
        - `x`: $(N, C_{in})$ point features.
        - `pos`: $(N, 3)$ point coordinates.
        - `batch`: $(N,)$ batch indices.
        - output: $(M, C_{out})$ features with matching `pos` / `batch`, where $M \le N$ is the
          number of points kept by the set-abstraction module ($M = N$ without one).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int,
        resolution: int,
        kernel_size: int,
        use_se: bool = False,
        normalize: bool = True,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        sa_module: Optional[SAModule] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        kwargs = dict(
            act=act,
            act_kwargs=act_kwargs,
            act_first=act_first,
            norm=norm,
            norm_kwargs=norm_kwargs,
        )

        self.sa_module = sa_module
        self.layers = nn.ModuleList([])
        for i in range(depth):
            in_channels = self.in_channels if i == 0 else self.out_channels
            if not resolution:
                # In case resolution is 0 or None, use a linear block
                layer = MLP([in_channels, self.out_channels], plain_last=False, **kwargs)
            else:
                layer = PVConv(
                    in_channels=in_channels,
                    out_channels=self.out_channels,
                    kernel_size=kernel_size,
                    resolution=resolution,
                    use_se=use_se,
                    normalize=normalize,
                    **kwargs,  # type: ignore[arg-type]
                )

            self.layers.append(layer)

    def forward(self, x: Tensor, pos: Tensor, batch: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        if self.sa_module is not None:
            x, pos, batch = self.sa_module(x, pos, batch)

        for layer in self.layers:
            x = layer(x) if isinstance(layer, MLP) else layer(x, pos, batch)
        return x, pos, batch


class PVCNN2Encoder(nn.Module):
    r"""Hierarchical PVCNN++ encoder.

    Block $0$ processes the full-resolution cloud with point-voxel conv layers; every following
    block $i > 0$ first downsamples with a set-abstraction module configured by the $(i-1)$-th
    entry of `ratios` / `radii` / `num_neighbors` / `sa_channels`, then refines with its own conv
    stack. Those four lists therefore pad to the block count and their final entry is unused.

    Args:
        channels: Per-block feature widths, one more entry than the number of blocks:
            `channels[i + 1]` is the output width of block $i$.
        depths: Number of point-voxel conv layers per block; $0$ keeps the block
            set-abstraction-only.
        resolutions: Voxel grid resolution per block; $0$ or `None` uses MLP layers instead of
            point-voxel convs.
        kernel_sizes: Voxel conv kernel size per block.
        use_se: Add a squeeze-and-excitation gate to every voxel branch.
        normalize: Normalize coordinates into the unit voxel grid before voxelization.
        sa_channels: MLP channels of each set-abstraction module. Nest twice for multi-scale
            grouping.
        ratios: Farthest-point-sampling ratio per set-abstraction module.
        radii: Ball-query radius per set-abstraction module. A nested sequence enables
            multi-scale grouping.
        num_neighbors: Maximum number of neighbors per ball query.
        act: Activation function.
        act_kwargs: Keyword arguments for the activation function.
        act_first: Apply the activation before the normalization.
        norm: Normalization function.
        norm_kwargs: Keyword arguments for the normalization function.

    Shape:
        - `x`: $(N, C)$ point features with $C = \text{channels}[0]$.
        - `pos`: $(N, 3)$ point coordinates.
        - `batch`: $(N,)$ batch indices.
        - output: $(M, \text{channels}[-1])$ features with matching `pos` / `batch`, where $M$ is
          the number of points left after all set-abstraction stages.
    """

    def __init__(
        self,
        *,
        channels: Sequence[int],
        depths: Sequence[int],
        resolutions: Sequence[Optional[int]],
        kernel_sizes: Sequence[int],
        use_se: bool = False,
        normalize: bool = True,
        sa_channels: Sequence[Sequence[Sequence[int]]],
        ratios: Sequence[float],
        radii: Sequence[Union[float, Sequence[float]]],
        num_neighbors: Sequence[Union[int, Sequence[int]]],
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.depths = ensure_tuple(depths)
        num_blocks = len(self.depths)

        self.channels = ensure_tuple_size(channels, size=num_blocks + 1)
        self.resolutions = ensure_tuple_size(resolutions, size=num_blocks)
        self.kernel_sizes = ensure_tuple_size(kernel_sizes, size=num_blocks)
        self.sa_channels = ensure_msg_list_size(sa_channels, size=num_blocks)
        self.ratios = ensure_tuple_size(ratios, size=num_blocks)
        self.radii = ensure_tuple_size(radii, size=num_blocks)
        self.num_neighbors = ensure_tuple_size(num_neighbors, size=num_blocks)

        self.blocks = nn.ModuleList([])
        for i in range(num_blocks):
            sa_block: Optional[SAModule] = None
            if i > 0:
                sa_block = SAModule(
                    in_channels=self.channels[i],
                    channels=self.sa_channels[i - 1],
                    ratio=self.ratios[i - 1],
                    radii=self.radii[i - 1],
                    num_neighbors=self.num_neighbors[i - 1],
                    act=act,
                    act_kwargs=act_kwargs,
                    act_first=act_first,
                    norm=norm,
                    norm_kwargs=norm_kwargs,
                )

            block = PVCNN2EncoderBlock(
                in_channels=self.channels[i] if i == 0 else self.channels[i + 1],
                out_channels=self.channels[i + 1],
                depth=self.depths[i],
                resolution=self.resolutions[i],
                kernel_size=self.kernel_sizes[i],
                use_se=use_se,
                normalize=normalize,
                act=act,
                act_kwargs=act_kwargs,
                act_first=act_first,
                norm=norm,
                norm_kwargs=norm_kwargs,
                sa_module=sa_block,
            )

            self.blocks.append(block)

    @overload
    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        batch: Tensor,
        return_intermediates: Literal[True],
    ) -> Tuple[Tensor, Tensor, Tensor, List[Dict[str, Tensor]]]: ...

    @overload
    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        batch: Tensor,
        return_intermediates: Literal[False] = False,
    ) -> Tuple[Tensor, Tensor, Tensor]: ...

    def forward(self, x: Tensor, pos: Tensor, batch: Tensor, return_intermediates: bool = False) -> Any:
        intermediates = []

        for block in self.blocks:
            if return_intermediates:
                intermediates.append({"features": x, "pos": pos, "batch": batch})
            x, pos, batch = block(x, pos, batch)

        if return_intermediates:
            return x, pos, batch, intermediates
        return x, pos, batch


class PVCNN2DecoderBlock(nn.Module):
    r"""One PVCNN++ decoder stage: feature propagation followed by point-voxel conv refinement.

    When `fp_module` is given, the block first upsamples the features to the skip resolution
    (k-NN interpolation + skip concatenation + MLP), then refines with `depth` point-voxel conv
    layers. A `resolution` of $0$ or `None` swaps each point-voxel conv for a plain MLP layer.

    Args:
        in_channels: Number of input feature channels of the conv stack.
        out_channels: Number of output feature channels of the conv stack.
        depth: Number of point-voxel conv (or MLP) layers.
        resolution: Voxel grid resolution of the point-voxel convs; $0$ or `None` uses MLP layers.
        kernel_size: Kernel size of the voxel branch convolutions.
        use_se: Add a squeeze-and-excitation gate to every voxel branch.
        normalize: Normalize coordinates into the unit voxel grid before voxelization.
        act: Activation function.
        act_kwargs: Keyword arguments for the activation function.
        act_first: Apply the activation before the normalization.
        norm: Normalization function.
        norm_kwargs: Keyword arguments for the normalization function.
        fp_module: Optional feature-propagation module applied before the conv stack.

    Shape:
        - `x`: $(M, C_{in})$ point features at the coarse resolution.
        - `pos`: $(M, 3)$ coarse point coordinates.
        - `batch`: $(M,)$ coarse batch indices.
        - `x_skip` / `pos_skip` / `batch_skip`: skip tensors at the target resolution $N$.
        - output: $(N, C_{out})$ features with the skip `pos` / `batch`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int,
        resolution: int,
        kernel_size: int,
        use_se: bool = False,
        normalize: bool = True,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        fp_module: Optional[FPModule] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        kwargs = dict(
            act=act,
            act_kwargs=act_kwargs,
            act_first=act_first,
            norm=norm,
            norm_kwargs=norm_kwargs,
        )

        self.fp_module = fp_module
        self.layers = nn.ModuleList([])
        for i in range(depth):
            in_channels = self.in_channels if i == 0 else self.out_channels
            if not resolution:
                # In case resolution is 0 or None, use a linear block
                layer = MLP([in_channels, self.out_channels], plain_last=False, **kwargs)
            else:
                layer = PVConv(
                    in_channels=in_channels,
                    out_channels=self.out_channels,
                    kernel_size=kernel_size,
                    resolution=resolution,
                    use_se=use_se,
                    normalize=normalize,
                    **kwargs,  # type: ignore[arg-type]
                )

            self.layers.append(layer)

    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        batch: Tensor,
        x_skip: OptTensor = None,
        pos_skip: OptTensor = None,
        batch_skip: OptTensor = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        if self.fp_module is not None:
            x, pos, batch = self.fp_module(x, pos, batch, x_skip, pos_skip, batch_skip)

        for layer in self.layers:
            x = layer(x) if isinstance(layer, MLP) else layer(x, pos, batch)
        return x, pos, batch


class PVCNN2Decoder(nn.Module):
    r"""PVCNN++ decoder: a chain of feature-propagation blocks with point-voxel conv refinement.

    Skips are consumed deepest-first from the encoder intermediates, and the last block always
    uses the earliest intermediate (the raw encoder input), so the chain ends at full resolution
    even when the encoder has more blocks than the decoder; the leftover shallow intermediates
    are skipped.

    Args:
        in_channels: Feature width entering the first block (the encoder output width).
        depths: Number of point-voxel conv layers per block; $0$ keeps the block
            feature-propagation-only.
        channels: Per-block feature widths of the conv stacks.
        skip_channels: Skip feature width consumed by each block, deepest first.
        fp_channels: MLP channels of each feature-propagation module.
        resolutions: Voxel grid resolution per block; $0$ or `None` uses MLP layers instead of
            point-voxel convs.
        kernel_sizes: Voxel conv kernel size per block.
        use_se: Add a squeeze-and-excitation gate to every voxel branch.
        normalize: Normalize coordinates into the unit voxel grid before voxelization.
        act: Activation function.
        act_kwargs: Keyword arguments for the activation function.
        act_first: Apply the activation before the normalization.
        norm: Normalization function.
        norm_kwargs: Keyword arguments for the normalization function.

    Shape:
        - `x`: $(M, C_{in})$ encoder output features.
        - `pos`: $(M, 3)$ encoder output coordinates.
        - `batch`: $(M,)$ encoder output batch indices.
        - `intermediates`: per-encoder-block skip dicts with `features` / `pos` / `batch` keys.
        - output: $(N, C_{out})$ features at the resolution of the first intermediate.
    """

    def __init__(
        self,
        in_channels: int,
        depths: Sequence[int],
        channels: Sequence[int],
        skip_channels: Sequence[int],
        fp_channels: Sequence[Sequence[int]],
        resolutions: Sequence[Optional[int]],
        kernel_sizes: Sequence[int],
        use_se: bool = False,
        normalize: bool = True,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.depths = ensure_tuple(depths)
        n = len(self.depths)

        extra_msg = f"The number of `{{param}}` must match the number of blocks ({n})."
        self.channels = ensure_tuple_size(channels, size=n, extra_msg=extra_msg.format(param="channels"))
        self.skip_channels = ensure_tuple_size(skip_channels, size=n, extra_msg=extra_msg.format(param="skip_channels"))
        self.fp_channels = ensure_tuple_size(fp_channels, size=n, extra_msg=extra_msg.format(param="fp_channels"))
        self.resolutions = ensure_tuple_size(resolutions, size=n, extra_msg=extra_msg.format(param="resolutions"))
        self.kernel_sizes = ensure_tuple_size(kernel_sizes, size=n, extra_msg=extra_msg.format(param="kernel_sizes"))
        self.out_channels = int(self.channels[-1]) if self.depths[-1] else int(self.fp_channels[-1][-1])

        self.blocks = nn.ModuleList([])
        for i in range(n):
            fp_in_channels = self.in_channels if i == 0 else int(self.fp_channels[i - 1][-1])
            fp_module = FPModule(
                in_channels=fp_in_channels + self.skip_channels[i],
                channels=self.fp_channels[i],
                k=3,  # TODO: replace with spatial_dim
                act=act,
                act_kwargs=act_kwargs,
                act_first=act_first,
                norm=norm,
                norm_kwargs=norm_kwargs,
            )

            block = PVCNN2DecoderBlock(
                in_channels=self.channels[i],
                out_channels=self.channels[i],
                depth=self.depths[i],
                resolution=self.resolutions[i],
                kernel_size=self.kernel_sizes[i],
                use_se=use_se,
                normalize=normalize,
                act=act,
                act_kwargs=act_kwargs,
                act_first=act_first,
                norm=norm,
                norm_kwargs=norm_kwargs,
                fp_module=fp_module,
            )
            self.blocks.append(block)

    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        batch: Tensor,
        intermediates: List[Dict[str, Tensor]],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        if len(intermediates) < len(self.blocks):
            raise ValueError(f"Expected at least {len(self.blocks)} intermediates, got {len(intermediates)}.")

        skips = [intermediates[-1 - i] for i in range(len(self.blocks) - 1)]
        skips.append(intermediates[0])
        for block, intermediate in zip(self.blocks, skips):
            x_skip, pos_skip, batch_skip = intermediate["features"], intermediate["pos"], intermediate["batch"]
            x, pos, batch = block(x, pos, batch, x_skip, pos_skip, batch_skip)
        return x, pos, batch


class PVCNN2Classification(ClassificationModel):
    r"""PVCNN++ classification model from
    :arxiv: [Point-Voxel CNN for Efficient 3D Deep Learning](https://arxiv.org/abs/1907.03739)
    by Zhijian Liu, Haotian Tang, Yujun Lin, Song Han.

    PVCNN++ grafts point-voxel convolutions onto a PointNet++-style hierarchy: every encoder
    block downsamples the cloud with a set-abstraction module (farthest point sampling + ball
    grouping), then refines the abstracted features with `PVConv` layers that fuse a coarse
    voxel-grid convolution branch with a per-point MLP branch. The final features are pooled
    into a global embedding and classified with a linear head.

    Args:
        in_channels: Number of input feature channels. Takes precedence over
            `encoder_channels[0]` when the two disagree.
        num_classes: Number of output classes. $0$ replaces the head with `nn.Identity`.
        ratios: Farthest-point-sampling ratio per encoder block. Block $i$ uses `ratios[i - 1]`
            (block $0$ has no set-abstraction module), so the list pads to the block count and
            its final entry is unused.
        radii: Ball-query radius per encoder block, aligned like `ratios`. A nested sequence
            enables multi-scale grouping.
        num_neighbors: Maximum number of neighbors per ball query, aligned like `ratios`.
        sa_channels: MLP channels of each set-abstraction module, aligned like `ratios`. Nest
            twice for multi-scale grouping.
        encoder_channels: Per-block feature widths, one more entry than the number of blocks:
            `encoder_channels[i + 1]` is the output width of block $i$.
        encoder_depths: Number of point-voxel conv layers per block; $0$ keeps the block
            set-abstraction-only.
        encoder_resolutions: Voxel grid resolution per block; $0$ or `None` uses MLP layers
            instead of point-voxel convs.
        encoder_kernel_sizes: Voxel conv kernel size per block.
        use_se: Add a squeeze-and-excitation gate to every voxel branch.
        normalize: Normalize coordinates into the unit voxel grid before voxelization.
        act: Activation function.
        act_kwargs: Keyword arguments for the activation function.
        act_first: Apply the activation before the normalization.
        norm: Normalization function.
        norm_kwargs: Keyword arguments for the normalization function.
        dropout: Dropout probability applied to the pooled global embedding.
        global_pool: Pooling used to aggregate per-point features into the global embedding.

    Shape:
        - `x`: $(N, C_{in})$ point features; when `None`, `pos` is used as features.
        - `pos`: $(N, 3)$ point coordinates.
        - `batch`: $(N,)$ batch indices.
        - output: $(B, \text{num\_classes})$ classification logits.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        *,
        ratios: Sequence[float],
        radii: Sequence[Union[float, Sequence[float]]],
        num_neighbors: Sequence[Union[int, Sequence[int]]],
        sa_channels: Sequence[Sequence[Union[int, Sequence[int]]]],
        encoder_channels: Sequence[int],
        encoder_depths: Sequence[int],
        encoder_resolutions: Sequence[Optional[int]],
        encoder_kernel_sizes: Sequence[int],
        use_se: bool = False,
        normalize: bool = True,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        dropout: float = 0.0,
        global_pool: PoolLike = "max",
    ):
        super().__init__(in_channels=in_channels, num_classes=num_classes)
        encoder_channels = [self.in_channels, *list(encoder_channels)[1:]]
        self.embedding_dim = encoder_channels[-1]
        sa_channels = ensure_msg_list(sa_channels)

        self.encoder = PVCNN2Encoder(
            channels=encoder_channels,
            depths=encoder_depths,
            resolutions=encoder_resolutions,
            kernel_sizes=encoder_kernel_sizes,
            sa_channels=sa_channels,
            ratios=ratios,
            radii=radii,
            num_neighbors=num_neighbors,
            use_se=use_se,
            normalize=normalize,
            act=act,
            act_kwargs=act_kwargs,
            act_first=act_first,
            norm=norm,
            norm_kwargs=norm_kwargs,
        )

        self.dropout = dropout
        self.global_pool = create_pool(global_pool)
        self.head = self.configure_head()

    def configure_head(self) -> nn.Module:
        return nn.Identity() if self.num_classes == 0 else nn.Linear(self.embedding_dim, self.num_classes)

    def reset_classifier(self, num_classes: int, **kwargs: Any) -> None:
        self.num_classes = num_classes
        self.head = self.configure_head()

    @overload
    def forward_features(
        self,
        x: OptTensor,
        pos: Tensor,
        batch: Tensor,
        return_intermediates: Literal[True],
    ) -> Tuple[Tensor, Tensor, Tensor, List[Dict[str, Tensor]]]: ...

    @overload
    def forward_features(
        self,
        x: OptTensor,
        pos: Tensor,
        batch: Tensor,
        return_intermediates: Literal[False] = False,
    ) -> Tuple[Tensor, Tensor, Tensor]: ...

    def forward_features(
        self,
        x: OptTensor,
        pos: Tensor,
        batch: Tensor,
        return_intermediates: bool = False,
    ) -> Any:
        x = x if x is not None else pos
        return self.encoder(x, pos, batch, return_intermediates=return_intermediates)

    def forward_head(self, x: Tensor, batch: Tensor, pre_logits: bool = False) -> Tensor:
        x = self.global_pool(x, batch)
        if self.dropout:
            x = F.dropout(x, p=float(self.dropout), training=self.training)
        return x if pre_logits else self.head(x)

    def forward(self, x: OptTensor, pos: Tensor, batch: Tensor) -> Tensor:
        x, pos, batch, intermediates = self.forward_features(x, pos, batch, return_intermediates=True)
        return self.forward_head(x, batch)


class PVCNN2Segmentation(SegmentationModel):
    r"""PVCNN++ segmentation model from
    :arxiv: [Point-Voxel CNN for Efficient 3D Deep Learning](https://arxiv.org/abs/1907.03739)
    by Zhijian Liu, Haotian Tang, Yujun Lin, Song Han.

    A U-shaped PVCNN++: the encoder alternates set-abstraction downsampling with point-voxel
    conv refinement, and the decoder upsamples back with feature-propagation blocks followed by
    point-voxel convs. Decoder skips are consumed deepest-first, and the last decoder block uses
    the extra input features (the input without its 3 leading coordinate channels) as its skip,
    so the encoder may have more blocks than the decoder (the leftover shallow intermediates are
    skipped).

    Args:
        in_channels: Number of input feature channels. Takes precedence over
            `encoder_channels[0]` when the two disagree; $0$ falls back to $3$ (positions used
            as features).
        num_classes: Number of output classes. $0$ replaces the head with `nn.Identity`.
        ratios: Farthest-point-sampling ratio per encoder block. Block $i$ uses `ratios[i - 1]`
            (block $0$ has no set-abstraction module), so the list pads to the block count and
            its final entry is unused.
        radii: Ball-query radius per encoder block, aligned like `ratios`. A nested sequence
            enables multi-scale grouping.
        num_neighbors: Maximum number of neighbors per ball query, aligned like `ratios`.
        sa_channels: MLP channels of each set-abstraction module, aligned like `ratios`. Nest
            twice for multi-scale grouping.
        encoder_channels: Per-block feature widths, one more entry than the number of encoder
            blocks: `encoder_channels[i + 1]` is the output width of block $i$.
        encoder_depths: Number of point-voxel conv layers per encoder block; $0$ keeps the block
            set-abstraction-only.
        encoder_resolutions: Voxel grid resolution per encoder block; $0$ or `None` uses MLP
            layers instead of point-voxel convs.
        encoder_kernel_sizes: Voxel conv kernel size per encoder block.
        fp_channels: MLP channels of each feature-propagation module, deepest stage first.
        decoder_channels: Per-block feature widths of the decoder conv stacks.
        decoder_depths: Number of point-voxel conv layers per decoder block; $0$ keeps the block
            feature-propagation-only. At most as many decoder blocks as encoder blocks.
        decoder_resolutions: Voxel grid resolution per decoder block; $0$ or `None` uses MLP
            layers instead of point-voxel convs.
        decoder_kernel_sizes: Voxel conv kernel size per decoder block.
        use_se: Add a squeeze-and-excitation gate to every voxel branch.
        normalize: Normalize coordinates into the unit voxel grid before voxelization.
        act: Activation function.
        act_kwargs: Keyword arguments for the activation function.
        act_first: Apply the activation before the normalization.
        norm: Normalization function.
        norm_kwargs: Keyword arguments for the normalization function.
        dropout: Dropout probability before the linear head. Ignored when `head_channels` is
            set: the MLP head applies its own `head_dropout`.
        head_channels: Hidden widths of the segmentation head MLP. `None` uses a single linear
            layer.
        head_dropout: Dropout probability inside the head MLP.

    Shape:
        - `x`: $(N, C_{in})$ point features; when `None`, `pos` is used as features.
        - `pos`: $(N, 3)$ point coordinates.
        - `batch`: $(N,)$ batch indices.
        - output: $(N, \text{num\_classes})$ per-point segmentation logits.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        *,
        ratios: Sequence[float],
        radii: Sequence[Union[float, Sequence[float]]],
        num_neighbors: Sequence[Union[int, Sequence[int]]],
        sa_channels: Sequence[Sequence[Union[int, Sequence[int]]]],
        encoder_channels: Sequence[int],
        encoder_depths: Sequence[int],
        encoder_resolutions: Sequence[Optional[int]],
        encoder_kernel_sizes: Sequence[int],
        fp_channels: Sequence[Sequence[int]],
        decoder_channels: Sequence[int],
        decoder_depths: Sequence[int],
        decoder_resolutions: Sequence[Optional[int]],
        decoder_kernel_sizes: Sequence[int],
        use_se: bool = False,
        normalize: bool = True,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        dropout: float = 0.0,
        head_channels: Optional[Sequence[int]] = None,
        head_dropout: float = 0.0,
    ):
        super().__init__(in_channels=in_channels or 3, num_classes=num_classes)
        sa_channels = ensure_msg_list(sa_channels)
        encoder_channels = [self.in_channels, *list(encoder_channels)[1:]]

        self.encoder = PVCNN2Encoder(
            channels=encoder_channels,
            depths=encoder_depths,
            resolutions=encoder_resolutions,
            kernel_sizes=encoder_kernel_sizes,
            sa_channels=sa_channels,
            ratios=ratios,
            radii=radii,
            num_neighbors=num_neighbors,
            use_se=use_se,
            normalize=normalize,
            act=act,
            act_kwargs=act_kwargs,
            act_first=act_first,
            norm=norm,
            norm_kwargs=norm_kwargs,
        )

        num_blocks = len(self.encoder.depths)
        num_decoder_blocks = len(ensure_tuple(decoder_depths))
        if num_decoder_blocks > num_blocks:
            raise ValueError(
                f"Expected at most {num_blocks} decoder blocks (one per encoder block), got {num_decoder_blocks}."
            )

        skip_channels = [int(self.encoder.channels[num_blocks - 1 - i]) for i in range(num_decoder_blocks - 1)]
        skip_channels.append(self.in_channels - 3)

        self.decoder = PVCNN2Decoder(
            in_channels=int(self.encoder.channels[-1]),
            depths=decoder_depths,
            channels=decoder_channels,
            skip_channels=skip_channels,
            fp_channels=fp_channels,
            resolutions=decoder_resolutions,
            kernel_sizes=decoder_kernel_sizes,
            use_se=use_se,
            normalize=normalize,
            act=act,
            act_kwargs=act_kwargs,
            act_first=act_first,
            norm=norm,
            norm_kwargs=norm_kwargs,
        )

        self.dropout = dropout
        self.act = act
        self.act_kwargs = act_kwargs
        self.act_first = act_first
        self.norm = norm
        self.norm_kwargs = norm_kwargs
        self.head_channels = ensure_tuple(head_channels, none_as_empty=True)
        self.head_dropout = head_dropout
        self.head = self.configure_head()

    @property
    def embedding_dim(self) -> int:
        return self.decoder.out_channels

    def configure_head(self) -> nn.Module:
        if not self.head_channels:
            return nn.Identity() if self.num_classes == 0 else nn.Linear(self.embedding_dim, self.num_classes)

        channels = [self.embedding_dim, *self.head_channels, self.num_classes]
        dropout = [self.head_dropout] * (len(channels) - 2) + [0.0]
        return MLP(
            channels,
            act=self.act,
            act_kwargs=self.act_kwargs,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
            dropout=dropout,
            plain_last=True,
        )

    def reset_classifier(self, num_classes: int, **kwargs: Any) -> None:
        self.num_classes = num_classes
        self.head = self.configure_head()

    @overload
    def forward_features(
        self,
        x: OptTensor,
        pos: Tensor,
        batch: Tensor,
        return_intermediates: Literal[True],
    ) -> Tuple[Tensor, Tensor, Tensor, List[Dict[str, Tensor]]]: ...

    @overload
    def forward_features(
        self,
        x: OptTensor,
        pos: Tensor,
        batch: Tensor,
        return_intermediates: Literal[False] = False,
    ) -> Tuple[Tensor, Tensor, Tensor]: ...

    def forward_features(
        self,
        x: OptTensor,
        pos: Tensor,
        batch: Tensor,
        return_intermediates: bool = False,
    ) -> Any:
        x = x if x is not None else pos
        return self.encoder(x, pos, batch, return_intermediates=return_intermediates)

    def forward_decoder(
        self,
        x: Tensor,
        pos: Tensor,
        batch: Tensor,
        intermediates: List[Dict[str, Tensor]],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        # The full-resolution skip is the extra-features slice: the reference feeds inputs[:, 3:]
        # (features without the 3 leading coordinate channels) as the final skip.
        first = dict(intermediates[0])
        first["features"] = first["features"][:, 3:]
        return self.decoder(x, pos, batch, [first, *intermediates[1:]])

    def forward_head(self, x: Tensor, pre_logits: bool = False) -> Tensor:
        if self.dropout and not self.head_channels:
            x = F.dropout(x, p=float(self.dropout), training=self.training)
        return x if pre_logits else self.head(x)

    def forward(self, x: OptTensor, pos: Tensor, batch: Tensor) -> Tensor:
        x, pos, batch, intermediates = self.forward_features(x, pos, batch, return_intermediates=True)
        x, _, _ = self.forward_decoder(x, pos, batch, intermediates)
        return self.forward_head(x)


@register_model(
    "pvcnn2.s3dis-area5",
    task="segmentation",
    # No ported pretrained weights for PVCNN2 yet.
    weights=None,
    hparams=dict(
        in_channels=9,
        num_classes=13,
        # The four SA lists pad to the encoder block count; block 0 has no SA module, so their last entry is unused.
        ratios=[0.125, 0.25, 0.25, 0.25, 0.25],
        radii=[0.1, 0.2, 0.4, 0.8, 0.8],
        num_neighbors=[32, 32, 32, 32, 32],
        sa_channels=[[32, 64], [64, 128], [128, 256], [256, 256, 512], [256, 256, 512]],
        encoder_channels=[9, 32, 64, 128, 256, 512],
        encoder_depths=[2, 3, 3, 0, 0],
        encoder_resolutions=[32, 16, 8, 0, 0],
        encoder_kernel_sizes=[3, 3, 3, 0, 0],
        fp_channels=[[256, 256], [256, 256], [256, 128], [128, 128, 64]],
        decoder_channels=[256, 256, 128, 64],
        decoder_depths=[1, 1, 2, 1],
        decoder_resolutions=[8, 8, 16, 32],
        decoder_kernel_sizes=[3, 3, 3, 3],
        use_se=True,
        normalize=True,
        head_channels=[128],
        head_dropout=0.5,
        act="relu",
    ),
    transform=T.Compose(
        [
            T.Cat(keys=[DataKeys.POS, DataKeys.COLOR, DataKeys.NORM_POS], dst_key=DataKeys.X),
        ]
    ),
)
def pvcnn2_s3dis_area5(**hparams: Any) -> PVCNN2Segmentation:
    r"""Paper-faithful PVCNN++ for S3DIS Area-5 semantic segmentation.

    The generic `PVCNN2Segmentation` uses `act="relu"` and `nn.BatchNorm3d` defaults for both
    branches. The reference training splits them: the voxel branch uses `LeakyReLU(0.1)` and
    `nn.BatchNorm3d` with $\epsilon=10^{-4}$, while the point branch keeps ReLU.
    """
    model = PVCNN2Segmentation(**hparams)
    for pv in model.modules():
        if not isinstance(pv, PVConv):
            continue
        for block in pv.voxel_layers:
            if isinstance(block, Conv3dBlock):
                if isinstance(block.norm, nn.BatchNorm3d):
                    block.norm.eps = 1e-4
                if isinstance(block.act, nn.ReLU):
                    block.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
    return model
