"""OctFormer classification and segmentation models.

{{ paper("2305.03045") }}
"""

from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union, overload

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import MLP

import torch_pointcloud.transforms as T
from torch_pointcloud.datasets.modelnet import MODELNET40_CLASSES
from torch_pointcloud.layers import PoolLike, create_pool
from torch_pointcloud.layers.octree_attention import OctreeAttention, OctreeT
from torch_pointcloud.layers.octree_blocks import OctreeConvBlock, OctreeDeconvBlock, _disable_triton
from torch_pointcloud.utils.conversion import ensure_list, ensure_list_size
from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.imports import _DWCONV_GITHUB_URL, _OCNN_GITHUB_URL, optional_import
from torch_pointcloud.utils.octree import octree_interpolate, octree_upsample
from torch_pointcloud.utils.types import OptTensor

from ._base import ClassificationModel, SegmentationModel
from ._registry import WeightsDict, register_model

if TYPE_CHECKING:
    import dwconv
    import ocnn
    from ocnn.octree import Octree, Points

dwconv, _ = optional_import("dwconv", url=_DWCONV_GITHUB_URL)
ocnn, _ = optional_import("ocnn", url=_OCNN_GITHUB_URL)
Octree, _ = optional_import("ocnn.octree", "Octree", url=_OCNN_GITHUB_URL)
Points, _ = optional_import("ocnn.octree", "Points", url=_OCNN_GITHUB_URL)


class CPE(nn.Module):
    """Conditional positional encoding: a depthwise octree convolution followed by batch normalization."""

    def __init__(
        self,
        in_channels: int,
        kernel_size: Union[int, Sequence[int]] = 3,
        nempty: bool = False,
        bias: bool = False,
        use_dwconv: bool = False,
    ):
        super().__init__()
        # OCNN expects the kernel size to be a list, otherwise assertion error will be raised.
        kernel_size = ensure_list(kernel_size)

        if use_dwconv:
            self.conv = dwconv.OctreeDWConv(
                in_channels,
                kernel_size=kernel_size,
                nempty=nempty,
                use_bias=bias,
            )
        else:
            _disable_triton()
            self.conv = ocnn.nn.OctreeGroupConv(
                in_channels,
                in_channels,
                kernel_size=kernel_size,
                nempty=nempty,
                use_bias=bias,
                stride=1,
                group=8,
            )

        self.norm = nn.BatchNorm1d(in_channels)

    def forward(self, x: Tensor, octree: "Octree", depth: int) -> Tensor:
        x = self.conv(x, octree, depth)
        return self.norm(x)


class OctFormerBlock(nn.Module):
    """Transformer block over octree patches: a `CPE` residual, octree attention and an MLP, both pre-normed.

    Note:
        `cpe_first` places the `CPE` residual before the attention; `False` places it after the attention residual.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int,
        patch_size: int = 32,
        dilation: int = 0,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_path: float = 0.0,
        nempty: bool = False,
        use_rpe: bool = True,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: bool = True,
        use_dwconv: bool = False,
        cpe_first: bool = True,
    ):
        super().__init__()
        self.cpe_first = cpe_first
        self.cpe = CPE(channels, kernel_size=3, nempty=nempty, bias=False, use_dwconv=use_dwconv)
        self.norm1 = nn.LayerNorm(channels)
        self.attention = OctreeAttention(
            channels=channels,
            patch_size=patch_size,
            num_heads=num_heads,
            dilation=dilation,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            use_rpe=use_rpe,
        )
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = MLP(
            in_channels=channels,
            hidden_channels=int(channels * mlp_ratio),
            out_channels=channels,
            num_layers=2,
            act=act,
            act_kwargs=act_kwargs,
            act_first=act_first,
            norm=None,
            bias=bias,
            dropout=proj_drop,
            plain_last=True,
        )
        self.drop_path = ocnn.nn.OctreeDropPath(drop_path, nempty)

    def forward(self, x: Tensor, octree: OctreeT, depth: int) -> Tensor:
        if self.cpe_first:
            x = self.cpe(x, octree, depth) + x

        attn = self.attention(self.norm1(x), octree, depth)
        x = x + self.drop_path(attn, octree, depth)

        if not self.cpe_first:
            x = self.cpe(x, octree, depth) + x

        ffn = self.mlp(self.norm2(x))
        x = x + self.drop_path(ffn, octree, depth)
        return x


class OctFormerEncoderLayer(nn.Module):
    """One encoder stage: an optional octree convolution downsampling followed by `num_blocks` `OctFormerBlock` units.

    Blocks alternate between a dilation of `1` and the configured `dilation`, so consecutive blocks
    attend to neighboring and to spread-out patches in turn.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int,
        patch_size: int = 32,
        dilation: int = 0,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_path: Union[float, Sequence[float]] = 0.0,
        nempty: bool = False,
        use_checkpoint: bool = True,
        use_rpe: bool = True,
        use_dwconv: bool = False,
        cpe_first: bool = True,
        num_blocks: int = 2,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: bool = True,
        downsample: Optional[nn.Module] = None,
    ):
        super().__init__()
        drop_path = ensure_list_size(drop_path, size=num_blocks)
        self.use_checkpoint = use_checkpoint
        self.downsample = downsample
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            block = OctFormerBlock(
                channels=channels,
                num_heads=num_heads,
                patch_size=patch_size,
                dilation=1 if (i % 2 == 0) else dilation,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                drop_path=(drop_path[i] if isinstance(drop_path, list) else drop_path),
                nempty=nempty,
                use_rpe=use_rpe,
                use_dwconv=use_dwconv,
                cpe_first=cpe_first,
                act=act,
                act_kwargs=act_kwargs,
                act_first=act_first,
                norm=norm,
                norm_kwargs=norm_kwargs,
                bias=bias,
            )
            self.blocks.append(block)

    def forward(self, x: Tensor, octree: OctreeT, depth: int) -> Tensor:
        if self.downsample is not None:
            x = self.downsample(x, octree, depth + 1)

        block: Union[nn.Module, Callable[[Tensor, OctreeT, int], Tensor]]  # For type hinting with partial
        for block in self.blocks:
            if self.use_checkpoint and self.training:
                block = partial(torch.utils.checkpoint.checkpoint, block, use_reentrant=False)

            x = block(x, octree, depth)

        return x


class OctFormerEncoder(nn.Module):
    """Stack of `OctFormerEncoderLayer` stages, each running one octree depth coarser than the previous one.

    When `return_intermediates=True` is passed to `forward`, the input features of every stage but
    the first are returned in coarse-to-fine order, ready to be consumed by `OctFormerDecoder`.
    """

    def __init__(
        self,
        channels: Sequence[int],
        num_blocks: Sequence[int],
        num_heads: Sequence[int],
        patch_size: int = 26,
        dilation: int = 4,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_path: float = 0.5,
        nempty: bool = False,
        use_checkpoint: bool = True,
        use_rpe: bool = True,
        use_dwconv: bool = False,
        cpe_first: bool = True,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: bool = True,
    ):
        super().__init__()
        self.nempty = nempty
        drop_paths = torch.linspace(0, drop_path, sum(num_blocks)).tolist()

        self.layers = nn.ModuleList()
        for i in range(len(channels)):
            downsample: Optional[nn.Module] = None
            if i > 0:
                downsample = OctreeConvBlock(
                    in_channels=channels[i - 1],
                    out_channels=channels[i],
                    kernel_size=2,
                    stride=2,
                    nempty=nempty,
                    act=None,
                    norm=norm,
                    norm_kwargs=norm_kwargs,
                    bias=bias,
                )

            layer = OctFormerEncoderLayer(
                channels=channels[i],
                num_heads=num_heads[i],
                patch_size=patch_size,
                dilation=dilation,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                drop_path=drop_paths[sum(num_blocks[:i]) : sum(num_blocks[: i + 1])],
                use_rpe=use_rpe,
                use_dwconv=use_dwconv,
                cpe_first=cpe_first,
                nempty=nempty,
                num_blocks=num_blocks[i],
                act=act,
                act_kwargs=act_kwargs,
                act_first=act_first,
                norm=norm,
                norm_kwargs=norm_kwargs,
                bias=bias,
                use_checkpoint=use_checkpoint,
                downsample=downsample,
            )
            self.layers.append(layer)

    @overload
    def forward(
        self,
        x: Tensor,
        octree: OctreeT,
        depth: int,
        return_intermediates: Literal[True] = ...,
    ) -> Tuple[Tensor, List[Tensor]]: ...

    @overload
    def forward(
        self,
        x: Tensor,
        octree: OctreeT,
        depth: int,
        return_intermediates: Literal[False] = False,
    ) -> Tensor: ...

    def forward(self, x: Tensor, octree: OctreeT, depth: int, return_intermediates: bool = False) -> Any:
        intermediates = []
        for i, layer in enumerate(self.layers):
            # Track only intermediate features (i.e. not the input or output, but everything in between)
            if return_intermediates and i > 0:
                intermediates.append(x)

            depth_i = depth - i
            x = layer(x, octree, depth_i)

        if return_intermediates:
            return x, intermediates[::-1]
        return x


class OctFormerDecoder(nn.Module):
    """Feature pyramid decoder: projects every encoder stage to `fpn_channels`, merges them top-down, and
    sums the results upsampled to the finest depth. `num_ups` octree deconvolutions then undo the stem strides.
    """

    def __init__(
        self,
        channels: Sequence[int],
        fpn_channels: int,
        num_ups: int = 1,
        nempty: bool = True,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: bool = True,
    ):
        super().__init__()
        # Default keyword arguments for OctreeConvBlock and OctreeDeconvBlock blocks.
        kwargs: Dict[str, Any] = dict(
            nempty=nempty,
            act=act,
            act_kwargs=act_kwargs,
            act_first=act_first,
            norm=norm,
            norm_kwargs=norm_kwargs,
            bias=False,
        )

        self.nempty = nempty
        self.lins = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        for i in range(len(channels)):
            lin = nn.Linear(channels[i], fpn_channels, bias=bias)
            fpn_block = OctreeConvBlock(fpn_channels, fpn_channels, kernel_size=3, stride=1, **kwargs)
            self.lins.append(lin)
            self.fpn_blocks.append(fpn_block)

        self.up_blocks = nn.ModuleList()
        for i in range(num_ups):
            up_block = OctreeDeconvBlock(fpn_channels, fpn_channels, kernel_size=3, stride=2, **kwargs)
            self.up_blocks.append(up_block)

    def forward(self, x: Tensor, octree: "Octree", depth: int, intermediates: List[Tensor]) -> Tensor:
        # List containing all features from the encoder, from the deepest to the shallowest.
        x_list = [x, *intermediates]
        dst_depth = depth + len(x_list) - 1

        x_fpn: Union[Tensor, float] = 0.0
        for i, x_skip in enumerate(x_list):
            if i > 0:
                x = octree_upsample(x, octree, depth - 1, depth, method="nearest", nempty=self.nempty)

            x = self.lins[i](x) if i == 0 else self.lins[i](x_skip) + x
            x_block = self.fpn_blocks[i](x, octree, depth)
            x_fpn += octree_upsample(x_block, octree, depth, dst_depth, method="nearest", nempty=self.nempty)
            depth += 1

        for i, block in enumerate(self.up_blocks):
            x_fpn = block(x_fpn, octree, dst_depth + i)
        return x_fpn  # type: ignore[return-value]


class OctreePatchEmbed(nn.Module):
    """Convolutional stem that embeds the octree signal, halving the resolution once per channel step."""

    def __init__(
        self,
        channels: Sequence[int],
        nempty: bool = False,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: bool = True,
    ):
        super().__init__()
        channels = ensure_list(channels)
        num_layers = len(channels) - 1

        # Default keyword arguments for OctreeConvBlock blocks.
        kwargs: Dict[str, Any] = dict(
            nempty=nempty,
            act=act,
            act_kwargs=act_kwargs,
            act_first=act_first,
            norm=norm,
            norm_kwargs=norm_kwargs,
            bias=bias,
        )

        self.convs = nn.ModuleList()
        for i in range(num_layers):
            conv = OctreeConvBlock(
                in_channels=channels[i] if i == 0 else channels[i + 1],
                out_channels=channels[i + 1],
                kernel_size=3,
                stride=1,
                **kwargs,
            )
            self.convs.append(conv)

        self.downsamples = nn.ModuleList()
        for i in range(1, num_layers):
            downsample = OctreeConvBlock(
                in_channels=channels[i],
                out_channels=channels[i + 1],
                kernel_size=2,
                stride=2,
                **kwargs,
            )
            self.downsamples.append(downsample)

    def forward(self, x: Tensor, octree: "Octree", depth: int) -> Tensor:
        for i in range(len(self.convs)):
            depth_i = depth - i

            # Apply downsample for all conv blocks except the first one.
            # NOTE: The associated depth for the downsampling block is the depth of the previous block,
            # such that the features are downsampled between block at depth i + 1 -> i (i.e. the depth decreases).
            if i > 0:
                x = self.downsamples[i - 1](x, octree, depth_i + 1)

            x = self.convs[i](x, octree, depth_i)

        return x


class OctFormerClassification(ClassificationModel):
    r"""OctFormer classification model from
    :arxiv: [OctFormer: Octree-based Transformers for 3D Point Clouds](https://arxiv.org/abs/2305.03045)
    by Peng-Shuai Wang.

    An octree convolution stem embeds the input signal, then attention runs over patches of equal
    point count sorted along the octree's space-filling curve, one stage per octree depth. Point
    features are pooled globally after the encoder for classification.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        *,
        stem_channels: Union[int, Sequence[int]],
        encoder_channels: Sequence[int],
        head_channels: Optional[Union[int, Sequence[int]]] = None,
        num_blocks: Sequence[int],
        num_heads: Sequence[int],
        patch_size: int = 26,
        dilation: int = 4,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_path: float = 0.5,
        nempty: bool = True,
        use_checkpoint: bool = True,
        use_rpe: bool = True,
        use_dwconv: bool = False,
        cpe_first: bool = True,
        act: Union[str, Callable, None] = "gelu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        head_act: Union[str, Callable, None] = "gelu",
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: bool = True,
        dropout: float = 0.0,
        global_pool: PoolLike = "mean",
    ):
        in_channels = in_channels if in_channels > 0 else 3
        super().__init__(in_channels=in_channels, num_classes=num_classes)
        self.stem_channels = ensure_list(stem_channels)
        self.encoder_channels = ensure_list(encoder_channels)
        self.head_channels = ensure_list(head_channels, none_as_empty=True)
        self.num_blocks = ensure_list(num_blocks)
        self.num_heads = ensure_list(num_heads)

        self.patch_size = patch_size
        self.dilation = dilation
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop
        self.drop_path = drop_path
        self.nempty = nempty
        self.use_checkpoint = use_checkpoint
        self.use_rpe = use_rpe
        self.use_dwconv = use_dwconv
        self.cpe_first = cpe_first
        self.act = act
        self.act_kwargs = act_kwargs
        self.act_first = act_first
        self.head_act = head_act
        self.norm = norm
        self.norm_kwargs = norm_kwargs
        self.bias = bias

        self.stem = self.configure_stem()
        self.encoder = self.configure_encoder()
        self.global_pool = create_pool(global_pool)
        self.dropout = dropout
        self.head = self.configure_head()

    def configure_stem(self) -> nn.Module:
        """Build the `OctreePatchEmbed` stem."""
        # NOTE: The original OctFormer stem uses hard-coded ReLU activation.
        # For reproducibility, we use ReLU also here.
        return OctreePatchEmbed(
            [self.in_channels, *self.stem_channels],
            nempty=self.nempty,
            act=self.act,
            act_kwargs=self.act_kwargs,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
            bias=False,
        )

    def configure_encoder(self) -> nn.Module:
        """Build the `OctFormerEncoder` backbone."""
        return OctFormerEncoder(
            channels=self.encoder_channels,
            num_blocks=self.num_blocks,
            num_heads=self.num_heads,
            patch_size=self.patch_size,
            dilation=self.dilation,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=self.qkv_bias,
            qk_scale=self.qk_scale,
            attn_drop=self.attn_drop,
            proj_drop=self.proj_drop,
            drop_path=self.drop_path,
            nempty=self.nempty,
            use_checkpoint=self.use_checkpoint,
            use_rpe=self.use_rpe,
            use_dwconv=self.use_dwconv,
            cpe_first=self.cpe_first,
            act=self.act,
            act_kwargs=self.act_kwargs,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
            bias=self.bias,
        )

    @property
    def num_features(self) -> int:
        """Feature dimension $C$ of the encoder output."""
        return self.encoder_channels[-1]

    def configure_head(self) -> nn.Module:
        if self.num_classes == 0:
            return nn.Identity()
        # NOTE: The original OctFormer uses a linear bias only for the last layer, with ReLU activation.
        channels = [self.num_features, *self.head_channels, self.num_classes]
        biases = [False] * max(0, len(channels) - 2) + [True]
        return MLP(
            channels,
            act=self.head_act,
            act_kwargs=self.act_kwargs,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
            bias=biases,
            dropout=self.dropout,
            plain_last=True,
        )

    def reset_classifier(self, num_classes: int, global_pool: Optional[PoolLike] = None, **kwargs: Any) -> None:
        self.num_classes = num_classes
        if global_pool is not None:
            self.global_pool = create_pool(global_pool)
        self.head = self.configure_head()

    @overload
    def forward_features(
        self,
        x: OptTensor,
        octree: "Octree",
        depth: int,
        return_intermediates: Literal[True],
    ) -> Tuple[Tensor, List[Tensor]]: ...

    @overload
    def forward_features(
        self,
        x: OptTensor,
        octree: "Octree",
        depth: int,
        return_intermediates: Literal[False] = False,
    ) -> Tensor: ...

    def forward_features(
        self,
        x: OptTensor,
        octree: "Octree",
        depth: int,
        return_intermediates: bool = False,
    ) -> Any:
        x = x if x is not None else octree.features[octree.depth]
        x = self.stem(x, octree, octree.depth)

        # Precompute the attention context for each stage / depth of the encoder.
        octree_t = OctreeT.from_octree(
            octree,
            patch_size=self.patch_size,
            dilation=self.dilation,
            nempty=self.nempty,
        )

        # While the octree may have more depths, here we only precompute context
        # required at the different depths of the encoder.
        max_depth = self.get_encoder_depth(depth)
        min_depth = self.get_head_depth(depth)
        octree_t.construct_all_attention_context(
            nempty=self.nempty,
            max_depth=max_depth,
            min_depth=min_depth,
        )

        return self.encoder(x, octree_t, max_depth, return_intermediates=return_intermediates)

    def forward_head(self, x: Tensor, octree: "Octree", depth: int, pre_logits: bool = False) -> Tensor:
        batch = octree.batch_id(depth, self.nempty)
        x = self.global_pool(x, batch)
        if self.dropout:
            x = F.dropout(x, p=float(self.dropout), training=self.training)
        return x if pre_logits else self.head(x)

    def forward(self, x: OptTensor, octree: "Octree", depth: int) -> Tensor:
        x = self.forward_features(x, octree, depth, return_intermediates=False)
        min_depth = self.get_head_depth(depth)
        return self.forward_head(x, octree, min_depth)

    def get_encoder_depth(self, depth: int) -> int:
        """Octree depth at which the first encoder stage runs, once the stem downsamplings are accounted for.

        Args:
            depth: Octree depth of the input signal.

        Returns:
            The octree depth of the encoder's first stage.
        """
        stem_depth = len(self.stem_channels) - 1
        return depth - stem_depth

    def get_head_depth(self, depth: int) -> int:
        """Octree depth of the encoder output, which is the depth the head pools over.

        Args:
            depth: Octree depth of the input signal.

        Returns:
            The octree depth of the encoder's last stage.
        """
        max_depth = self.get_encoder_depth(depth)
        encoder_depth = len(self.encoder_channels) - 1
        return max_depth - encoder_depth


class OctFormerSegmentation(SegmentationModel):
    r"""OctFormer segmentation model from
    :arxiv: [OctFormer: Octree-based Transformers for 3D Point Clouds](https://arxiv.org/abs/2305.03045)
    by Peng-Shuai Wang.

    The octree attention encoder is followed by a feature pyramid decoder that merges every stage at
    `fpn_channels` and upsamples back to the input octree depth, then a per-point MLP head.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        *,
        stem_channels: Union[int, Sequence[int]],
        channels: Sequence[int],
        num_blocks: Sequence[int],
        num_heads: Sequence[int],
        head_channels: Optional[Union[int, Sequence[int]]] = None,
        fpn_channels: int,
        patch_size: int = 26,
        dilation: int = 4,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_path: float = 0.5,
        nempty: bool = True,
        use_checkpoint: bool = True,
        use_rpe: bool = True,
        use_dwconv: bool = False,
        cpe_first: bool = True,
        act: Union[str, Callable, None] = "gelu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        head_act: Union[str, Callable, None] = "gelu",
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: bool = True,
        dropout: float = 0.5,
    ):
        in_channels = in_channels if in_channels > 0 else 3
        super().__init__(in_channels=in_channels, num_classes=num_classes)
        self.stem_channels = ensure_list(stem_channels)
        self.channels = ensure_list(channels)
        self.num_blocks = ensure_list(num_blocks)
        self.num_heads = ensure_list(num_heads)
        self.head_channels = ensure_list(head_channels, none_as_empty=True)
        self.fpn_channels = fpn_channels

        self.patch_size = patch_size
        self.dilation = dilation
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop
        self.drop_path = drop_path
        self.nempty = nempty
        self.use_checkpoint = use_checkpoint
        self.use_rpe = use_rpe
        self.use_dwconv = use_dwconv
        self.cpe_first = cpe_first
        self.act = act
        self.act_kwargs = act_kwargs
        self.act_first = act_first
        self.head_act = head_act
        self.norm = norm
        self.norm_kwargs = norm_kwargs
        self.bias = bias

        self.stem = self.configure_stem()
        self.encoder = self.configure_encoder()
        self.decoder = self.configure_decoder()
        self.dropout = dropout
        self.head = self.configure_head()

    def configure_stem(self) -> nn.Module:
        """Build the `OctreePatchEmbed` stem."""
        # NOTE: The original OctFormer stem uses hard-coded ReLU activation.
        return OctreePatchEmbed(
            [self.in_channels, *self.stem_channels],
            nempty=self.nempty,
            act=self.act,
            act_kwargs=self.act_kwargs,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
            bias=False,
        )

    def configure_encoder(self) -> nn.Module:
        """Build the `OctFormerEncoder` backbone."""
        return OctFormerEncoder(
            channels=self.channels,
            num_blocks=self.num_blocks,
            num_heads=self.num_heads,
            patch_size=self.patch_size,
            dilation=self.dilation,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=self.qkv_bias,
            qk_scale=self.qk_scale,
            attn_drop=self.attn_drop,
            proj_drop=self.proj_drop,
            drop_path=self.drop_path,
            nempty=self.nempty,
            use_checkpoint=self.use_checkpoint,
            use_rpe=self.use_rpe,
            use_dwconv=self.use_dwconv,
            cpe_first=self.cpe_first,
            act=self.act,
            act_kwargs=self.act_kwargs,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
            bias=self.bias,
        )

    def configure_decoder(self) -> nn.Module:
        """Build the `OctFormerDecoder`, with one upsampling per stem downsampling."""
        # The original OctFormer decoder uses hard-coded ReLU activation.
        num_ups = len(self.stem_channels) - 1
        return OctFormerDecoder(
            channels=self.channels[::-1],
            fpn_channels=self.fpn_channels,
            num_ups=num_ups,
            nempty=self.nempty,
            act=self.act,
            act_kwargs=self.act_kwargs,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
            bias=self.bias,
        )

    @property
    def num_features(self) -> int:
        """Feature dimension $C$ of the features entering the head."""
        return self.fpn_channels

    def configure_head(self) -> nn.Module:
        if self.num_classes == 0:
            return nn.Identity()
        channels = [self.num_features, *self.head_channels, self.num_classes]
        return MLP(
            channels,
            act=self.head_act,
            act_kwargs=self.act_kwargs,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
            bias=self.bias,
            plain_last=True,
        )

    def reset_classifier(self, num_classes: int, **kwargs: Any) -> None:
        self.num_classes = num_classes
        self.head = self.configure_head()

    @overload
    def forward_features(
        self,
        x: OptTensor,
        octree: "Octree",
        depth: int,
        return_intermediates: Literal[True],
    ) -> Tuple[Tensor, List[Tensor]]: ...

    @overload
    def forward_features(
        self,
        x: OptTensor,
        octree: "Octree",
        depth: int,
        return_intermediates: Literal[False] = False,
    ) -> Tensor: ...

    def forward_features(
        self,
        x: OptTensor,
        octree: "Octree",
        depth: int,
        return_intermediates: bool = False,
    ) -> Any:
        x = x if x is not None else octree.features[octree.depth]
        x = self.stem(x, octree, octree.depth)

        # Precompute the attention context for each stage / depth of the encoder.
        octree_t = OctreeT.from_octree(
            octree,
            patch_size=self.patch_size,
            dilation=self.dilation,
            nempty=self.nempty,
        )

        # While the octree may have more depths, here we only precompute context
        # required at the different depths of the encoder.
        stem_depth = len(self.stem_channels) - 1
        encoder_depth = len(self.channels) - 1
        max_depth = depth - stem_depth
        min_depth = max_depth - encoder_depth
        octree_t.construct_all_attention_context(
            nempty=self.nempty,
            min_depth=min_depth,
            max_depth=max_depth,
        )

        return self.encoder(x, octree_t, max_depth, return_intermediates=return_intermediates)

    def forward_decoder(self, x: Tensor, octree: "Octree", depth: int, intermediates: List[Tensor]) -> Tensor:
        stem_depth = len(self.stem_channels) - 1
        encoder_depth = len(self.channels) - 1
        max_depth = depth - stem_depth
        min_depth = max_depth - encoder_depth
        return self.decoder(x, octree, min_depth, intermediates)

    def forward_head(
        self,
        x: Tensor,
        octree: "Octree",
        depth: int,
        pos: Tensor,
        batch: Tensor,
        pre_logits: bool = False,
    ) -> Tensor:
        # We need to convert the octree features back to points resolution,
        # the destination points are expected to be in the format $(x, y, z, batch)$.
        pts = torch.cat([pos, batch.unsqueeze(-1)], dim=1).contiguous()
        x = octree_interpolate(x, octree, depth, pts, method="nearest", nempty=self.nempty)
        if self.dropout:
            x = F.dropout(x, p=float(self.dropout), training=self.training)
        return x if pre_logits else self.head(x)

    def forward(self, x: OptTensor, octree: "Octree", depth: int, pos: Tensor, batch: Tensor) -> Tensor:
        x, intermediates = self.forward_features(x, octree, depth, return_intermediates=True)
        x = self.forward_decoder(x, octree, depth, intermediates)
        return self.forward_head(x, octree, depth, pos, batch)


def _octformer_base_clf(**hparams: Any) -> OctFormerClassification:
    # The original OctFormer for classification uses a hard-coded ReLU activation in the stem and head.
    # The head gets it via `head_act` (so `reset_classifier` keeps it); the stem is overridden manually.
    hparams.setdefault("head_act", "relu")
    model = OctFormerClassification(**hparams)

    for name, _ in model.stem.named_modules():
        if name.endswith(".act"):
            model.set_submodule(f"stem.{name}", nn.ReLU(inplace=True))

    return model


def _octformer_base_seg(**hparams: Any) -> OctFormerSegmentation:
    # The original OctFormer for segmentation uses a hard-coded ReLU activation in the stem, decoder and head.
    # The head gets it via `head_act` (so `reset_classifier` keeps it); stem and decoder are overridden manually.
    hparams.setdefault("head_act", "relu")
    model = OctFormerSegmentation(**hparams)

    for name, _ in model.stem.named_modules():
        if name.endswith(".act"):
            model.set_submodule(f"stem.{name}", nn.ReLU(inplace=True))

    for name, _ in model.decoder.named_modules():
        if name.endswith(".act"):
            model.set_submodule(f"decoder.{name}", nn.ReLU(inplace=True))

    return model


@register_model(
    name="octformer-base.modelnet40.octree-nn",
    task="classification",
    weights=WeightsDict(
        url="hf://torch-pointcloud/octformer-base.modelnet40.octree-nn/resolve/main/model.safetensors",
        dataset="modelnet40",
        metrics={"OA": 92.02},
        classes=MODELNET40_CLASSES,
        author="octree-nn",
        license="MIT",
    ),
    hparams=dict(
        in_channels=4,
        num_classes=40,
        stem_channels=(24, 48, 96),
        encoder_channels=(96, 192),
        head_channels=(256,),
        num_blocks=(6, 6),
        num_heads=(6, 12),
        patch_size=32,
        dilation=2,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.5,
        nempty=False,
        use_checkpoint=True,
        use_rpe=True,
        use_dwconv=True,
        cpe_first=False,
        act="gelu",
        act_kwargs=None,
        act_first=False,
        norm="batch_norm",
        norm_kwargs=None,
        bias=True,
        dropout=0.5,
        global_pool="mean",
    ),
    transform=T.Compose(
        [
            T.RandomSampleFaceVertices(
                keys=DataKeys.POS,
                face_key=DataKeys.FACE,
                normal_key=DataKeys.NORMAL,
                num_samples=8000,
            ),
            T.Shift(keys=DataKeys.POS, method="bbox"),
            T.Rescale(keys=DataKeys.POS, method="bbox"),
            T.Rescale(keys=DataKeys.POS, method="min_sphere"),
            # NOTE: Original OctFormer uses inbox masking to remove outliers, but at the cost of performance drop,
            # we found that removing this step improves performance by ~1% on ModelNet40.
            # T.BoxMask(keys=DataKeys.POS, bbox=(-0.99, -0.99, -0.99, 0.99, 0.99, 0.99), dst_keys=DataKeys.BOX_MASK),
            # T.ApplyMask(keys=[DataKeys.POS, DataKeys.NORMAL], mask_key=DataKeys.BOX_MASK),
            T.Abs(keys=DataKeys.NORMAL),
            T.ToTensor(keys=[DataKeys.POS, DataKeys.NORMAL], dtype=torch.float32),
            T.BuildOctree(
                pos_key=DataKeys.POS,
                octree_key=DataKeys.OCTREE,
                depth=6,
                full_depth=2,
                batch_size=1,
                normal_key=DataKeys.NORMAL,
            ),
            T.OctreeFeatures(
                keys=DataKeys.OCTREE,
                features_type="ND",
                nempty=False,
                dst_keys=DataKeys.X,
            ),
        ]
    ),
)
def octformer_base_modelnet40_clf(**hparams: Any) -> OctFormerClassification:
    return _octformer_base_clf(**hparams)


@register_model(
    name="octformer-base.scannet20.octree-nn",
    weights=WeightsDict(
        url="hf://torch-pointcloud/octformer-base.scannet20.octree-nn/resolve/main/model.safetensors",
        dataset="scannet20",
        metrics={"mIoU": 74.78},
        author="octree-nn",
        license="MIT",
    ),
    task="segmentation",
    hparams=dict(
        in_channels=10,
        num_classes=21,
        stem_channels=(24, 48, 96),
        channels=(96, 192, 384, 384),
        num_blocks=(2, 2, 18, 2),
        num_heads=(6, 12, 24, 24),
        head_channels=168,
        fpn_channels=168,
        patch_size=32,
        dilation=4,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.5,
        nempty=True,
        use_checkpoint=True,
        use_rpe=True,
        use_dwconv=True,
        cpe_first=True,
        act="gelu",
        act_kwargs=None,
        act_first=False,
        norm="batch_norm",
        norm_kwargs=None,
        bias=True,
        dropout=0.5,
    ),
    transform=T.Compose(
        [
            T.Shift(keys=DataKeys.POS, method="bbox"),
            T.Divide(keys=[DataKeys.POS, DataKeys.COLOR], divisor=[10.24, 255]),
            T.AlignAxis(keys=DataKeys.POS, dim=-1),
            T.BuildOctree(
                pos_key=DataKeys.POS,
                normal_key=DataKeys.NORMAL,
                feature_key=DataKeys.COLOR,
                label_key=DataKeys.SEGMENT,
                points_key=DataKeys.POINTS,
                octree_key=DataKeys.OCTREE,
                depth=11,
                full_depth=2,
                batch_size=1,
            ),
            T.OctreeFeatures(
                keys=DataKeys.OCTREE,
                features_type="NDFP",
                nempty=True,
                dst_keys=DataKeys.X,
            ),
        ]
    ),
)
def octformer_base_scannet_seg(**hparams: Any) -> OctFormerSegmentation:
    return _octformer_base_seg(**hparams)


@register_model(
    name="octformer-base.scannet200.octree-nn",
    weights=WeightsDict(
        url="hf://torch-pointcloud/octformer-base.scannet200.octree-nn/resolve/main/model.safetensors",
        dataset="scannet200",
        metrics={"mIoU": 31.71},
        author="octree-nn",
        license="MIT",
    ),
    task="segmentation",
    hparams=dict(
        in_channels=10,
        num_classes=201,
        stem_channels=(24, 48, 96),
        channels=(96, 192, 384, 384),
        num_blocks=(2, 2, 18, 2),
        num_heads=(6, 12, 24, 24),
        head_channels=168,
        fpn_channels=168,
        patch_size=32,
        dilation=4,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.5,
        nempty=True,
        use_checkpoint=True,
        use_rpe=True,
        use_dwconv=True,
        cpe_first=True,
        act="gelu",
        act_kwargs=None,
        act_first=False,
        norm="batch_norm",
        norm_kwargs=None,
        bias=True,
        dropout=0.5,
    ),
    transform=T.Compose(
        [
            T.Shift(keys=DataKeys.POS, method="bbox"),
            T.Divide(keys=[DataKeys.POS, DataKeys.COLOR], divisor=[10.24, 255]),
            T.AlignAxis(keys=DataKeys.POS, dim=-1),
            T.BuildOctree(
                pos_key=DataKeys.POS,
                normal_key=DataKeys.NORMAL,
                feature_key=DataKeys.COLOR,
                label_key=DataKeys.SEGMENT,
                points_key=DataKeys.POINTS,
                octree_key=DataKeys.OCTREE,
                depth=11,
                full_depth=2,
                batch_size=1,
            ),
            T.OctreeFeatures(
                keys=DataKeys.OCTREE,
                features_type="NDFP",
                nempty=True,
                dst_keys=DataKeys.X,
            ),
        ]
    ),
)
def octformer_base_scannet200_seg(**hparams: Any) -> OctFormerSegmentation:
    return _octformer_base_seg(**hparams)


@register_model(
    name="octformer-lg",
    task="segmentation",
    hparams=dict(
        stem_channels=(24, 48, 96),
        channels=(192, 384, 768, 768),
        num_blocks=(2, 2, 18, 2),
        num_heads=(12, 24, 48, 48),
        head_channels=168,
        fpn_channels=168,
        patch_size=32,
        dilation=4,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.5,
        nempty=True,
        use_checkpoint=True,
        use_rpe=True,
        use_dwconv=True,
        cpe_first=True,
        act="gelu",
        act_kwargs=None,
        act_first=False,
        norm="batch_norm",
        norm_kwargs=None,
        bias=True,
        dropout=0.5,
    ),
)
def octformer_lg_seg(**hparams: Any) -> OctFormerSegmentation:
    return _octformer_base_seg(**hparams)


@register_model(
    name="octformer-sm",
    task="segmentation",
    hparams=dict(
        stem_channels=(24, 48, 96),
        channels=(96, 192, 384, 384),
        num_blocks=(2, 2, 6, 2),
        num_heads=(6, 12, 24, 24),
        head_channels=168,
        fpn_channels=168,
        patch_size=32,
        dilation=4,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.5,
        nempty=True,
        use_checkpoint=True,
        use_rpe=True,
        use_dwconv=True,
        cpe_first=True,
        act="gelu",
        act_kwargs=None,
        act_first=False,
        norm="batch_norm",
        norm_kwargs=None,
        bias=True,
        dropout=0.5,
    ),
)
def octformer_sm_seg(**hparams: Any) -> OctFormerSegmentation:
    return _octformer_base_seg(**hparams)
