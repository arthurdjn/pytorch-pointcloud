from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union, overload

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import MLP

from torch_pointcloud.layers import PoolLike, create_cls_head
from torch_pointcloud.layers.layer_container import LayerContainer
from torch_pointcloud.layers.octree_attention import OctreeAttention, OctreeT
from torch_pointcloud.layers.octree_blocks import OctreeConvBlock, OctreeDeconvBlock
from torch_pointcloud.utils.conversion import ensure_list, ensure_list_size
from torch_pointcloud.utils.imports import optional_import
from torch_pointcloud.utils.types import OptTensor

from ._base import ClassificationModel, SegmentationModel
from ._registry import register_model

if TYPE_CHECKING:
    import dwconv  # pyright: ignore[reportMissingImports]
    import ocnn  # pyright: ignore[reportMissingImports]
    from ocnn.octree import Octree, Points  # pyright: ignore[reportMissingImports]

dwconv, _DWCONV_AVAILABLE = optional_import("dwconv")
ocnn, _ = optional_import("ocnn")
Octree, _ = optional_import("ocnn.octree", "Octree")
Points, _ = optional_import("ocnn.octree", "Points")


class CPE(nn.Module):
    def __init__(
        self,
        in_channels: int,
        kernel_size: Union[int, Sequence[int]] = 3,
        stride: int = 1,
        nempty: bool = False,
        use_bias: bool = False,
        group: int = 8,
        use_dwconv: bool = False,
    ):
        super().__init__()
        # OCNN expects the kernel size to be a list, otherwise assertion error will be raised.
        kernel_size = ensure_list(kernel_size)

        if use_dwconv:
            self.conv = dwconv.OctreeDWConv(in_channels, kernel_size, nempty, use_bias)
        else:
            self.conv = ocnn.nn.OctreeGroupConv(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                stride=stride,
                nempty=nempty,
                use_bias=use_bias,
                group=group,
            )

        self.norm = nn.BatchNorm1d(in_channels)

    def forward(self, x: Tensor, octree: Octree, depth: int) -> Tensor:
        x = self.conv(x, octree, depth)
        return self.norm(x)


class OctFormerBlock(nn.Module):
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
        nempty: bool = True,
        use_rpe: bool = True,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: bool = True,
        **kwargs: Any,
    ):
        super().__init__()
        self.cpe = CPE(channels, nempty=nempty)
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
        x = self.cpe(x, octree, depth) + x
        attn = self.attention(self.norm1(x), octree, depth)
        x = x + self.drop_path(attn, octree, depth)
        ffn = self.mlp(self.norm2(x))
        x = x + self.drop_path(ffn, octree, depth)
        return x


class OctFormerEncoderLayer(nn.Module):
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
        nempty: bool = True,
        use_checkpoint: bool = True,
        use_rpe: bool = True,
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

        for block in self.blocks:
            if self.use_checkpoint and self.training:
                block = partial(torch.utils.checkpoint.checkpoint, block, use_reentrant=False)

            x = block(x, octree, depth)

        return x


class OctFormerEncoder(LayerContainer):
    layer_name = "layer"

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
        nempty: bool = True,
        use_checkpoint: bool = True,
        use_rpe: bool = True,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: bool = True,
    ):
        super().__init__()
        drop_paths = torch.linspace(0, drop_path, sum(num_blocks)).tolist()

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
            self.add_layer(layer)

    @overload
    def forward(
        self,
        x: Tensor,
        octree: OctreeT,
        start_depth: int = 0,
        return_intermediates: Literal[True] = ...,
    ) -> Tuple[Tensor, List[Tensor]]: ...

    @overload
    def forward(
        self,
        x: Tensor,
        octree: OctreeT,
        start_depth: int = 0,
        return_intermediates: Literal[False] = False,
    ) -> Tensor: ...

    def forward(self, x: Tensor, octree: OctreeT, start_depth: int = 0, return_intermediates: bool = False) -> Any:
        max_depth = octree.depth - start_depth
        intermediates = []
        for i, layer in enumerate(self.iter_layers()):
            if return_intermediates and i > 0:
                # Track only intermediate features (i.e. not the input or output, but everything in between)
                intermediates.append(x)

            depth_i = max_depth - i
            x = layer(x, octree, depth_i)

        if return_intermediates:
            return x, intermediates[::-1]
        return x


class OctFormerDecoderLayer(nn.Module):
    def __init__(
        self,
        channels: int,
        fpn_channels: int,
        nempty: bool,
        head_up: int = 1,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: bool = True,
        upsample: Optional[nn.Module] = None,
    ):
        super().__init__()

        self.upsample = upsample
        self.lin = nn.Linear(channels, fpn_channels, bias=bias)
        self.conv = OctreeDeconvBlock(
            fpn_channels,
            fpn_channels,
            kernel_size=3,
            stride=1,
            nempty=nempty,
            act=act,
            act_kwargs=act_kwargs,
            act_first=act_first,
            norm=norm,
            norm_kwargs=norm_kwargs,
            bias=bias,
        )

    def forward(self, x: Tensor, x_skip: Tensor, octree: Octree, depth: int) -> Tensor:
        if self.upsample is not None:
            x = self.upsample(x, octree, depth - 1)

        # If the upsample is defined, use a residual connection
        x = self.lin(x_skip) if self.upsample is None else self.lin(x_skip) + x
        x = self.conv(x, octree, depth)
        return x


class OctFormerDecoder(LayerContainer):
    layer_name = "layer"

    def __init__(
        self,
        channels: Sequence[int],
        fpn_channels: int,
        nempty: bool,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: bool = True,
    ):
        super().__init__()
        self.upsample = ocnn.nn.OctreeUpsample("nearest", nempty)
        for i in range(len(channels)):
            upsample = ocnn.nn.OctreeUpsample("nearest", nempty) if i > 0 else None
            layer = OctFormerDecoderLayer(
                channels=channels[i],
                fpn_channels=fpn_channels,
                nempty=nempty,
                act=act,
                act_kwargs=act_kwargs,
                act_first=act_first,
                norm=norm,
                norm_kwargs=norm_kwargs,
                bias=bias,
                upsample=upsample,
            )
            self.add_layer(layer)

    def forward(self, x: Tensor, octree: Octree, depth: int, intermediates: List[Tensor]) -> Tensor:
        # List containing all features from the encoder, from the deepest to the shallowest.
        x_list = [x, *intermediates]
        min_depth = depth - len(x_list) + 1

        x_fpn = 0
        for i, (layer, x_skip) in enumerate(zip(self.iter_layers(), x_list)):
            depth_i = min_depth + i
            x = layer(x, x_skip, octree, depth_i)
            x_fpn += self.upsample(x, octree, depth_i, depth)

        return x_fpn  # type: ignore[return-value]


class OctreePatchEmbed(nn.Module):
    def __init__(
        self,
        channels: Sequence[int],
        nempty: bool = True,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: bool = True,
    ):
        super().__init__()
        channels = ensure_list(channels)
        self.num_layers = len(channels) - 1
        if self.num_layers <= 1:
            raise ValueError(
                f"The number of layers must be greater than 1, but got {self.num_layers} layers. "
                f"Make sure to increase the number of channels, got {channels} channels."
            )

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
        for i in range(self.num_layers):
            conv = OctreeConvBlock(
                in_channels=channels[i] if i == 0 else channels[i + 1],
                out_channels=channels[i + 1],
                kernel_size=3,
                stride=1,
                **kwargs,
            )
            self.convs.append(conv)

        self.downsamples = nn.ModuleList()
        for i in range(1, self.num_layers):
            downsample = OctreeConvBlock(
                in_channels=channels[i],
                out_channels=channels[i + 1],
                kernel_size=2,
                stride=2,
                **kwargs,
            )
            self.downsamples.append(downsample)

    def forward(self, x: Tensor, octree: Octree, depth: int) -> Tensor:
        for i in range(self.num_layers):
            # Decrease the depth depending the deeper the block is.
            depth_i = depth - i

            # Apply downsample for all conv blocks except the first one.
            # NOTE: The associated depth for the downsampling block is the depth of the previous block,
            # such that the features are downsampled between block at depth i + 1 -> i (i.e. the depth decreases).
            if i > 0:
                x = self.downsamples[i - 1](x, octree, depth_i + 1)

            x = self.convs[i](x, octree, depth_i)

        return x


class OctFormerClassification(ClassificationModel):
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
        act: Union[str, Callable, None] = "gelu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: bool = True,
        dropout: float = 0.0,
        global_pool: PoolLike = "mean",
    ):
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
        self.act = act
        self.act_kwargs = act_kwargs
        self.act_first = act_first
        self.norm = norm
        self.norm_kwargs = norm_kwargs
        self.bias = bias

        self.stem = self.configure_stem()
        self.encoder = self.configure_encoder()
        self.global_pool = ocnn.nn.OctreeGlobalPool(self.nempty)
        self.dropout = dropout
        self.head = self.configure_head()

    def configure_stem(self) -> nn.Module:
        # NOTE: The original OctFormer stem uses hard-coded ReLU activation.
        # For reproducibility, we use ReLU also here.
        return OctreePatchEmbed(
            [self.in_channels, *self.stem_channels],
            nempty=self.nempty,
            act="relu",
            act_kwargs=None,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
            bias=False,
        )

    def configure_encoder(self) -> nn.Module:
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
            act=self.act,
            act_kwargs=self.act_kwargs,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
            bias=self.bias,
        )

    def configure_head(self) -> nn.Module:
        # NOTE: The original OctFormer uses a linear bias only for the last layer, with ReLU activation.
        channels = [self.embedding_dim, *self.head_channels, self.num_classes]
        biases = [False] * max(0, len(channels) - 2) + [True]
        return MLP(
            channels,
            act="relu",
            act_kwargs=None,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
            bias=biases,
            dropout=self.dropout,
            plain_last=True,
        )

    @property
    def embedding_dim(self) -> int:
        return self.encoder_channels[-1]

    def reset_classifier(self, num_classes: int, global_pool: PoolLike = "mean", **kwargs: Any) -> None:
        # TODO: Allow changing the global pooling method.
        self.num_classes = num_classes
        self.head = self.configure_head()

    @overload
    def forward_features(
        self,
        x: OptTensor,
        octree: Octree,
        return_intermediates: Literal[True],
    ) -> Tuple[Tensor, List[Tensor]]: ...

    @overload
    def forward_features(
        self,
        x: OptTensor,
        octree: Octree,
        return_intermediates: Literal[False] = False,
    ) -> Tensor: ...

    def forward_features(
        self,
        x: OptTensor,
        octree: Octree,
        return_intermediates: bool = False,
    ) -> Any:
        x = self.stem(octree.features[octree.depth], octree, octree.depth)
        octree_t = OctreeT.from_octree(
            octree,
            patch_size=self.patch_size,
            dilation=self.dilation,
            nempty=self.nempty,
        )

        # Precompute the attention context for each stage / depth of the encoder.
        # While the octree may have more depths, here we only precompute context
        # required at the different depths of the encoder.
        stem_depth = len(self.stem_channels) - 1
        encoder_depth = len(self.encoder_channels) - 1
        max_depth = octree_t.depth - stem_depth
        min_depth = max_depth - encoder_depth
        octree_t.construct_all_attention_context(
            nempty=self.nempty,
            min_depth=min_depth,
            max_depth=max_depth,
        )

        return self.encoder(x, octree_t, stem_depth, return_intermediates=return_intermediates)

    def forward_head(self, x: Tensor, octree: Octree, pre_logits: bool = False) -> Tensor:
        stem_depth = len(self.stem_channels) - 1
        encoder_depth = len(self.encoder_channels) - 1
        max_depth = octree.depth - stem_depth
        min_depth = max_depth - encoder_depth

        x = self.global_pool(x, octree, min_depth)
        if self.dropout:
            x = F.dropout(x, p=float(self.dropout), training=self.training)
        return x if pre_logits else self.head(x)

    def forward(self, x: OptTensor, octree: Octree) -> Tensor:
        x = self.forward_features(x, octree, return_intermediates=False)
        return self.forward_head(x, octree)


class OctFormerSegmentation(SegmentationModel):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        *,
        stem_channels: Union[int, Sequence[int]],
        encoder_channels: Sequence[int],
        encoder_num_blocks: Sequence[int],
        encoder_num_heads: Sequence[int],
        decoder_channels: Sequence[int],
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
        act: Union[str, Callable, None] = "gelu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: bool = True,
    ):
        super().__init__(in_channels=in_channels, num_classes=num_classes)
        self.stem_channels = ensure_list(stem_channels)
        self.encoder_channels = ensure_list(encoder_channels)
        self.encoder_num_blocks = ensure_list(encoder_num_blocks)
        self.encoder_num_heads = ensure_list(encoder_num_heads)
        self.decoder_channels = ensure_list(decoder_channels)
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
        self.act = act
        self.act_kwargs = act_kwargs
        self.act_first = act_first
        self.norm = norm
        self.norm_kwargs = norm_kwargs
        self.bias = bias

        self.stem = self.configure_stem()
        self.encoder = self.configure_encoder()
        self.decoder = self.configure_decoder()
        self.head = self.configure_head()

    def configure_stem(self) -> nn.Module:
        # NOTE: The original OctFormer stem uses hard-coded ReLU activation.
        # For reproducibility, we use ReLU also here.
        return OctreePatchEmbed(
            [self.in_channels, *self.stem_channels],
            nempty=self.nempty,
            act="relu",
            act_kwargs=None,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
            bias=False,
        )

    def configure_encoder(self) -> nn.Module:
        return OctFormerEncoder(
            channels=self.encoder_channels,
            num_blocks=self.encoder_num_blocks,
            num_heads=self.encoder_num_heads,
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
            act=self.act,
            act_kwargs=self.act_kwargs,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
            bias=self.bias,
        )

    def configure_decoder(self) -> nn.Module:
        return OctFormerDecoder(
            channels=self.decoder_channels,
            fpn_channels=self.fpn_channels,
            nempty=self.nempty,
            act=self.act,
            act_kwargs=self.act_kwargs,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
            bias=self.bias,
        )

    def configure_head(self) -> nn.Module:
        return create_cls_head(num_features=self.embedding_dim, num_classes=self.num_classes)

    @property
    def embedding_dim(self) -> int:
        return self.fpn_channels

    def reset_classifier(self, num_classes: int, **kwargs: Any) -> None:
        self.num_classes = num_classes
        self.head = create_cls_head(num_features=self.embedding_dim, num_classes=self.num_classes, **kwargs)

    @overload
    def forward_features(
        self,
        x: OptTensor,
        octree: Octree,
        return_intermediates: Literal[True],
    ) -> Tuple[Tensor, List[Tensor]]: ...

    @overload
    def forward_features(
        self,
        x: OptTensor,
        octree: Octree,
        return_intermediates: Literal[False] = False,
    ) -> Tensor: ...

    def forward_features(
        self,
        x: OptTensor,
        octree: Octree,
        return_intermediates: bool = False,
    ) -> Any:
        x = self.stem(octree.features[octree.depth], octree, octree.depth)

        octree_t = OctreeT.from_octree(
            octree,
            patch_size=self.patch_size,
            dilation=self.dilation,
            nempty=self.nempty,
        )

        # Precompute the attention context for each stage / depth of the encoder.
        # While the octree may have more depths, here we only precompute context
        # required at the different depths of the encoder.
        stem_depth = len(self.stem_channels) - 1
        encoder_depth = len(self.encoder_channels) - 1
        max_depth = octree_t.depth - stem_depth
        min_depth = max_depth - encoder_depth
        octree_t.construct_all_attention_context(
            nempty=self.nempty,
            min_depth=min_depth,
            max_depth=max_depth,
        )

        return self.encoder(x, octree_t, stem_depth, return_intermediates=return_intermediates)

    def forward_decoder(self, x: Tensor, octree: Octree, intermediates: List[Tensor]) -> Tensor:
        stem_depth = len(self.stem_channels) - 1
        max_depth = octree.depth - stem_depth

        return self.decoder(x, octree, max_depth, intermediates)

    def forward_head(self, x: Tensor, pre_logits: bool = False) -> Tensor:
        return x if pre_logits else self.head(x)

    def forward(self, x: OptTensor, octree: Octree) -> Tensor:
        x, intermediates = self.forward_features(x, octree, return_intermediates=True)
        x = self.forward_decoder(x, octree, intermediates)
        return self.forward_head(x)


@register_model(name="octformer-base", task="classification")
def octformer_base_clf(in_channels: int, num_classes: int, **kwargs: Any) -> OctFormerClassification:
    hparams: Dict[str, Any] = dict(
        in_channels=in_channels,
        num_classes=num_classes,
        stem_channels=(24, 48, 96),
        encoder_channels=(96, 192, 384, 384),
        head_channels=256,
        num_blocks=(2, 2, 18, 2),
        num_heads=(6, 12, 24, 24),
        patch_size=26,
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
        act="gelu",
        act_kwargs=None,
        act_first=False,
        norm="batch_norm",
        norm_kwargs=None,
        bias=True,
        dropout=0.5,
        global_pool="max",
    )
    hparams.update(kwargs)
    return OctFormerClassification(**hparams)


@register_model(name="octformer-sm", task="classification")
def octformer_sm_clf(in_channels: int, num_classes: int, **kwargs: Any) -> OctFormerClassification:
    hparams: Dict[str, Any] = dict(
        in_channels=in_channels,
        num_classes=num_classes,
        stem_channels=(24, 48, 96),
        encoder_channels=(96, 192, 384, 384),
        head_channels=256,
        num_blocks=(2, 2, 6, 2),
        num_heads=(6, 12, 24, 24),
        patch_size=26,
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
        act="gelu",
        act_kwargs=None,
        act_first=False,
        norm="batch_norm",
        norm_kwargs=None,
        bias=True,
        dropout=0.5,
        global_pool="max",
    )
    hparams.update(kwargs)
    return OctFormerClassification(**hparams)


@register_model(name="octformer-base", task="segmentation")
def octformer_base_seg(in_channels: int, num_classes: int, **kwargs: Any) -> OctFormerSegmentation:
    hparams: Dict[str, Any] = dict(
        in_channels=in_channels,
        num_classes=num_classes,
        stem_channels=(24, 48, 96),
        encoder_channels=(96, 192, 384, 384),
        encoder_num_blocks=(2, 2, 18, 2),
        encoder_num_heads=(6, 12, 24, 24),
        decoder_channels=(384, 384, 192, 96),
        fpn_channels=168,
    )
    hparams.update(kwargs)
    return OctFormerSegmentation(**hparams)
