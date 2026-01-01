from functools import partial
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
    overload,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import MLP
from torch_geometric.nn.resolver import normalization_resolver

from torch_pointcloud.layers import PoolLike, create_cls_head
from torch_pointcloud.layers.octree_attention import OctreeAttention, OctreeT
from torch_pointcloud.layers.octree_blocks import OctreeConvBlock
from torch_pointcloud.utils.conversion import ensure_list, ensure_list_size
from torch_pointcloud.utils.imports import optional_import
from torch_pointcloud.utils.types import OptTensor

from ._base import ClassificationModel

if TYPE_CHECKING:
    import dwconv  # pyright: ignore[reportMissingImports]
    import ocnn  # pyright: ignore[reportMissingImports]
    from ocnn.octree import Octree, Points  # pyright: ignore[reportMissingImports]

dwconv, _DWCONV_AVAILABLE = optional_import("dwconv")
ocnn, _ = optional_import("ocnn")
Octree, _ = optional_import("ocnn.octree", "Octree")
Points, _ = optional_import("ocnn.octree", "Points")


class OctFormerIntermediate(NamedTuple):
    x: Tensor
    depth: int


def points_to_octree(points: Points, **kwargs: Any) -> "Octree":
    octree = ocnn.octree.Octree(**kwargs)
    octree.build_octree(points)
    return octree


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
            norm=norm,
            norm_kwargs=norm_kwargs,
            bias=bias,
            dropout=proj_drop,
            plain_last=False,
        )
        self.drop_path = ocnn.nn.OctreeDropPath(drop_path, nempty)
        self.cpe = CPE(channels, nempty=nempty)

    def forward(self, x: Tensor, octree: OctreeT, depth: int) -> Tensor:
        x = self.cpe(x, octree, depth) + x
        attn = self.attention(self.norm1(x), octree, depth)
        x = x + self.drop_path(attn, octree, depth)
        ffn = self.mlp(self.norm2(x))
        x = x + self.drop_path(ffn, octree, depth)
        return x


class OctFormerEncoderBlock(nn.Module):
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
                block = partial(torch.utils.checkpoint.checkpoint, block)

            x = block(x, octree, depth)

        return x


class OctreeConvNorm(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Sequence[int] = (2,),
        stride: int = 2,
        nempty: bool = True,
        use_bias: bool = True,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        norm_kwargs = norm_kwargs or {}
        kernel_size = ensure_list(kernel_size)

        self.conv = ocnn.nn.OctreeConv(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            nempty=nempty,
            use_bias=use_bias,
        )
        self.norm = normalization_resolver(norm, out_channels, **norm_kwargs)

    def forward(self, x: Tensor, octree: Octree, depth: int) -> Tensor:
        x = self.conv(x, octree, depth)
        x = self.norm(x)
        return x


class PatchEmbed(nn.Module):
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
        num_blocks = len(channels) - 2
        if num_blocks <= 0:
            raise ValueError(f"The number of channels must be greater than 2, but got {len(channels)} channels.")

        block_hparams: Dict[str, Any] = dict(
            act=act,
            act_kwargs=act_kwargs,
            act_first=act_first,
            norm=norm,
            norm_kwargs=norm_kwargs,
        )

        self.convs = nn.ModuleList()
        for i in range(num_blocks):
            conv = OctreeConvBlock(
                in_channels=channels[i] if i == 0 else channels[i + 1],
                out_channels=channels[i + 1],
                kernel_size=3,
                stride=1,
                nempty=nempty,
                **block_hparams,
            )
            self.convs.append(conv)

        self.downsamples = nn.ModuleList()
        for i in range(num_blocks):
            downsample = OctreeConvBlock(
                in_channels=channels[i + 1],
                out_channels=channels[i + 2],
                kernel_size=2,
                stride=2,
                nempty=nempty,
                **block_hparams,
            )
            self.downsamples.append(downsample)

        self.proj = OctreeConvBlock(
            in_channels=channels[-1],
            out_channels=channels[-1],
            kernel_size=3,
            stride=1,
            nempty=nempty,
            **block_hparams,
        )

    def forward(self, x: Tensor, octree: Octree, depth: int) -> Tensor:
        if not len(self.convs) == len(self.downsamples):
            raise ValueError(
                "The number of convs and downsamples should be the same, "
                f"but got {len(self.convs)} convs and {len(self.downsamples)} downsample blocks."
            )

        for i, (conv, down) in enumerate(zip(self.convs, self.downsamples)):
            depth_i = depth - i
            # TODO: Downsample first, then conv.
            # TODO: Also, the proj attribute is not necessary
            x = conv(x, octree, depth_i)
            x = down(x, octree, depth_i)

        x = self.proj(x, octree, depth_i - 1)
        return x


class OctFormerEncoder(nn.Module):
    block_name = "stage{i}"

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

        self.blocks = nn.ModuleList()
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

            block = OctFormerEncoderBlock(
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
            # Register the block as a module with the name "stage{i}"
            self.blocks.add_module(self.block_name.format(i=i), block)

    def forward(self, x: Tensor, octree: OctreeT, start_depth: int = 0, return_intermediates: bool = False) -> Tensor:
        max_depth = octree.depth - start_depth
        for i, block in enumerate(self.blocks):
            depth_i = max_depth - i
            x = block(x, octree, depth_i)
        return x


class OctFormerClassification(ClassificationModel):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        *,
        stem_channels: Union[int, Sequence[int]] = (24, 48, 96),
        encoder_channels: Sequence[int] = (96, 192, 384, 384),
        num_blocks: Sequence[int] = (2, 2, 18, 2),
        num_heads: Sequence[int] = (6, 12, 24, 24),
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
        octree_scale_factor: float = 10.24,
        octree_depth: int = 11,
        octree_full_depth: int = 2,
        dropout: float = 0.0,
        global_pool: PoolLike = "max",
    ):
        super().__init__(in_channels=in_channels, num_classes=num_classes)
        self.stem_channels = ensure_list(stem_channels)
        self.encoder_channels = ensure_list(encoder_channels)
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

        self.octree_scale_factor = octree_scale_factor
        self.octree_depth = octree_depth
        self.octree_full_depth = octree_full_depth

        self.stem = self.configure_stem()
        self.encoder = self.configure_encoder()
        self.global_pool = ocnn.nn.OctreeGlobalPool(self.nempty)
        self.dropout = dropout
        self.head = create_cls_head(num_features=self.embedding_dim, num_classes=self.num_classes)

    def configure_stem(self) -> nn.Module:
        return PatchEmbed([self.in_channels, *self.stem_channels], nempty=self.nempty)

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

    @property
    def embedding_dim(self) -> int:
        return self.encoder_channels[-1]

    def reset_classifier(self, num_classes: int, global_pool: PoolLike = "max", **kwargs: Any) -> None:
        self.num_classes = num_classes
        self.global_pool = ocnn.nn.OctreeGlobalPool(self.nempty)
        self.head = create_cls_head(num_features=self.embedding_dim, num_classes=self.num_classes, **kwargs)

    @overload
    def forward_features(
        self,
        x: OptTensor,
        octree: Octree,
        return_intermediates: Literal[True],
    ) -> Tuple[Tensor, List[OctFormerIntermediate]]: ...

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
        depth = octree.depth - len(self.stem_channels)
        x = self.stem(octree.features[octree.depth], octree, octree.depth)

        octree = OctreeT.from_octree(
            octree,
            patch_size=self.patch_size,
            dilation=self.dilation,
            nempty=self.nempty,
        )

        start_depth = depth - len(self.encoder_channels) + 2
        end_depth = depth + 1
        octree.construct_all_attention_context(
            nempty=self.nempty,
            start_depth=start_depth,
            end_depth=end_depth,
        )

        print(f"[OctreeT] {start_depth = }, {end_depth = }")
        return self.encoder(x, octree, 2, return_intermediates=return_intermediates)

    def forward_head(self, x: Tensor, octree: Octree, pre_logits: bool = False) -> Tensor:
        depth = octree.depth - len(self.stem_channels) + 1
        x = self.global_pool(x, octree, depth)
        if self.dropout:
            x = F.dropout(x, p=float(self.dropout), training=self.training)
        return x if pre_logits else self.head(x)

    def forward(self, x: OptTensor, octree: Octree) -> Tensor:
        x = self.forward_features(x, octree, return_intermediates=False)
        return self.forward_head(x, octree)
