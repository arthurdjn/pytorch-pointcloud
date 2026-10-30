"""Volt (Volume Transformer) segmentation model.

{{ paper("2604.19609") }}
"""

import math
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union, overload

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import MLP
from torch_geometric.nn import Linear as PyGLinear

import torch_pointcloud.transforms as T
from torch_pointcloud.datasets.nuscenes import NUSCENES_LIDARSEG_CLASSES, NUSCENES_LIDARSEG_LEARNING_MAP
from torch_pointcloud.datasets.scannet import SCANNET20_CLASSES, SCANNET200_CLASSES
from torch_pointcloud.datasets.semantickitti import SEMANTIC_KITTI_CLASSES, SEMANTIC_KITTI_LEARNING_MAP
from torch_pointcloud.layers import AxialRoPE, PackedTransformerBlock, VoxelPatchEmbed, VoxelPatchExpand
from torch_pointcloud.layers.act import create_act
from torch_pointcloud.layers.norms import create_norm
from torch_pointcloud.models._base import SemanticSegmentationModel
from torch_pointcloud.models._registry import WeightsDict, register_model
from torch_pointcloud.ops.voxelization import voxel_patch_partition
from torch_pointcloud.utils.conversion import batch_to_cu_seqlens
from torch_pointcloud.utils.data import DataKeys


def _init_volt_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Linear, PyGLinear)):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class VoltEncoder(nn.Module):
    r"""Patch tokenizer and plain transformer encoder of Volt.

    Groups the voxels into non-overlapping patches of $K^3$ voxels (`voxel_patch_partition`), embeds every patch into
    one token (`VoxelPatchEmbed`) and runs `depth` blocks of global attention over the tokens of each sample
    (`PackedTransformerBlock`), with axial rotary embeddings of the integer patch coordinates.

    Args:
        in_channels: Voxel feature channels.
        embed_dim: Token dimension.
        depth: Number of transformer blocks.
        num_heads: Number of attention heads.
        patch_size: Patch side $K$, in voxels.
        mlp_ratio: Hidden-to-input ratio of the block MLPs.
        qk_norm: Whether queries and keys are layer-normalized per head.
        drop_path: Stochastic-depth rate of the last block, increasing linearly from $0$ over the blocks.
        rope_frequencies: Channel pairs rotated by each axis, see `AxialRoPE`.
        rope_base: Frequency base of `AxialRoPE`.
        act: Activation of the block MLPs.
        act_kwargs: Extra arguments for the activation.
        use_flash_attn: Whether attention runs the flash attention kernel.

    Shape:
        - Input: $(N, C)$ voxel features, $(N, 3)$ integer voxel coordinates and the $(N,)$ batch index.
        - Output: $(P, D)$ patch tokens, then the $(N,)$ patch of every voxel and its $(N,)$ position in the patch.
          With `return_intermediates`, also the $(P, D)$ tokens after every block.
    """

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        patch_size: int,
        mlp_ratio: float,
        qk_norm: bool,
        drop_path: float,
        rope_frequencies: Sequence[int],
        rope_base: float,
        act: Union[str, Callable, None],
        act_kwargs: Optional[Dict[str, Any]],
        use_flash_attn: bool,
    ) -> None:
        if 2 * sum(rope_frequencies) != embed_dim // num_heads:
            raise ValueError(
                f"`rope_frequencies` {tuple(rope_frequencies)} rotate {2 * sum(rope_frequencies)} channels, but a "
                f"head has {embed_dim // num_heads}."
            )

        super().__init__()
        self.patch_size = patch_size
        self.patch_embed = VoxelPatchEmbed(in_channels, embed_dim, patch_size)
        drop_paths = torch.linspace(0, drop_path, depth).tolist()
        self.blocks = nn.ModuleList(
            [
                PackedTransformerBlock(
                    embed_dim,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    qk_norm=qk_norm,
                    drop_path=drop_paths[i],
                    act=act,
                    act_kwargs=act_kwargs,
                    rope=AxialRoPE(rope_frequencies, base=rope_base),
                    use_flash_attn=use_flash_attn,
                )
                for i in range(depth)
            ]
        )

    @overload
    def forward(
        self,
        x: Tensor,
        pos_grid: Tensor,
        batch: Tensor,
        return_intermediates: Literal[True],
    ) -> Tuple[Tensor, Tensor, Tensor, List[Tensor]]: ...

    @overload
    def forward(
        self,
        x: Tensor,
        pos_grid: Tensor,
        batch: Tensor,
        return_intermediates: Literal[False] = False,
    ) -> Tuple[Tensor, Tensor, Tensor]: ...

    def forward(self, x: Tensor, pos_grid: Tensor, batch: Tensor, return_intermediates: bool = False) -> Any:
        patch_grid, patch_batch, inverse, offset = voxel_patch_partition(pos_grid, batch, self.patch_size)
        x = self.patch_embed(x, inverse, offset, patch_grid.size(0))

        cu_seqlens = batch_to_cu_seqlens(patch_batch).int()
        max_seqlen = int(cu_seqlens.diff().max())
        intermediates: List[Tensor] = []
        for block in self.blocks:
            x = block(x, cu_seqlens, max_seqlen, pos=patch_grid)
            if return_intermediates:
                intermediates.append(x)

        if return_intermediates:
            return x, inverse, offset, intermediates
        return x, inverse, offset


class VoltDecoder(nn.Module):
    r"""Upsampling decoder of Volt: normalizes the tokens, projects them to `channels`, then expands every token to
    the voxels of its patch with one transposed convolution (`VoxelPatchExpand`).

    Args:
        in_channels: Token channels.
        channels: Output voxel feature channels.
        patch_size: Patch side $K$, in voxels.
        act: Activation function.
        act_kwargs: Extra arguments for the activation.
        norm: Normalization function; `"layer_norm"` resolves to a per-token `LayerNorm`.
        norm_kwargs: Extra arguments for the normalization.

    Shape:
        - Input: $(P, C)$ tokens, the $(N,)$ patch of every voxel and its $(N,)$ position in the patch.
        - Output: $(N, \text{channels})$ voxel features.
    """

    def __init__(
        self,
        in_channels: int,
        channels: int,
        patch_size: int,
        act: Union[str, Callable, None],
        act_kwargs: Optional[Dict[str, Any]],
        norm: Union[str, Callable, None],
        norm_kwargs: Optional[Dict[str, Any]],
    ) -> None:
        super().__init__()
        if norm == "layer_norm":
            # PyG resolves "layer_norm" to a graph LayerNorm, which normalizes over every sample of the packed batch.
            norm_kwargs = {"mode": "node", **(norm_kwargs or {})}

        self.norm = create_norm(norm, in_channels, **(norm_kwargs or {})) or nn.Identity()
        self.act = create_act(act, **(act_kwargs or {})) or nn.Identity()
        # PyG's `MLP` instantiates a norm or an activation given by name only: built here, a class works too.
        self.mlp = MLP(
            [in_channels, channels],
            act=create_act(act, **(act_kwargs or {})),
            norm=create_norm(norm, channels, **(norm_kwargs or {})),
            bias=False,
            plain_last=False,
        )
        self.expand = VoxelPatchExpand(channels, channels, patch_size)
        self.post_norm = create_norm(norm, channels, **(norm_kwargs or {})) or nn.Identity()
        self.post_act = create_act(act, **(act_kwargs or {})) or nn.Identity()

    def forward(self, x: Tensor, inverse: Tensor, offset: Tensor) -> Tensor:
        x = self.mlp(self.act(self.norm(x)))
        x = self.expand(x, inverse, offset)
        return self.post_act(self.post_norm(x))


class VoltSegmentation(SemanticSegmentationModel):
    r"""Volt (Volume Transformer) semantic segmentation model.

    From :arxiv: [Volume Transformer: Revisiting Vanilla Transformers for 3D Scene Understanding](https://arxiv.org/abs/2604.19609)
    (ECCV 2026). The scene is cut into non-overlapping volumetric patches of $K^3$ voxels, each embedded into one
    token by a linear tokenizer; a plain transformer encoder with global attention over every scene processes the
    tokens, and one transposed convolution brings them back to the voxels, followed by a linear classifier.

    Note:
        The registered checkpoints run attention with the flash attention kernel (`use_flash_attn=True`), which needs
        the `flash-attn` package and a CUDA device; pass `use_flash_attn=False` to run anywhere.

    Args:
        in_channels: Voxel feature channels.
        num_classes: Number of classes; `0` drops the classifier and returns the decoder features.
        patch_size: Patch side $K$, in voxels.
        embed_dim: Token dimension.
        depth: Number of transformer blocks.
        num_heads: Number of attention heads.
        mlp_ratio: Hidden-to-input ratio of the block MLPs.
        qk_norm: Whether queries and keys are layer-normalized per head.
        drop_path: Stochastic-depth rate of the last block.
        decoder_channels: Voxel feature channels of the decoder, entering the classifier.
        rope_frequencies: Channel pairs rotated by each axis in attention, summing to half the head dimension.
        rope_base: Frequency base of the rotary embedding.
        act: Activation function.
        act_kwargs: Extra arguments for the activation.
        norm: Normalization of the decoder.
        norm_kwargs: Extra arguments for the normalization; `None` gives the authors' `eps=1e-3` and `momentum=0.01`
            with `"batch_norm"`, and the defaults of any other normalization.
        use_flash_attn: Whether attention runs the flash attention kernel.

    Example:
        ```python
        import torch
        from torch_pointcloud.models import VoltSegmentation

        model = VoltSegmentation(in_channels=6, num_classes=20, embed_dim=96, depth=2, num_heads=2, decoder_channels=32,
                                 rope_frequencies=(8, 8, 8), use_flash_attn=False).eval()
        pos_grid = torch.unique(torch.randint(0, 64, (2048, 3)), dim=0)
        batch = torch.zeros(pos_grid.size(0), dtype=torch.long)
        logits = model(torch.randn(pos_grid.size(0), 6), pos_grid, batch)  # (N, 20)
        ```
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        *,
        patch_size: int = 5,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        qk_norm: bool = True,
        drop_path: float = 0.3,
        decoder_channels: int = 128,
        rope_frequencies: Sequence[int] = (12, 12, 8),
        rope_base: float = 100.0,
        act: Union[str, Callable, None] = "gelu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        use_flash_attn: bool = True,
    ) -> None:
        super().__init__(in_channels=in_channels, num_classes=num_classes)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.qk_norm = qk_norm
        self.drop_path = drop_path
        self.decoder_channels = decoder_channels
        self.rope_frequencies = tuple(rope_frequencies)
        self.rope_base = rope_base
        self.act = act
        self.act_kwargs = act_kwargs
        self.norm = norm
        self.norm_kwargs = norm_kwargs
        if norm_kwargs is None and norm == "batch_norm":
            self.norm_kwargs = {"eps": 1e-3, "momentum": 0.01}

        self.use_flash_attn = use_flash_attn

        self.encoder = self.configure_encoder()
        self.decoder = self.configure_decoder()
        self.head = self.configure_head()
        self.apply(_init_volt_weights)

    def configure_encoder(self) -> VoltEncoder:
        """Build the patch tokenizer and the transformer encoder."""
        return VoltEncoder(
            in_channels=self.in_channels,
            embed_dim=self.embed_dim,
            depth=self.depth,
            num_heads=self.num_heads,
            patch_size=self.patch_size,
            mlp_ratio=self.mlp_ratio,
            qk_norm=self.qk_norm,
            drop_path=self.drop_path,
            rope_frequencies=self.rope_frequencies,
            rope_base=self.rope_base,
            act=self.act,
            act_kwargs=self.act_kwargs,
            use_flash_attn=self.use_flash_attn,
        )

    def configure_decoder(self) -> VoltDecoder:
        """Build the decoder bringing the tokens back to the voxels."""
        return VoltDecoder(
            in_channels=self.embed_dim,
            channels=self.decoder_channels,
            patch_size=self.patch_size,
            act=self.act,
            act_kwargs=self.act_kwargs,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
        )

    @property
    def num_features(self) -> int:
        """Channel count $C$ of the voxel features entering the classifier."""
        return self.decoder_channels

    def configure_head(self) -> nn.Module:
        return nn.Identity() if self.num_classes == 0 else nn.Linear(self.num_features, self.num_classes)

    def reset_classifier(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.head = self.configure_head()
        self.head.apply(_init_volt_weights)

    @overload
    def forward_features(
        self,
        x: Tensor,
        pos_grid: Tensor,
        batch: Tensor,
        return_intermediates: Literal[True],
    ) -> Tuple[Tensor, Tensor, Tensor, List[Tensor]]: ...

    @overload
    def forward_features(
        self,
        x: Tensor,
        pos_grid: Tensor,
        batch: Tensor,
        return_intermediates: Literal[False] = False,
    ) -> Tuple[Tensor, Tensor, Tensor]: ...

    def forward_features(self, x: Tensor, pos_grid: Tensor, batch: Tensor, return_intermediates: bool = False) -> Any:
        r"""Encode the voxels into patch tokens.

        Args:
            x: Voxel features of shape $(N, C)$, one row per voxel.
            pos_grid: Integer voxel coordinates of shape $(N, 3)$.
            batch: Batch index of every voxel, shape $(N,)$.
            return_intermediates: Whether to also return the $(P, D)$ tokens after every transformer block.

        Returns:
            A tuple `(tokens, inverse, offset)`: the $(P, D)$ patch tokens, the $(N,)$ patch of every voxel and its
            $(N,)$ position inside the patch, the inputs of `forward_decoder`. With `return_intermediates`, the list
            of block outputs is appended, the last one being `tokens`.
        """
        return self.encoder(x, pos_grid, batch, return_intermediates=return_intermediates)

    def forward_decoder(self, x: Tensor, inverse: Tensor, offset: Tensor) -> Tensor:
        return self.decoder(x, inverse, offset)

    def forward_head(self, x: Tensor, pre_logits: bool = False) -> Tensor:
        return x if pre_logits else self.head(x)

    def forward(self, x: Tensor, pos_grid: Tensor, batch: Tensor) -> Tensor:
        r"""Forward pass.

        Args:
            x: Voxel features of shape $(N, C)$, one row per voxel.
            pos_grid: Integer voxel coordinates of shape $(N, 3)$.
            batch: Batch index of every voxel, shape $(N,)$.

        Returns:
            Per-voxel logits of shape $(N, \text{num\_classes})$, or the $(N, \text{num\_features})$ decoder features
            when `num_classes=0`.
        """
        tokens, inverse, offset = self.forward_features(x, pos_grid, batch)
        features = self.forward_decoder(tokens, inverse, offset)
        return self.forward_head(features)


def _scannet_transforms(labels: Sequence[int]) -> T.Compose:
    """Color and normal features of the ScanNet checkpoints, one point per 2 cm voxel; `labels` are the dataset classes
    mapped in order onto $0..K-1$."""
    return T.Compose(
        [
            T.Shift(keys=DataKeys.POS, method="bbox", axes=[0, 1]),
            T.Shift(keys=DataKeys.POS, method="min", axes=[2]),
            # The released weights were trained on colors in $[0, 1]$.
            T.Normalize(keys=DataKeys.COLOR, mean=[0.0, 0.0, 0.0], std=[255.0, 255.0, 255.0]),
            T.Cat(keys=[DataKeys.COLOR, DataKeys.NORMAL], dst_key=DataKeys.X, dim=1),
            T.Relabel(keys=DataKeys.SEGMENT, labels=labels, default=-1, allow_missing_keys=True),
            T.CopyItems(
                keys=[DataKeys.POS, DataKeys.SEGMENT],
                dst_keys=[DataKeys.ORIGIN_POS, DataKeys.ORIGIN_SEGMENT],
                allow_missing_keys=True,
            ),
            T.Voxelize(
                pos_key=DataKeys.POS,
                pos_reduce="first",
                dst_pos_grid_key=DataKeys.POS_GRID,
                keys=[DataKeys.X, DataKeys.SEGMENT, DataKeys.COLOR, DataKeys.NORMAL, DataKeys.INSTANCE],
                reduce="first",
                size=0.02,
                method="fnv",
                allow_missing_keys=True,
                dst_inverse_key=DataKeys.INVERSE,
            ),
        ]
    )


def _nuscenes_transforms() -> T.Compose:
    """Position and intensity features of the nuScenes checkpoints, one point per 5 cm voxel."""
    return T.Compose(
        [
            T.Relabel(
                keys=DataKeys.SEGMENT,
                labels=NUSCENES_LIDARSEG_LEARNING_MAP,
                default=-1,
                allow_missing_keys=True,
            ),
            # The released weights were trained on intensities in $[0, 1]$.
            T.Normalize(keys=DataKeys.INTENSITY, mean=[0.0], std=[255.0]),
            T.CopyItems(
                keys=[DataKeys.POS, DataKeys.SEGMENT],
                dst_keys=[DataKeys.ORIGIN_POS, DataKeys.ORIGIN_SEGMENT],
                allow_missing_keys=True,
            ),
            # Points beyond 70 m are pulled back onto that range and heights clamped, as in training.
            T.ClampNorm(keys=DataKeys.POS, max_norm=70.0),
            T.Clamp(keys=DataKeys.POS, min=[-math.inf, -math.inf, -4.0], max=[math.inf, math.inf, 2.0]),
            T.Cat(keys=[DataKeys.POS, DataKeys.INTENSITY], dst_key=DataKeys.X, dim=1),
            T.Voxelize(
                pos_key=DataKeys.POS,
                pos_reduce="first",
                dst_pos_grid_key=DataKeys.POS_GRID,
                keys=[DataKeys.X, DataKeys.SEGMENT, DataKeys.INTENSITY],
                reduce="first",
                size=0.05,
                method="fnv",
                allow_missing_keys=True,
                dst_inverse_key=DataKeys.INVERSE,
            ),
        ]
    )


def _semantickitti_transforms() -> T.Compose:
    """Position and intensity features of the SemanticKITTI checkpoints, one point per 5 cm voxel."""
    return T.Compose(
        [
            T.Relabel(
                keys=DataKeys.SEGMENT,
                labels=SEMANTIC_KITTI_LEARNING_MAP,
                default=-1,
                allow_missing_keys=True,
            ),
            T.CopyItems(
                keys=[DataKeys.POS, DataKeys.SEGMENT],
                dst_keys=[DataKeys.ORIGIN_POS, DataKeys.ORIGIN_SEGMENT],
                allow_missing_keys=True,
            ),
            # Points beyond 50 m are pulled back onto that range and heights clamped, as in training.
            T.ClampNorm(keys=DataKeys.POS, max_norm=50.0),
            T.Clamp(keys=DataKeys.POS, min=[-math.inf, -math.inf, -4.0], max=[math.inf, math.inf, 2.0]),
            T.Cat(keys=[DataKeys.POS, DataKeys.INTENSITY], dst_key=DataKeys.X, dim=1),
            T.Voxelize(
                pos_key=DataKeys.POS,
                pos_reduce="first",
                dst_pos_grid_key=DataKeys.POS_GRID,
                keys=[DataKeys.X, DataKeys.SEGMENT, DataKeys.INTENSITY],
                reduce="first",
                size=0.05,
                method="fnv",
                allow_missing_keys=True,
                dst_inverse_key=DataKeys.INVERSE,
            ),
        ]
    )


@register_model(
    "volt-s.scannet20.kadir-yilmaz",
    task="semantic-segmentation",
    input_keys=("x", "pos_grid", "batch"),
    weights=WeightsDict(
        url="hf://torch-pointcloud/volt-s.scannet20.kadir-yilmaz/resolve/0000000000000000000000000000000000000000/model.safetensors",
        dataset="scannet20",
        metrics={"mIoU": 77.27, "OA": 91.77, "simple/mIoU": 76.38, "simple/OA": 91.36},
        classes=SCANNET20_CLASSES,
        author="kadir-yilmaz",
        license="MIT",
    ),
    transform=_scannet_transforms(range(1, 21)),
    hparams=dict(
        in_channels=6,
        num_classes=20,
        patch_size=5,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        qk_norm=True,
        drop_path=0.3,
        decoder_channels=128,
        rope_frequencies=(12, 12, 8),
        rope_base=100.0,
        norm_kwargs={"eps": 1e-3, "momentum": 0.01},
        use_flash_attn=True,
    ),
)
def volt_s_scannet20(**hparams: Any) -> VoltSegmentation:
    return VoltSegmentation(**hparams)


@register_model(
    "volt-s.scannet200.kadir-yilmaz",
    task="semantic-segmentation",
    input_keys=("x", "pos_grid", "batch"),
    weights=WeightsDict(
        url="hf://torch-pointcloud/volt-s.scannet200.kadir-yilmaz/resolve/0000000000000000000000000000000000000000/model.safetensors",
        dataset="scannet200",
        metrics={"simple/mIoU": 35.36, "simple/OA": 83.32},
        classes=SCANNET200_CLASSES,
        author="kadir-yilmaz",
        license="MIT",
    ),
    transform=_scannet_transforms(range(1, 201)),
    hparams=dict(
        in_channels=6,
        num_classes=200,
        patch_size=5,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        qk_norm=True,
        drop_path=0.3,
        decoder_channels=128,
        rope_frequencies=(12, 12, 8),
        rope_base=100.0,
        norm_kwargs={"eps": 1e-3, "momentum": 0.01},
        use_flash_attn=True,
    ),
)
def volt_s_scannet200(**hparams: Any) -> VoltSegmentation:
    return VoltSegmentation(**hparams)


@register_model(
    "volt-s.nuscenes.kadir-yilmaz",
    task="semantic-segmentation",
    input_keys=("x", "pos_grid", "batch"),
    weights=WeightsDict(
        url="hf://torch-pointcloud/volt-s.nuscenes.kadir-yilmaz/resolve/0000000000000000000000000000000000000000/model.safetensors",
        dataset="nuscenes",
        metrics={"mIoU": 80.57, "OA": 94.54, "simple/mIoU": 80.57, "simple/OA": 94.54},
        classes=NUSCENES_LIDARSEG_CLASSES,
        author="kadir-yilmaz",
        license="MIT",
    ),
    transform=_nuscenes_transforms(),
    hparams=dict(
        in_channels=4,
        num_classes=16,
        patch_size=5,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        qk_norm=True,
        drop_path=0.3,
        decoder_channels=128,
        rope_frequencies=(12, 12, 8),
        rope_base=100.0,
        norm_kwargs={"eps": 1e-3, "momentum": 0.01},
        use_flash_attn=True,
    ),
)
def volt_s_nuscenes(**hparams: Any) -> VoltSegmentation:
    return VoltSegmentation(**hparams)


@register_model(
    "volt-s.semantickitti.kadir-yilmaz",
    task="semantic-segmentation",
    input_keys=("x", "pos_grid", "batch"),
    weights=WeightsDict(
        url="hf://torch-pointcloud/volt-s.semantickitti.kadir-yilmaz/resolve/0000000000000000000000000000000000000000/model.safetensors",
        dataset="semantickitti",
        metrics={"mIoU": 69.42, "OA": 92.64, "simple/mIoU": 69.42, "simple/OA": 92.64},
        classes=SEMANTIC_KITTI_CLASSES,
        author="kadir-yilmaz",
        license="MIT",
    ),
    transform=_semantickitti_transforms(),
    hparams=dict(
        in_channels=4,
        num_classes=19,
        patch_size=5,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        qk_norm=True,
        drop_path=0.3,
        decoder_channels=128,
        rope_frequencies=(12, 12, 8),
        rope_base=100.0,
        norm_kwargs={"eps": 1e-3, "momentum": 0.01},
        use_flash_attn=True,
    ),
)
def volt_s_semantickitti(**hparams: Any) -> VoltSegmentation:
    return VoltSegmentation(**hparams)


@register_model(
    "volt-s-joint.scannet20.kadir-yilmaz",
    task="semantic-segmentation",
    input_keys=("x", "pos_grid", "batch"),
    weights=WeightsDict(
        url="hf://torch-pointcloud/volt-s-joint.scannet20.kadir-yilmaz/resolve/0000000000000000000000000000000000000000/model.safetensors",
        dataset="scannet20",
        metrics={"mIoU": 80.18, "OA": 92.81, "simple/mIoU": 79.50, "simple/OA": 92.45},
        classes=SCANNET20_CLASSES,
        author="kadir-yilmaz",
        license="MIT",
    ),
    transform=_scannet_transforms(range(1, 21)),
    hparams=dict(
        in_channels=6,
        num_classes=20,
        patch_size=5,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        qk_norm=True,
        drop_path=0.3,
        decoder_channels=128,
        rope_frequencies=(12, 12, 8),
        rope_base=100.0,
        norm_kwargs={"eps": 1e-3, "momentum": 0.01},
        use_flash_attn=True,
    ),
)
def volt_s_joint_scannet20(**hparams: Any) -> VoltSegmentation:
    return VoltSegmentation(**hparams)


@register_model(
    "volt-s-joint.scannet200.kadir-yilmaz",
    task="semantic-segmentation",
    input_keys=("x", "pos_grid", "batch"),
    weights=WeightsDict(
        url="hf://torch-pointcloud/volt-s-joint.scannet200.kadir-yilmaz/resolve/0000000000000000000000000000000000000000/model.safetensors",
        dataset="scannet200",
        metrics={"mIoU": 38.53, "OA": 84.79, "simple/mIoU": 37.91, "simple/OA": 84.27},
        classes=SCANNET200_CLASSES,
        author="kadir-yilmaz",
        license="MIT",
    ),
    transform=_scannet_transforms(range(1, 201)),
    hparams=dict(
        in_channels=6,
        num_classes=200,
        patch_size=5,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        qk_norm=True,
        drop_path=0.3,
        decoder_channels=128,
        rope_frequencies=(12, 12, 8),
        rope_base=100.0,
        norm_kwargs={"eps": 1e-3, "momentum": 0.01},
        use_flash_attn=True,
    ),
)
def volt_s_joint_scannet200(**hparams: Any) -> VoltSegmentation:
    return VoltSegmentation(**hparams)


@register_model(
    "volt-b-joint.scannet20.kadir-yilmaz",
    task="semantic-segmentation",
    input_keys=("x", "pos_grid", "batch"),
    weights=WeightsDict(
        url="hf://torch-pointcloud/volt-b-joint.scannet20.kadir-yilmaz/resolve/0000000000000000000000000000000000000000/model.safetensors",
        dataset="scannet20",
        metrics={"simple/mIoU": 79.69, "simple/OA": 92.78},
        classes=SCANNET20_CLASSES,
        author="kadir-yilmaz",
        license="MIT",
    ),
    transform=_scannet_transforms(range(1, 21)),
    hparams=dict(
        in_channels=6,
        num_classes=20,
        patch_size=5,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qk_norm=True,
        drop_path=0.3,
        decoder_channels=256,
        rope_frequencies=(12, 12, 8),
        rope_base=100.0,
        norm_kwargs={"eps": 1e-3, "momentum": 0.01},
        use_flash_attn=True,
    ),
)
def volt_b_joint_scannet20(**hparams: Any) -> VoltSegmentation:
    return VoltSegmentation(**hparams)


@register_model(
    "volt-b-joint.scannet200.kadir-yilmaz",
    task="semantic-segmentation",
    input_keys=("x", "pos_grid", "batch"),
    weights=WeightsDict(
        url="hf://torch-pointcloud/volt-b-joint.scannet200.kadir-yilmaz/resolve/0000000000000000000000000000000000000000/model.safetensors",
        dataset="scannet200",
        metrics={"simple/mIoU": 39.47, "simple/OA": 85.01},
        classes=SCANNET200_CLASSES,
        author="kadir-yilmaz",
        license="MIT",
    ),
    transform=_scannet_transforms(range(1, 201)),
    hparams=dict(
        in_channels=6,
        num_classes=200,
        patch_size=5,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qk_norm=True,
        drop_path=0.3,
        decoder_channels=256,
        rope_frequencies=(12, 12, 8),
        rope_base=100.0,
        norm_kwargs={"eps": 1e-3, "momentum": 0.01},
        use_flash_attn=True,
    ),
)
def volt_b_joint_scannet200(**hparams: Any) -> VoltSegmentation:
    return VoltSegmentation(**hparams)
