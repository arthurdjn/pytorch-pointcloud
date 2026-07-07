r"""XCube: large-scale 3D generative modeling using sparse voxel hierarchies.

:arxiv: [XCube: Large-Scale 3D Generative Modeling using Sparse Voxel Hierarchies](https://arxiv.org/abs/2312.03806)

XCube generates high-resolution sparse voxel grids with a hierarchy of two latent diffusion stages. Each
stage pairs a sparse structure VAE (`XCubeVAE`) that compresses a voxel grid into a low-resolution latent
grid, with a latent diffusion model (`XCubeDiffusion`) that samples new latents. The coarse stage generates
a full shape at low resolution; the fine stage upsamples it conditioned on the coarse output. Sparse voxel
grids are represented with :github: [fvdb](https://github.com/voxel-foundation/fvdb).

Adapted from :github: [nv-tlabs/XCube](https://github.com/nv-tlabs/XCube).
"""

import math
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torch_pointcloud.layers.act import create_act
from torch_pointcloud.layers.norms import create_norm
from torch_pointcloud.models._base import BaseModel
from torch_pointcloud.models._registry import register_model
from torch_pointcloud.utils.diffusion import DDIMScheduler
from torch_pointcloud.utils.imports import _FVDB_GITHUB_URL, optional_import

if TYPE_CHECKING:
    from fvdb import ConvolutionPlan, GridBatch, JaggedTensor
else:
    ConvolutionPlan, _ = optional_import("fvdb", "ConvolutionPlan", url=_FVDB_GITHUB_URL)
    GridBatch, _ = optional_import("fvdb", "GridBatch", url=_FVDB_GITHUB_URL)
    JaggedTensor, _ = optional_import("fvdb", "JaggedTensor", url=_FVDB_GITHUB_URL)

fvnn, _FVDB_AVAILABLE = optional_import("fvdb.nn", url=_FVDB_GITHUB_URL)


def fourier_encode(x: Tensor, num_freqs: int, include_input: bool = True) -> Tensor:
    r"""NeRF-style sinusoidal encoding with log-spaced frequencies $2^0, \ldots, 2^{\text{num\_freqs}-1}$.

    Args:
        x: Input coordinates.
        num_freqs: Number of frequency bands.
        include_input: Prepend the raw input to the encoding.

    Returns:
        The encoded features, ordered as $[x, \sin(2^0 x), \cos(2^0 x), \ldots]$.

    Shape:
        - Input: $(N, C)$
        - Output: $(N, C \cdot (2 \cdot \text{num\_freqs} + 1))$ (or without the leading $C$ block if
          `include_input` is false)
    """
    outputs = [x] if include_input else []
    freqs = 2.0 ** torch.linspace(0.0, num_freqs - 1, steps=num_freqs, device=x.device, dtype=x.dtype)
    for freq in freqs:
        outputs.append(torch.sin(x * freq))
        outputs.append(torch.cos(x * freq))
    return torch.cat(outputs, dim=-1)


def timestep_encoding(timesteps: Tensor, dim: int, max_period: int = 10000) -> Tensor:
    r"""Sinusoidal diffusion timestep embedding.

    Args:
        timesteps: One timestep index per batch element.
        dim: Embedding dimension.
        max_period: Controls the minimum embedding frequency.

    Returns:
        The timestep embeddings.

    Shape:
        - Input: $(B,)$
        - Output: $(B, \text{dim})$
    """
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32, device=timesteps.device) / half)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def _submanifold_plan(grid: "GridBatch") -> "ConvolutionPlan":
    return ConvolutionPlan.from_grid_batch(3, 1, grid, target_grid=grid)


class SparseGroupNorm(nn.GroupNorm):
    r"""Group normalization over the voxels of each grid in a batch.

    Normalizes each batch element independently over all of its voxels, matching `fvdb.nn.GroupNorm`.

    Args:
        num_groups: Number of channel groups.
        num_channels: Number of feature channels.

    Shape:
        - `x`: $(N, C)$ flat voxel features.
        - `offsets`: $(B + 1,)$ cumulative voxel counts per grid.
        - Output: $(N, C)$
    """

    def forward(self, x: Tensor, offsets: Tensor) -> Tensor:  # type: ignore[override]
        out = torch.empty_like(x)
        for b in range(offsets.numel() - 1):
            start, end = int(offsets[b]), int(offsets[b + 1])
            if end > start:
                feat = x[start:end].transpose(0, 1).reshape(1, self.num_channels, -1)
                feat = super().forward(feat)
                out[start:end] = feat.reshape(self.num_channels, -1).transpose(0, 1)
        return out


def create_sparse_norm(norm: Union[str, Callable], channels: int, **norm_kwargs: Any) -> nn.Module:
    r"""Resolve a normalization layer for jagged per-grid voxel features.

    Strings resolve to `SparseGroupNorm`; a callable is invoked as `norm(channels, **norm_kwargs)` and
    must return a module whose forward takes the flat features and the per-grid offsets.

    Args:
        norm: Norm name (`"group_norm"`), a class or a callable.
        channels: Number of feature channels.
        **norm_kwargs: Forwarded to the norm constructor (e.g. `num_groups`).

    Returns:
        The instantiated norm module.
    """
    if isinstance(norm, str):
        if norm.lower().replace("-", "_") in {"group_norm", "groupnorm", "gn"}:
            return SparseGroupNorm(num_channels=channels, **norm_kwargs)
        raise ValueError(f"Unknown sparse norm string {norm!r}. Use 'group_norm' or pass a callable.")
    return norm(channels, **norm_kwargs)


class SparseConvBlock(nn.Module):
    r"""Pre-norm sparse convolution block: group norm, $3^3$ sparse convolution, activation.

    Args:
        in_channels: Input feature channels.
        out_channels: Output feature channels.
        norm: Normalization name or callable (see `create_sparse_norm`).
        norm_kwargs: Extra normalization arguments; `num_groups` is clamped to $1$ when `in_channels`
            is smaller.
        act: Activation name, callable or `None`.
        act_kwargs: Extra activation arguments.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm: Union[str, Callable] = "group_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        norm_kwargs = dict(norm_kwargs) if norm_kwargs is not None else {"num_groups": 8}
        if in_channels < norm_kwargs.get("num_groups", 1):
            norm_kwargs["num_groups"] = 1
        self.norm = create_sparse_norm(norm, in_channels, **norm_kwargs)
        self.conv = fvnn.SparseConv3d(in_channels, out_channels, 3, 1, bias=False)
        self.act = create_act(act, **(act_kwargs or {}))

    def forward(self, x: Tensor, grid: "GridBatch", plan: "ConvolutionPlan") -> Tensor:
        x = self.norm(x, grid.ijk.joffsets)
        x = self.conv(grid.ijk.jagged_like(x), plan).jdata
        if self.act is not None:
            x = self.act(x)
        return x


class SparseDoubleConv(nn.Module):
    r"""Two `SparseConvBlock`s with an optional max-pool entry, the XCube VAE building block.

    Args:
        in_channels: Input feature channels.
        mid_channels: Channels between the two conv blocks.
        out_channels: Output feature channels.
        norm: Normalization name or callable.
        norm_kwargs: Extra normalization arguments.
        act: Activation name, callable or `None`.
        act_kwargs: Extra activation arguments.
        pool_factor: Down-sampling factor applied before the convolutions, or `None` for no pooling.
    """

    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        norm: Union[str, Callable] = "group_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        pool_factor: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.pool_factor = pool_factor
        block_kwargs: Dict[str, Any] = dict(norm=norm, norm_kwargs=norm_kwargs, act=act, act_kwargs=act_kwargs)
        self.conv1 = SparseConvBlock(in_channels, mid_channels, **block_kwargs)
        self.conv2 = SparseConvBlock(mid_channels, out_channels, **block_kwargs)

    def forward(
        self,
        x: Tensor,
        grid: "GridBatch",
        plan: "ConvolutionPlan",
        coarse_grid: Optional["GridBatch"] = None,
    ) -> Tuple[Tensor, "GridBatch", "ConvolutionPlan"]:
        if self.pool_factor is not None:
            data, grid = grid.max_pool(
                self.pool_factor, grid.ijk.jagged_like(x), stride=self.pool_factor, coarse_grid=coarse_grid
            )
            x = data.jdata
            x = torch.where(torch.isinf(x), torch.zeros_like(x), x)
            plan = _submanifold_plan(grid)
        x = self.conv1(x, grid, plan)
        x = self.conv2(x, grid, plan)
        return x, grid, plan


class SparseHead(nn.Module):
    r"""Per-voxel prediction head: one `SparseConvBlock` followed by a linear projection.

    Args:
        channels: Input feature channels.
        out_channels: Output channels.
        norm: Normalization name or callable.
        norm_kwargs: Extra normalization arguments.
        act: Activation name, callable or `None`.
        act_kwargs: Extra activation arguments.
    """

    def __init__(
        self,
        channels: int,
        out_channels: int,
        norm: Union[str, Callable] = "group_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.conv = SparseConvBlock(
            channels, channels, norm=norm, norm_kwargs=norm_kwargs, act=act, act_kwargs=act_kwargs
        )
        self.out = nn.Linear(channels, out_channels)

    def forward(self, x: Tensor, grid: "GridBatch", plan: "ConvolutionPlan") -> Tensor:
        return self.out(self.conv(x, grid, plan))


class XCubeGridEncoder(nn.Module):
    r"""Per-voxel input featurizer of the XCube VAE.

    Encodes voxel centers with a sinusoidal position encoding, optionally concatenates splatted point
    normals, and mixes the result with a linear layer.

    Args:
        out_channels: Output feature channels.
        use_normal: Concatenate trilinearly splatted (and renormalized) point normals.
        num_freqs: Frequency bands of the position encoding.
    """

    def __init__(self, out_channels: int, use_normal: bool = True, num_freqs: int = 5) -> None:
        super().__init__()
        self.use_normal = use_normal
        self.num_freqs = num_freqs
        in_channels = 3 * (2 * num_freqs + 1) + (3 if use_normal else 0)
        self.mix = nn.Linear(in_channels, out_channels)

    def forward(self, grid: "GridBatch", points: "JaggedTensor", normal: Optional[Tensor] = None) -> Tensor:
        coords = grid.voxel_to_world(grid.ijk.float()).jdata
        feat = fourier_encode(coords, self.num_freqs)
        if self.use_normal:
            if normal is None:
                raise ValueError("This encoder was built with `use_normal=True` but no normals were given.")
            splatted = grid.splat_trilinear(points, points.jagged_like(normal)).jdata
            splatted = splatted / (splatted.norm(dim=1, keepdim=True) + 1e-6)
            feat = torch.cat([feat, splatted], dim=1)
        return self.mix(feat)


class XCubeStructureUNet(nn.Module):
    r"""Hierarchical sparse structure UNet of the XCube VAE.

    The encoder pools the input grid down `len(channels) - 1` times (guided by a hash tree of dilated
    grids when given); the bottleneck maps to a $2 \cdot \text{latent\_channels}$ Gaussian posterior. The
    decoder subdivides back up, predicting at every level which voxels exist (structure logits) and pruning
    accordingly. Optional heads predict per-voxel normals and semantics at the finest level.

    Args:
        in_channels: Input feature channels.
        channels: Feature channels per level, coarse to fine in reverse (index $0$ is the finest level).
        latent_channels: Channels of the latent grid.
        norm: Normalization name or callable.
        norm_kwargs: Extra normalization arguments (defaults to $8$ groups).
        act: Activation name, callable or `None`.
        act_kwargs: Extra activation arguments.
        neck_bound: Half-extent of a dense latent grid (the encoder output is padded onto it). `None`
            keeps the sparse topology of the deepest level.
        with_normal_head: Predict per-voxel normals at the finest level.
        with_semantic_head: Predict per-voxel semantic logits at the finest level.
        num_classes: Number of semantic classes for the semantic head.
    """

    def __init__(
        self,
        in_channels: int,
        channels: Sequence[int],
        latent_channels: int,
        norm: Union[str, Callable] = "group_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        neck_bound: Optional[Tuple[int, int, int]] = None,
        with_normal_head: bool = False,
        with_semantic_head: bool = False,
        num_classes: int = 0,
    ) -> None:
        super().__init__()
        self.channels = tuple(channels)
        self.latent_channels = latent_channels
        self.neck_bound = neck_bound
        norm_kwargs = dict(norm_kwargs) if norm_kwargs is not None else {"num_groups": 8}
        block_kwargs: Dict[str, Any] = dict(norm=norm, norm_kwargs=norm_kwargs, act=act, act_kwargs=act_kwargs)

        self.pre_conv = nn.Linear(in_channels, in_channels)
        encoders = []
        feature_sizes = [in_channels, *channels]
        for i in range(len(channels)):
            block_in, block_out = feature_sizes[i], feature_sizes[i + 1]
            mid = max(block_out // 2, block_in)
            encoders.append(
                SparseDoubleConv(block_in, mid, block_out, pool_factor=2 if i > 0 else None, **block_kwargs)
            )
        self.encoders = nn.ModuleList(encoders)

        deepest = self.channels[-1]
        self.pre_bottleneck = nn.ModuleList(
            [
                SparseDoubleConv(deepest, deepest, deepest, **block_kwargs),
                SparseDoubleConv(deepest, 2 * latent_channels, 2 * latent_channels, **block_kwargs),
            ]
        )
        self.post_bottleneck = nn.ModuleList(
            [
                SparseDoubleConv(latent_channels, deepest, deepest, **block_kwargs),
                SparseDoubleConv(deepest, deepest, deepest, **block_kwargs),
            ]
        )

        self.struct_heads = nn.ModuleList(
            SparseHead(self.channels[-1 - i], 2, **block_kwargs) for i in range(len(channels))
        )
        self.decoders = nn.ModuleList(
            SparseDoubleConv(self.channels[-1 - i], self.channels[-2 - i], self.channels[-2 - i], **block_kwargs)
            for i in range(len(channels) - 1)
        )

        self.normal_head = SparseHead(self.channels[0], 3, **block_kwargs) if with_normal_head else None
        self.semantic_head = SparseHead(self.channels[0], num_classes, **block_kwargs) if with_semantic_head else None

    def encode(
        self,
        x: Tensor,
        grid: "GridBatch",
        hash_tree: Optional[List["GridBatch"]] = None,
    ) -> Tuple[Tensor, Tensor, "GridBatch"]:
        r"""Encode voxel features into the Gaussian posterior of the latent grid.

        Args:
            x: Per-voxel features of the input grid.
            grid: Input grid.
            hash_tree: Pre-built (dilated) target grid per level; entry $d$ is used as the pooling target
                of level $d$.

        Returns:
            The posterior mean, posterior log-variance and the latent grid.

        Shape:
            - `x`: $(N, C_\text{in})$
            - Output: $(M, C_\text{latent})$, $(M, C_\text{latent})$ where $M$ is the latent voxel count.
        """
        x = self.pre_conv(x)
        plan = _submanifold_plan(grid)
        for depth, encoder in enumerate(self.encoders):
            coarse_grid = hash_tree[depth] if hash_tree is not None and encoder.pool_factor is not None else None
            x, grid, plan = encoder(x, grid, plan, coarse_grid)

        if self.neck_bound is not None:
            dims = [2 * b for b in self.neck_bound]
            ijk_min = [-b for b in self.neck_bound]
            neck = GridBatch.from_dense(
                grid.grid_count,
                dims,
                ijk_min,
                voxel_sizes=grid.voxel_size_at(0),
                origins=grid.origin_at(0),
                device=grid.device,
            )
            x = neck.inject_from(grid, grid.ijk.jagged_like(x)).jdata
            grid = neck
            plan = _submanifold_plan(grid)

        for block in self.pre_bottleneck:
            x, grid, plan = block(x, grid, plan)
        mu, logvar = torch.chunk(x, 2, dim=1)
        return mu, logvar, grid

    def decode(self, x: Tensor, grid: "GridBatch") -> Dict[str, Any]:
        r"""Decode a latent grid into the predicted voxel hierarchy.

        Args:
            x: Latent features.
            grid: Latent grid.

        Returns:
            A dict with, keyed by level ($0$ is finest): the per-voxel existence `structure_logits`, the
            (unpruned) `structure_logit_grids` they live on, and the pruned `structure_grids`. The
            features `x` and pruned grid `grid` of the finest level, and `normal` / `semantic`
            predictions when the heads exist.

        Shape:
            - `x`: $(M, C_\text{latent})$
        """
        plan = _submanifold_plan(grid)
        for block in self.post_bottleneck:
            x, grid, plan = block(x, grid, plan)

        out: Dict[str, Any] = {"structure_logits": {}, "structure_logit_grids": {}, "structure_grids": {}}
        mask: Optional["JaggedTensor"] = None
        depth = len(self.channels) - 1
        for i, head in enumerate(self.struct_heads):
            if i > 0:
                data, grid = grid.refine(2, grid.ijk.jagged_like(x), mask=mask)
                x = data.jdata
                plan = _submanifold_plan(grid)
                x, grid, plan = self.decoders[i - 1](x, grid, plan)
            logits = head(x, grid, plan)
            out["structure_logits"][depth] = logits
            out["structure_logit_grids"][depth] = grid
            mask = grid.ijk.jagged_like(logits[:, 0] > logits[:, 1])
            out["structure_grids"][depth] = grid.refine(1, grid.ijk.jagged_like(x), mask=mask)[1]
            depth -= 1

        data, grid = grid.refine(1, grid.ijk.jagged_like(x), mask=mask)
        x = data.jdata
        out["x"], out["grid"] = x, grid
        if grid.total_voxels > 0:
            plan = _submanifold_plan(grid)
            if self.normal_head is not None:
                out["normal"] = self.normal_head(x, grid, plan)
            if self.semantic_head is not None:
                out["semantic"] = self.semantic_head(x, grid, plan)
        return out


class XCubeVAE(BaseModel):
    r"""XCube sparse structure VAE.

    :arxiv: [XCube](https://arxiv.org/abs/2312.03806) stage-one model: voxelizes a point cloud, encodes
    per-voxel features into a low-resolution latent grid with a Gaussian posterior, and decodes it back
    into a sparse voxel hierarchy by predicting the structure (existence) of voxels level by level.

    Args:
        in_channels: Raw input coordinate channels (kept for the model registry; always $3$).
        encoder_channels: Output channels of the input featurizer.
        channels: UNet feature channels per level, finest first.
        latent_channels: Channels of the latent grid.
        voxel_size: Edge length of the finest voxels.
        neck_bound: Half-extent of a dense latent grid, or `None` to keep a sparse latent topology.
        use_normal: Use point normals as encoder input.
        use_hash_tree: Pool onto splatted (dilated) grids built from the input instead of plain
            coarsenings.
        norm: Normalization name or callable.
        norm_kwargs: Extra normalization arguments (defaults to $8$ groups).
        act: Activation name, callable or `None`.
        act_kwargs: Extra activation arguments.
        pos_encoding_freqs: Frequency bands of the voxel-center position encoding.
        with_normal_head: Predict per-voxel normals at the finest level.
        with_semantic_head: Predict per-voxel semantics at the finest level.
        num_classes: Number of semantic classes.

    Example:
        ```python
        model = create_model("xcube-vae-coarse-nvidia.shapenet-chair", task="base", pretrained=True)
        out = model(pos, batch, normal=normal)
        recon_grid = out["grid"]
        ```
    """

    def __init__(
        self,
        in_channels: int = 3,
        encoder_channels: int = 32,
        channels: Sequence[int] = (64, 128, 256, 512),
        latent_channels: int = 16,
        voxel_size: float = 0.01,
        neck_bound: Optional[Tuple[int, int, int]] = None,
        use_normal: bool = True,
        use_hash_tree: bool = True,
        norm: Union[str, Callable] = "group_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        pos_encoding_freqs: int = 5,
        with_normal_head: bool = False,
        with_semantic_head: bool = False,
        num_classes: int = 0,
    ) -> None:
        super().__init__(in_channels)
        self.voxel_size = voxel_size
        self.latent_channels = latent_channels
        self.num_levels = len(channels)
        self.use_hash_tree = use_hash_tree
        self.encoder = XCubeGridEncoder(encoder_channels, use_normal=use_normal, num_freqs=pos_encoding_freqs)
        self.unet = XCubeStructureUNet(
            encoder_channels,
            channels,
            latent_channels,
            norm=norm,
            norm_kwargs=norm_kwargs,
            act=act,
            act_kwargs=act_kwargs,
            neck_bound=neck_bound,
            with_normal_head=with_normal_head,
            with_semantic_head=with_semantic_head,
            num_classes=num_classes,
        )

    def build_grid(self, pos: Tensor, batch: Tensor) -> Tuple["GridBatch", "JaggedTensor"]:
        r"""Voxelize a packed point cloud onto the finest grid.

        Args:
            pos: Point positions.
            batch: Batch indices.

        Returns:
            The voxel grid and the points as a `JaggedTensor`.

        Shape:
            - `pos`: $(N, 3)$
            - `batch`: $(N,)$
        """
        batch_size = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
        points = JaggedTensor.from_data_and_indices(pos, batch.int(), batch_size)
        origins = [self.voxel_size / 2.0] * 3
        grid = GridBatch.from_points(points, voxel_sizes=self.voxel_size, origins=origins)
        return grid, points

    def build_hash_tree(self, grid: "GridBatch") -> List["GridBatch"]:
        r"""Build the dilated target grid of every encoder level from the finest grid.

        Level $d$ covers the voxels of size $\text{voxel\_size} \cdot 2^d$ nearest to each finest voxel
        center (a one-voxel dilation, Sec. 3.4 of the paper).

        Args:
            grid: Finest voxel grid.

        Returns:
            One grid per level; entry $0$ is `grid` itself.
        """
        centers = grid.voxel_to_world(grid.ijk.float())
        tree = [grid]
        for depth in range(1, self.num_levels):
            size = self.voxel_size * 2**depth
            tree.append(GridBatch.from_nearest_voxels_to_points(centers, voxel_sizes=size, origins=[size / 2.0] * 3))
        return tree

    def encode(
        self,
        pos: Tensor,
        batch: Tensor,
        normal: Optional[Tensor] = None,
        sample: bool = False,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[Tensor, "GridBatch"]:
        r"""Encode a point cloud into latent grid features.

        Args:
            pos: Point positions.
            batch: Batch indices.
            normal: Per-point normals (required if the encoder uses normals).
            sample: Sample the posterior instead of returning its mean.
            generator: Random generator for posterior sampling.

        Returns:
            The latent features and the latent grid.

        Shape:
            - `pos`: $(N, 3)$, `batch`: $(N,)$, `normal`: $(N, 3)$
            - Output: $(M, C_\text{latent})$
        """
        grid, points = self.build_grid(pos, batch)
        hash_tree = self.build_hash_tree(grid) if self.use_hash_tree else None
        feat = self.encoder(grid, points, normal)
        mu, logvar, latent_grid = self.unet.encode(feat, grid, hash_tree)
        if sample:
            std = (logvar / 2.0).exp()
            noise = torch.randn(std.shape, generator=generator, device=std.device, dtype=std.dtype)
            return mu + std * noise, latent_grid
        return mu, latent_grid

    def decode(self, z: Tensor, grid: "GridBatch") -> Dict[str, Any]:
        r"""Decode latent grid features into a sparse voxel hierarchy.

        Args:
            z: Latent features.
            grid: Latent grid.

        Returns:
            The decoder output dict (see `XCubeStructureUNet.decode`).

        Shape:
            - `z`: $(M, C_\text{latent})$
        """
        return self.unet.decode(z, grid)

    def forward(self, pos: Tensor, batch: Tensor, normal: Optional[Tensor] = None) -> Dict[str, Any]:
        r"""Encode and reconstruct a point cloud.

        Args:
            pos: Point positions.
            batch: Batch indices.
            normal: Per-point normals.

        Returns:
            The decoder output dict, extended with the posterior `mu` / `logvar`, the voxelized
            `input_grid` and the encoder `hash_tree` (when enabled).

        Shape:
            - `pos`: $(N, 3)$, `batch`: $(N,)$, `normal`: $(N, 3)$
        """
        grid, points = self.build_grid(pos, batch)
        hash_tree = self.build_hash_tree(grid) if self.use_hash_tree else None
        feat = self.encoder(grid, points, normal)
        mu, logvar, latent_grid = self.unet.encode(feat, grid, hash_tree)
        std = (logvar / 2.0).exp()
        z = mu + std * torch.randn_like(std) if self.training else mu
        out = self.unet.decode(z, latent_grid)
        out.update({"mu": mu, "logvar": logvar, "latent_grid": latent_grid, "input_grid": grid})
        if hash_tree is not None:
            out["hash_tree"] = hash_tree
        return out


class SparseResBlock(nn.Module):
    r"""Sparse diffusion UNet residual block with timestep scale-shift conditioning.

    Args:
        channels: Input feature channels.
        embed_dim: Timestep embedding dimension.
        out_channels: Output channels (defaults to `channels`).
        dropout: Dropout probability.
        norm: Normalization name or callable.
        norm_kwargs: Extra normalization arguments (defaults to $32$ groups).
        act: Activation name, callable or `None`.
        act_kwargs: Extra activation arguments.
        up: Refine the grid by `factor` before the first convolution.
        down: Pool the grid by `factor` before the first convolution.
        factor: Up / down sampling factor.
    """

    def __init__(
        self,
        channels: int,
        embed_dim: int,
        out_channels: Optional[int] = None,
        dropout: float = 0.0,
        norm: Union[str, Callable] = "group_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        act: Union[str, Callable, None] = "silu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        up: bool = False,
        down: bool = False,
        factor: int = 2,
    ) -> None:
        super().__init__()
        self.out_channels = out_channels or channels
        self.up, self.down, self.factor = up, down, factor
        norm_kwargs = dict(norm_kwargs) if norm_kwargs is not None else {"num_groups": 32}

        self.in_norm = create_sparse_norm(norm, channels, **norm_kwargs)
        self.in_act = create_act(act, **(act_kwargs or {}))
        self.in_conv = fvnn.SparseConv3d(channels, self.out_channels, 3, 1, bias=True)
        self.emb_layers = nn.Sequential(nn.SiLU(), nn.Linear(embed_dim, 2 * self.out_channels))
        self.out_norm = create_sparse_norm(norm, self.out_channels, **norm_kwargs)
        self.out_act = create_act(act, **(act_kwargs or {}))
        self.dropout = nn.Dropout(dropout)
        self.out_conv = fvnn.SparseConv3d(self.out_channels, self.out_channels, 3, 1, bias=True)
        self.skip = nn.Linear(channels, self.out_channels) if self.out_channels != channels else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def _resample(self, x: Tensor, grid: "GridBatch", target_grid: Optional["GridBatch"]) -> Tuple[Tensor, "GridBatch"]:
        data = grid.ijk.jagged_like(x)
        if self.down:
            out, new_grid = grid.avg_pool(self.factor, data, stride=self.factor, coarse_grid=target_grid)
        else:
            out, new_grid = grid.refine(self.factor, data, fine_grid=target_grid)
        return out.jdata, new_grid

    def forward(
        self,
        x: Tensor,
        grid: "GridBatch",
        emb: Tensor,
        plans: Dict[int, "ConvolutionPlan"],
        target_grid: Optional["GridBatch"] = None,
    ) -> Tuple[Tensor, "GridBatch"]:
        if id(grid) not in plans:
            plans[id(grid)] = _submanifold_plan(grid)
        h = self.in_norm(x, grid.ijk.joffsets)
        if self.in_act is not None:
            h = self.in_act(h)
        if self.up or self.down:
            h, new_grid = self._resample(h, grid, target_grid)
            x, _ = self._resample(x, grid, new_grid)
            grid = new_grid
            if id(grid) not in plans:
                plans[id(grid)] = _submanifold_plan(grid)
        plan = plans[id(grid)]
        h = self.in_conv(grid.ijk.jagged_like(h), plan).jdata

        emb_out = self.emb_layers(emb)
        scale, shift = emb_out.chunk(2, dim=-1)
        jidx = grid.ijk.jidx.long()
        h = self.out_norm(h, grid.ijk.joffsets) * (1 + scale[jidx]) + shift[jidx]
        if self.out_act is not None:
            h = self.out_act(h)
        h = self.out_conv(grid.ijk.jagged_like(self.dropout(h)), plan).jdata
        return self.skip(x) + h, grid


class SparseAttentionBlock(nn.Module):
    r"""Per-grid multi-head self-attention over all voxels of each batch element.

    Args:
        channels: Feature channels.
        num_heads: Number of attention heads.
        norm: Normalization name or callable.
        norm_kwargs: Extra normalization arguments (defaults to $32$ groups).
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 1,
        norm: Union[str, Callable] = "group_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        norm_kwargs = dict(norm_kwargs) if norm_kwargs is not None else {"num_groups": 32}
        self.norm = create_sparse_norm(norm, channels, **norm_kwargs)
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: Tensor, offsets: Tensor) -> Tensor:
        qkv = self.qkv(self.norm(x, offsets))
        values = torch.empty_like(x)
        head_dim = x.shape[1] // self.num_heads
        for b in range(offsets.numel() - 1):
            start, end = int(offsets[b]), int(offsets[b + 1])
            part = qkv[start:end].reshape(1, end - start, self.num_heads, 3 * head_dim).permute(0, 2, 1, 3)
            q, k, v = part.chunk(3, dim=-1)
            out = F.scaled_dot_product_attention(q, k, v)
            values[start:end] = out.permute(0, 2, 1, 3).reshape(end - start, -1)
        return x + self.proj(values)


class XCubeSparseUNet(nn.Module):
    r"""Sparse latent diffusion UNet operating on the voxels of a latent grid.

    A sparse-convolution port of the LDM UNet: timestep-conditioned residual blocks with scale-shift
    norm, average-pool / nearest-refine resampling between levels, and per-grid attention at the
    configured downsample rates.

    Args:
        in_channels: Input feature channels.
        model_channels: Base channel count.
        out_channels: Output channels (defaults to `in_channels`).
        num_res_blocks: Residual blocks per level.
        channel_mult: Channel multiplier per level.
        attention_resolutions: Downsample rates at which attention blocks are inserted.
        num_heads: Attention heads.
        use_middle_attention: Insert an attention block in the middle stage.
        dropout: Dropout probability.
        norm: Normalization name or callable.
        norm_kwargs: Extra normalization arguments (defaults to $32$ groups).
        act: Activation name, callable or `None`.
        act_kwargs: Extra activation arguments.
    """

    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        out_channels: Optional[int] = None,
        num_res_blocks: int = 2,
        channel_mult: Sequence[int] = (1, 2, 4, 8),
        attention_resolutions: Sequence[int] = (),
        num_heads: int = 8,
        use_middle_attention: bool = False,
        dropout: float = 0.0,
        norm: Union[str, Callable] = "group_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        act: Union[str, Callable, None] = "silu",
        act_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.model_channels = model_channels
        self.out_channels = out_channels or in_channels
        norm_kwargs = dict(norm_kwargs) if norm_kwargs is not None else {"num_groups": 32}
        block_kwargs: Dict[str, Any] = dict(norm=norm, norm_kwargs=norm_kwargs, act=act, act_kwargs=act_kwargs)
        attention_kwargs: Dict[str, Any] = dict(num_heads=num_heads, norm=norm, norm_kwargs=norm_kwargs)

        embed_dim = 4 * model_channels
        self.time_emb = nn.Sequential(nn.Linear(model_channels, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim))

        self.encoder_blocks = nn.ModuleList(
            [nn.ModuleList([fvnn.SparseConv3d(in_channels, model_channels, 3, 1, bias=True)])]
        )
        encoder_channels = [model_channels]
        current = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers: List[nn.Module] = [
                    SparseResBlock(current, embed_dim, model_channels * mult, dropout, **block_kwargs)
                ]
                current = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(SparseAttentionBlock(current, **attention_kwargs))
                self.encoder_blocks.append(nn.ModuleList(layers))
                encoder_channels.append(current)
            if level < len(channel_mult) - 1:
                self.encoder_blocks.append(
                    nn.ModuleList([SparseResBlock(current, embed_dim, current, dropout, down=True, **block_kwargs)])
                )
                encoder_channels.append(current)
                ds *= 2

        middle: List[nn.Module] = [SparseResBlock(current, embed_dim, None, dropout, **block_kwargs)]
        if use_middle_attention:
            middle.append(SparseAttentionBlock(current, **attention_kwargs))
        middle.append(SparseResBlock(current, embed_dim, None, dropout, **block_kwargs))
        self.middle_block = nn.ModuleList(middle)

        self.decoder_blocks = nn.ModuleList()
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                skip_channels = encoder_channels.pop()
                layers = [
                    SparseResBlock(current + skip_channels, embed_dim, model_channels * mult, dropout, **block_kwargs)
                ]
                current = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(SparseAttentionBlock(current, **attention_kwargs))
                if level > 0 and i == num_res_blocks:
                    layers.append(SparseResBlock(current, embed_dim, current, dropout, up=True, **block_kwargs))
                    ds //= 2
                self.decoder_blocks.append(nn.ModuleList(layers))

        self.out_norm = create_sparse_norm(norm, current, **norm_kwargs)
        self.out_act = create_act(act, **(act_kwargs or {}))
        self.out_conv = fvnn.SparseConv3d(current, self.out_channels, 3, 1, bias=True)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def _run_layers(
        self,
        layers: Iterable[nn.Module],
        x: Tensor,
        grid: "GridBatch",
        emb: Tensor,
        plans: Dict[int, "ConvolutionPlan"],
        target_grid: Optional["GridBatch"] = None,
    ) -> Tuple[Tensor, "GridBatch"]:
        for layer in layers:
            if isinstance(layer, SparseResBlock):
                x, grid = layer(x, grid, emb, plans, target_grid)
            elif isinstance(layer, SparseAttentionBlock):
                x = layer(x, grid.ijk.joffsets)
            else:
                if id(grid) not in plans:
                    plans[id(grid)] = _submanifold_plan(grid)
                x = layer(grid.ijk.jagged_like(x), plans[id(grid)]).jdata
        return x, grid

    def forward(self, x: Tensor, grid: "GridBatch", timesteps: Tensor) -> Tensor:
        r"""Predict the diffusion target for the voxels of `grid` at the given timesteps.

        Args:
            x: Per-voxel input features.
            grid: Latent grid.
            timesteps: Diffusion timesteps, one per batch element (or a scalar).

        Returns:
            The per-voxel prediction.

        Shape:
            - `x`: $(N, C_\text{in})$, `timesteps`: $(B,)$ or scalar.
            - Output: $(N, C_\text{out})$
        """
        if timesteps.dim() == 0:
            timesteps = timesteps.expand(grid.grid_count).to(grid.device)
        emb = self.time_emb(timestep_encoding(timesteps, self.model_channels))

        plans: Dict[int, "ConvolutionPlan"] = {}
        hs: List[Tuple[Tensor, "GridBatch"]] = []
        for block in self.encoder_blocks:
            assert isinstance(block, nn.ModuleList)
            x, grid = self._run_layers(block, x, grid, emb, plans)
            hs.append((x, grid))
        x, grid = self._run_layers(self.middle_block, x, grid, emb, plans)
        for block in self.decoder_blocks:
            assert isinstance(block, nn.ModuleList)
            skip_x, _ = hs.pop()
            x = torch.cat([skip_x, x], dim=1)
            target_grid = hs[-1][1] if hs else None
            x, grid = self._run_layers(block, x, grid, emb, plans, target_grid)

        x = self.out_norm(x, grid.ijk.joffsets)
        if self.out_act is not None:
            x = self.out_act(x)
        if id(grid) not in plans:
            plans[id(grid)] = _submanifold_plan(grid)
        return self.out_conv(grid.ijk.jagged_like(x), plans[id(grid)]).jdata


class DenseResBlock(nn.Module):
    r"""Dense diffusion UNet residual block with optional scale-shift conditioning and resampling.

    Args:
        channels: Input feature channels.
        embed_dim: Timestep embedding dimension.
        out_channels: Output channels (defaults to `channels`).
        dropout: Dropout probability.
        norm: Normalization name or callable.
        norm_kwargs: Extra normalization arguments (defaults to $32$ groups).
        act: Activation name, callable or `None`.
        act_kwargs: Extra activation arguments.
        use_scale_shift_norm: FiLM-style conditioning instead of additive embedding.
        up: Nearest-upsample by $2$ before the first convolution.
        down: Average-pool by $2$ before the first convolution.
    """

    def __init__(
        self,
        channels: int,
        embed_dim: int,
        out_channels: Optional[int] = None,
        dropout: float = 0.0,
        norm: Union[str, Callable] = "group_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        act: Union[str, Callable, None] = "silu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        use_scale_shift_norm: bool = False,
        up: bool = False,
        down: bool = False,
    ) -> None:
        super().__init__()
        self.out_channels = out_channels or channels
        self.use_scale_shift_norm = use_scale_shift_norm
        self.up, self.down = up, down
        norm_kwargs = dict(norm_kwargs) if norm_kwargs is not None else {"num_groups": 32}

        in_norm = create_norm(norm, channels, dim=3, **norm_kwargs)
        out_norm = create_norm(norm, self.out_channels, dim=3, **norm_kwargs)
        assert in_norm is not None and out_norm is not None
        self.in_norm = in_norm
        self.in_act = create_act(act, **(act_kwargs or {}))
        self.in_conv = nn.Conv3d(channels, self.out_channels, 3, padding=1)
        emb_out_channels = 2 * self.out_channels if use_scale_shift_norm else self.out_channels
        self.emb_layers = nn.Sequential(nn.SiLU(), nn.Linear(embed_dim, emb_out_channels))
        self.out_norm = out_norm
        self.out_act = create_act(act, **(act_kwargs or {}))
        self.dropout = nn.Dropout(dropout)
        self.out_conv = nn.Conv3d(self.out_channels, self.out_channels, 3, padding=1)
        self.skip = nn.Conv3d(channels, self.out_channels, 1) if self.out_channels != channels else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.out_conv.weight)
        if self.out_conv.bias is not None:
            nn.init.zeros_(self.out_conv.bias)

    def _resample(self, x: Tensor) -> Tensor:
        if self.down:
            return F.avg_pool3d(x, kernel_size=2, stride=2)
        return F.interpolate(x, scale_factor=2, mode="nearest")

    def forward(self, x: Tensor, emb: Tensor) -> Tensor:
        h = self.in_norm(x)
        if self.in_act is not None:
            h = self.in_act(h)
        if self.up or self.down:
            h = self._resample(h)
            x = self._resample(x)
        h = self.in_conv(h)

        emb_out = self.emb_layers(emb)[..., None, None, None]
        if self.use_scale_shift_norm:
            scale, shift = emb_out.chunk(2, dim=1)
            h = self.out_norm(h) * (1 + scale) + shift
        else:
            h = self.out_norm(h + emb_out)
        if self.out_act is not None:
            h = self.out_act(h)
        return self.skip(x) + self.out_conv(self.dropout(h))


class DenseAttentionBlock(nn.Module):
    r"""Dense multi-head self-attention over all spatial positions.

    Args:
        channels: Feature channels.
        num_heads: Number of attention heads.
        norm: Normalization name or callable.
        norm_kwargs: Extra normalization arguments (defaults to $32$ groups).
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 1,
        norm: Union[str, Callable] = "group_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        norm_kwargs = dict(norm_kwargs) if norm_kwargs is not None else {"num_groups": 32}
        norm_module = create_norm(norm, channels, dim=3, **norm_kwargs)
        assert norm_module is not None
        self.norm = norm_module
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: Tensor) -> Tensor:
        batch, channels, *spatial = x.shape
        flat = x.reshape(batch, channels, -1)
        qkv = self.qkv(self.norm(x).reshape(batch, channels, -1).transpose(1, 2))
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.reshape(batch, -1, self.num_heads, channels // self.num_heads).transpose(1, 2)
        k = k.reshape(batch, -1, self.num_heads, channels // self.num_heads).transpose(1, 2)
        v = v.reshape(batch, -1, self.num_heads, channels // self.num_heads).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(batch, -1, channels)
        out = self.proj(out).transpose(1, 2)
        return (flat + out).reshape(batch, channels, *spatial)


class XCubeDenseUNet(nn.Module):
    r"""Dense 3D latent diffusion UNet for the coarse XCube stage.

    The standard LDM UNet with 3D convolutions, operating on the dense tensor of a fully occupied latent
    grid.

    Args:
        in_channels: Input feature channels.
        model_channels: Base channel count.
        out_channels: Output channels (defaults to `in_channels`).
        num_res_blocks: Residual blocks per level.
        channel_mult: Channel multiplier per level.
        attention_resolutions: Downsample rates at which attention blocks are inserted.
        num_heads: Attention heads.
        use_scale_shift_norm: FiLM-style timestep conditioning.
        dropout: Dropout probability.
        norm: Normalization name or callable.
        norm_kwargs: Extra normalization arguments (defaults to $32$ groups).
        act: Activation name, callable or `None`.
        act_kwargs: Extra activation arguments.
    """

    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        out_channels: Optional[int] = None,
        num_res_blocks: int = 2,
        channel_mult: Sequence[int] = (1, 2, 4, 8),
        attention_resolutions: Sequence[int] = (),
        num_heads: int = 8,
        use_scale_shift_norm: bool = True,
        dropout: float = 0.0,
        norm: Union[str, Callable] = "group_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        act: Union[str, Callable, None] = "silu",
        act_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.model_channels = model_channels
        self.out_channels = out_channels or in_channels
        norm_kwargs = dict(norm_kwargs) if norm_kwargs is not None else {"num_groups": 32}
        attention_kwargs: Dict[str, Any] = dict(num_heads=num_heads, norm=norm, norm_kwargs=norm_kwargs)

        embed_dim = 4 * model_channels
        self.time_emb = nn.Sequential(nn.Linear(model_channels, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim))

        def res_block(channels: int, out: Optional[int] = None, up: bool = False, down: bool = False) -> DenseResBlock:
            return DenseResBlock(
                channels,
                embed_dim,
                out,
                dropout,
                norm=norm,
                norm_kwargs=norm_kwargs,
                act=act,
                act_kwargs=act_kwargs,
                use_scale_shift_norm=use_scale_shift_norm,
                up=up,
                down=down,
            )

        self.encoder_blocks = nn.ModuleList([nn.ModuleList([nn.Conv3d(in_channels, model_channels, 3, padding=1)])])
        encoder_channels = [model_channels]
        current = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers: List[nn.Module] = [res_block(current, model_channels * mult)]
                current = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(DenseAttentionBlock(current, **attention_kwargs))
                self.encoder_blocks.append(nn.ModuleList(layers))
                encoder_channels.append(current)
            if level < len(channel_mult) - 1:
                self.encoder_blocks.append(nn.ModuleList([res_block(current, down=True)]))
                encoder_channels.append(current)
                ds *= 2

        self.middle_block = nn.ModuleList(
            [res_block(current), DenseAttentionBlock(current, **attention_kwargs), res_block(current)]
        )

        self.decoder_blocks = nn.ModuleList()
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                skip_channels = encoder_channels.pop()
                layers = [res_block(current + skip_channels, model_channels * mult)]
                current = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(DenseAttentionBlock(current, **attention_kwargs))
                if level > 0 and i == num_res_blocks:
                    layers.append(res_block(current, up=True))
                    ds //= 2
                self.decoder_blocks.append(nn.ModuleList(layers))

        out_norm = create_norm(norm, current, dim=3, **norm_kwargs)
        assert out_norm is not None
        self.out_norm = out_norm
        self.out_act = create_act(act, **(act_kwargs or {}))
        self.out_conv = nn.Conv3d(current, self.out_channels, 3, padding=1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.out_conv.weight)
        if self.out_conv.bias is not None:
            nn.init.zeros_(self.out_conv.bias)

    @staticmethod
    def _run_layers(layers: Iterable[nn.Module], x: Tensor, emb: Tensor) -> Tensor:
        for layer in layers:
            x = layer(x, emb) if isinstance(layer, DenseResBlock) else layer(x)
        return x

    def forward(self, x: Tensor, timesteps: Tensor) -> Tensor:
        r"""Predict the diffusion target for a dense latent volume at the given timesteps.

        Args:
            x: Dense latent volume.
            timesteps: Diffusion timesteps, one per batch element (or a scalar).

        Returns:
            The prediction.

        Shape:
            - `x`: $(B, C, D, H, W)$, `timesteps`: $(B,)$ or scalar.
            - Output: $(B, C_\text{out}, D, H, W)$
        """
        if timesteps.dim() == 0:
            timesteps = timesteps.expand(x.shape[0]).to(x.device)
        emb = self.time_emb(timestep_encoding(timesteps, self.model_channels))

        hs = []
        for block in self.encoder_blocks:
            assert isinstance(block, nn.ModuleList)
            x = self._run_layers(block, x, emb)
            hs.append(x)
        x = self._run_layers(self.middle_block, x, emb)
        for block in self.decoder_blocks:
            assert isinstance(block, nn.ModuleList)
            x = torch.cat([x, hs.pop()], dim=1)
            x = self._run_layers(block, x, emb)

        x = self.out_norm(x)
        if self.out_act is not None:
            x = self.out_act(x)
        return self.out_conv(x)


class XCubeDiffusion(BaseModel):
    r"""XCube latent diffusion model over the latent grid of a (frozen) `XCubeVAE`.

    :arxiv: [XCube](https://arxiv.org/abs/2312.03806) stage-two model: denoises Gaussian noise on a
    latent voxel grid with a 3D UNet (`v`-prediction DDIM by default) and decodes the result with the
    VAE into a sparse voxel hierarchy. The coarse variant runs a dense UNet on a fully occupied latent
    grid; the fine variant runs a sparse UNet on the topology generated by the coarse stage, conditioned
    on its predicted normals and the voxel coordinates.

    Args:
        vae: First-stage VAE (frozen).
        model_channels: Base UNet channel count.
        channel_mult: Channel multiplier per UNet level.
        num_res_blocks: Residual blocks per level.
        attention_resolutions: Downsample rates at which attention blocks are inserted.
        num_heads: Attention heads.
        dense: Use the dense UNet (coarse stage) instead of the sparse UNet (fine stage).
        latent_size: Dense latent grid extent per axis, used to build the sampling grid of the coarse
            stage.
        pos_embed_ijk: Concatenate the voxel `ijk` coordinates to the UNet input.
        normal_cond: Concatenate per-voxel normals (from the coarse stage) to the UNet input.
        use_middle_attention: Middle attention block of the sparse UNet.
        use_scale_shift_norm: FiLM-style timestep conditioning of the dense UNet.
        num_train_timesteps: Diffusion steps of the noise schedule.
        beta_start: First value of the linear $\beta$ schedule.
        beta_end: Last value of the linear $\beta$ schedule.
        prediction_type: Diffusion parameterization (`"epsilon"`, `"sample"` or `"v_prediction"`).
        norm: Normalization name or callable for the UNet.
        norm_kwargs: Extra normalization arguments (defaults to $32$ groups).
        act: Activation name, callable or `None`.
        act_kwargs: Extra activation arguments.

    Example:
        ```python
        coarse = create_model("xcube-diffusion-coarse-nvidia.shapenet-chair", task="base", pretrained=True)
        fine = create_model("xcube-diffusion-fine-nvidia.shapenet-chair", task="base", pretrained=True)
        out = coarse.sample(batch_size=4, num_steps=100)
        out = fine.sample(grid=out["grid"], normal=out["normal"], num_steps=100)
        ```
    """

    def __init__(
        self,
        vae: XCubeVAE,
        model_channels: int,
        channel_mult: Sequence[int] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        attention_resolutions: Sequence[int] = (),
        num_heads: int = 8,
        dense: bool = False,
        latent_size: int = 16,
        pos_embed_ijk: bool = False,
        normal_cond: bool = False,
        use_middle_attention: bool = True,
        use_scale_shift_norm: bool = True,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        prediction_type: str = "v_prediction",
        norm: Union[str, Callable] = "group_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        act: Union[str, Callable, None] = "silu",
        act_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(vae.in_channels)
        self.vae = vae
        self.vae.requires_grad_(False)
        self.dense = dense
        self.latent_size = latent_size
        self.pos_embed_ijk = pos_embed_ijk
        self.normal_cond = normal_cond

        in_channels = vae.latent_channels + (3 if pos_embed_ijk else 0) + (3 if normal_cond else 0)
        self.unet: nn.Module
        if dense:
            self.unet = XCubeDenseUNet(
                in_channels,
                model_channels,
                out_channels=vae.latent_channels,
                num_res_blocks=num_res_blocks,
                channel_mult=channel_mult,
                attention_resolutions=attention_resolutions,
                num_heads=num_heads,
                use_scale_shift_norm=use_scale_shift_norm,
                norm=norm,
                norm_kwargs=norm_kwargs,
                act=act,
                act_kwargs=act_kwargs,
            )
        else:
            self.unet = XCubeSparseUNet(
                in_channels,
                model_channels,
                out_channels=vae.latent_channels,
                num_res_blocks=num_res_blocks,
                channel_mult=channel_mult,
                attention_resolutions=attention_resolutions,
                num_heads=num_heads,
                use_middle_attention=use_middle_attention,
                norm=norm,
                norm_kwargs=norm_kwargs,
                act=act,
                act_kwargs=act_kwargs,
            )
        self.scheduler = DDIMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            prediction_type=prediction_type,
        )
        self.register_buffer("scale_factor", torch.ones(()))

    def latent_grid(self, batch_size: int, device: Union[str, torch.device]) -> "GridBatch":
        r"""Build the fully occupied latent grid used for unconditional (coarse) sampling.

        Args:
            batch_size: Number of grids.
            device: Target device.

        Returns:
            A dense latent grid of extent `latent_size` per axis, centered at the origin.
        """
        voxel_size = self.vae.voxel_size * 2 ** (self.vae.num_levels - 1)
        return GridBatch.from_dense(
            batch_size,
            [self.latent_size] * 3,
            [-self.latent_size // 2] * 3,
            voxel_sizes=[voxel_size] * 3,
            origins=[voxel_size / 2.0] * 3,
            device=device,
        )

    def _densify(self, x: Tensor, grid: "GridBatch") -> Tensor:
        ijk = grid.ijk.jdata.long()
        ijk_min = ijk.min(dim=0).values
        ijk = ijk - ijk_min
        dims = [int(d) + 1 for d in ijk.max(dim=0).values]
        dense = x.new_zeros(grid.grid_count, *dims, x.shape[1])
        dense[grid.ijk.jidx.long(), ijk[:, 0], ijk[:, 1], ijk[:, 2]] = x
        return dense.permute(0, 4, 1, 2, 3).contiguous()

    def _sparsify(self, dense: Tensor, grid: "GridBatch") -> Tensor:
        ijk = grid.ijk.jdata.long()
        ijk = ijk - ijk.min(dim=0).values
        return dense.permute(0, 2, 3, 4, 1)[grid.ijk.jidx.long(), ijk[:, 0], ijk[:, 1], ijk[:, 2]]

    def denoise(self, x: Tensor, grid: "GridBatch", timesteps: Tensor, normal: Optional[Tensor] = None) -> Tensor:
        r"""Run the denoising UNet on noisy latents.

        Args:
            x: Noisy latent features.
            grid: Latent grid.
            timesteps: Diffusion timesteps, one per batch element (or a scalar).
            normal: Per-voxel conditioning normals (required when `normal_cond` is enabled).

        Returns:
            The model prediction (per `prediction_type`).

        Shape:
            - `x`: $(N, C_\text{latent})$, `normal`: $(N, 3)$
            - Output: $(N, C_\text{latent})$
        """
        feats = [x]
        if self.pos_embed_ijk:
            feats.append(grid.ijk.jdata.float())
        if self.normal_cond:
            if normal is None:
                raise ValueError("This model was built with `normal_cond=True` but no normals were given.")
            feats.append(normal)
        h = torch.cat(feats, dim=1) if len(feats) > 1 else x
        if self.dense:
            assert isinstance(self.unet, XCubeDenseUNet)
            return self._sparsify(self.unet(self._densify(h, grid), timesteps), grid)
        assert isinstance(self.unet, XCubeSparseUNet)
        return self.unet(h, grid, timesteps)

    @torch.no_grad()
    def sample(
        self,
        batch_size: Optional[int] = None,
        grid: Optional["GridBatch"] = None,
        normal: Optional[Tensor] = None,
        num_steps: int = 100,
        eta: float = 1.0,
        generator: Optional[torch.Generator] = None,
        decode: bool = True,
    ) -> Dict[str, Any]:
        r"""Sample latents with DDIM and decode them with the VAE.

        Args:
            batch_size: Number of unconditional samples (coarse stage; builds a dense latent grid).
            grid: Latent grid to sample on (fine stage; the structure generated by the coarse stage).
            normal: Per-voxel conditioning normals aligned with `grid`.
            num_steps: DDIM steps.
            eta: DDIM noise scale ($1.0$ matches the reference sampling).
            generator: Random generator.
            decode: Decode the sampled latents with the VAE.

        Returns:
            The VAE decoder output dict (or `{"z", "grid"}` when `decode` is false). Normals are
            renormalized to unit length.
        """
        if grid is None:
            if batch_size is None:
                raise ValueError("Provide either `batch_size` or `grid`.")
            device = next(self.parameters()).device
            grid = self.latent_grid(batch_size, device)
        if normal is not None:
            normal = normal / (normal.norm(dim=1, keepdim=True) + 1e-6)

        scale = self.scale_factor
        assert isinstance(scale, Tensor)
        z = torch.randn(
            (grid.total_voxels, self.vae.latent_channels),
            generator=generator,
            device=grid.device,
        )
        self.scheduler.set_timesteps(num_steps, device=grid.device)
        for t in self.scheduler.timesteps:
            pred = self.denoise(z, grid, t, normal=normal)
            z = self.scheduler.step(pred, int(t), z, eta=eta, generator=generator)
        z = z / scale

        if not decode:
            return {"z": z, "grid": grid}
        out = self.vae.decode(z, grid)
        if "normal" in out:
            out["normal"] = out["normal"] / (out["normal"].norm(dim=1, keepdim=True) + 1e-6)
        return out

    def forward(
        self,
        pos: Tensor,
        batch: Tensor,
        normal: Optional[Tensor] = None,
        timesteps: Optional[Tensor] = None,
        noise: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        r"""Compute the diffusion training pair (prediction, target) for a point cloud.

        Args:
            pos: Point positions.
            batch: Batch indices.
            normal: Per-point normals (encoder input and, when `normal_cond` is enabled, splatted onto
                the latent grid as conditioning).
            timesteps: Per-sample timesteps (random when `None`).
            noise: Diffusion noise (random when `None`).

        Returns:
            A dict with the model prediction `pred` and the diffusion target `target`.

        Shape:
            - `pos`: $(N, 3)$, `batch`: $(N,)$, `normal`: $(N, 3)$
        """
        with torch.no_grad():
            z, grid = self.vae.encode(pos, batch, normal=normal, sample=True)
        scale = self.scale_factor
        assert isinstance(scale, Tensor)
        z = z * scale

        if noise is None:
            noise = torch.randn_like(z)
        if timesteps is None:
            timesteps = torch.randint(0, self.scheduler.num_train_timesteps, (grid.grid_count,), device=z.device)
        timesteps_per_voxel = timesteps[grid.ijk.jidx.long()]
        noisy = self.scheduler.add_noise(z, noise, timesteps_per_voxel)

        cond_normal: Optional[Tensor] = None
        if self.normal_cond:
            if normal is None:
                raise ValueError("This model was built with `normal_cond=True` but no normals were given.")
            batch_size = grid.grid_count
            points = JaggedTensor.from_data_and_indices(pos, batch.int(), batch_size)
            cond_normal = grid.splat_trilinear(points, points.jagged_like(normal)).jdata
            cond_normal = cond_normal / (cond_normal.norm(dim=1, keepdim=True) + 1e-6)
        pred = self.denoise(noisy, grid, timesteps, normal=cond_normal)

        if self.scheduler.prediction_type == "epsilon":
            target = noise
        elif self.scheduler.prediction_type == "v_prediction":
            target = self.scheduler.get_velocity(z, noise, timesteps_per_voxel)
        else:
            target = z
        return {"pred": pred, "target": target}


def _shapenet_coarse_vae_hparams() -> Dict[str, Any]:
    return dict(
        in_channels=3,
        encoder_channels=32,
        channels=(64, 128, 256, 512),
        latent_channels=16,
        voxel_size=0.01,
        neck_bound=(8, 8, 8),
        use_normal=True,
        use_hash_tree=True,
        with_normal_head=True,
    )


def _shapenet_fine_vae_hparams() -> Dict[str, Any]:
    return dict(
        in_channels=3,
        encoder_channels=32,
        channels=(32, 64, 128),
        latent_channels=8,
        voxel_size=0.0025,
        neck_bound=None,
        use_normal=True,
        use_hash_tree=True,
        with_normal_head=True,
    )


def _shapenet_coarse_diffusion_hparams() -> Dict[str, Any]:
    return dict(
        model_channels=192,
        channel_mult=(1, 2, 4, 4),
        num_res_blocks=2,
        attention_resolutions=(1, 2, 4),
        num_heads=8,
        dense=True,
        latent_size=16,
        use_scale_shift_norm=True,
        prediction_type="v_prediction",
    )


def _shapenet_fine_diffusion_hparams(model_channels: int = 128) -> Dict[str, Any]:
    return dict(
        model_channels=model_channels,
        channel_mult=(1, 2, 2, 4),
        num_res_blocks=2,
        attention_resolutions=(4, 8),
        num_heads=8,
        dense=False,
        pos_embed_ijk=True,
        normal_cond=True,
        use_middle_attention=True,
        prediction_type="v_prediction",
    )


@register_model(
    "xcube-vae-coarse-nvidia.shapenet-chair",
    task="base",
    weights="hf://torch-pointcloud/xcube/xcube-vae-coarse-nvidia.shapenet-chair.pt",
    hparams=_shapenet_coarse_vae_hparams(),
)
def xcube_vae_coarse_shapenet_chair(**hparams: Any) -> XCubeVAE:
    return XCubeVAE(**hparams)


@register_model(
    "xcube-vae-fine-nvidia.shapenet-chair",
    task="base",
    weights="hf://torch-pointcloud/xcube/xcube-vae-fine-nvidia.shapenet-chair.pt",
    hparams=_shapenet_fine_vae_hparams(),
)
def xcube_vae_fine_shapenet_chair(**hparams: Any) -> XCubeVAE:
    return XCubeVAE(**hparams)


@register_model(
    "xcube-vae-coarse-nvidia.shapenet-car",
    task="base",
    weights="hf://torch-pointcloud/xcube/xcube-vae-coarse-nvidia.shapenet-car.pt",
    hparams=_shapenet_coarse_vae_hparams(),
)
def xcube_vae_coarse_shapenet_car(**hparams: Any) -> XCubeVAE:
    return XCubeVAE(**hparams)


@register_model(
    "xcube-vae-fine-nvidia.shapenet-car",
    task="base",
    weights="hf://torch-pointcloud/xcube/xcube-vae-fine-nvidia.shapenet-car.pt",
    hparams=_shapenet_fine_vae_hparams(),
)
def xcube_vae_fine_shapenet_car(**hparams: Any) -> XCubeVAE:
    return XCubeVAE(**hparams)


@register_model(
    "xcube-vae-coarse-nvidia.shapenet-plane",
    task="base",
    weights="hf://torch-pointcloud/xcube/xcube-vae-coarse-nvidia.shapenet-plane.pt",
    hparams=_shapenet_coarse_vae_hparams(),
)
def xcube_vae_coarse_shapenet_plane(**hparams: Any) -> XCubeVAE:
    return XCubeVAE(**hparams)


@register_model(
    "xcube-vae-fine-nvidia.shapenet-plane",
    task="base",
    weights="hf://torch-pointcloud/xcube/xcube-vae-fine-nvidia.shapenet-plane.pt",
    hparams=_shapenet_fine_vae_hparams(),
)
def xcube_vae_fine_shapenet_plane(**hparams: Any) -> XCubeVAE:
    return XCubeVAE(**hparams)


@register_model(
    "xcube-diffusion-coarse-nvidia.shapenet-chair",
    task="base",
    weights="hf://torch-pointcloud/xcube/xcube-diffusion-coarse-nvidia.shapenet-chair.pt",
    hparams=_shapenet_coarse_diffusion_hparams(),
)
def xcube_diffusion_coarse_shapenet_chair(**hparams: Any) -> XCubeDiffusion:
    return XCubeDiffusion(vae=XCubeVAE(**_shapenet_coarse_vae_hparams()), **hparams)


@register_model(
    "xcube-diffusion-fine-nvidia.shapenet-chair",
    task="base",
    weights="hf://torch-pointcloud/xcube/xcube-diffusion-fine-nvidia.shapenet-chair.pt",
    hparams=_shapenet_fine_diffusion_hparams(model_channels=64),
)
def xcube_diffusion_fine_shapenet_chair(**hparams: Any) -> XCubeDiffusion:
    return XCubeDiffusion(vae=XCubeVAE(**_shapenet_fine_vae_hparams()), **hparams)


@register_model(
    "xcube-diffusion-coarse-nvidia.shapenet-car",
    task="base",
    weights="hf://torch-pointcloud/xcube/xcube-diffusion-coarse-nvidia.shapenet-car.pt",
    hparams=_shapenet_coarse_diffusion_hparams(),
)
def xcube_diffusion_coarse_shapenet_car(**hparams: Any) -> XCubeDiffusion:
    return XCubeDiffusion(vae=XCubeVAE(**_shapenet_coarse_vae_hparams()), **hparams)


@register_model(
    "xcube-diffusion-fine-nvidia.shapenet-car",
    task="base",
    weights="hf://torch-pointcloud/xcube/xcube-diffusion-fine-nvidia.shapenet-car.pt",
    hparams=_shapenet_fine_diffusion_hparams(),
)
def xcube_diffusion_fine_shapenet_car(**hparams: Any) -> XCubeDiffusion:
    return XCubeDiffusion(vae=XCubeVAE(**_shapenet_fine_vae_hparams()), **hparams)


@register_model(
    "xcube-diffusion-coarse-nvidia.shapenet-plane",
    task="base",
    weights="hf://torch-pointcloud/xcube/xcube-diffusion-coarse-nvidia.shapenet-plane.pt",
    hparams=_shapenet_coarse_diffusion_hparams(),
)
def xcube_diffusion_coarse_shapenet_plane(**hparams: Any) -> XCubeDiffusion:
    return XCubeDiffusion(vae=XCubeVAE(**_shapenet_coarse_vae_hparams()), **hparams)


@register_model(
    "xcube-diffusion-fine-nvidia.shapenet-plane",
    task="base",
    weights="hf://torch-pointcloud/xcube/xcube-diffusion-fine-nvidia.shapenet-plane.pt",
    hparams=_shapenet_fine_diffusion_hparams(),
)
def xcube_diffusion_fine_shapenet_plane(**hparams: Any) -> XCubeDiffusion:
    return XCubeDiffusion(vae=XCubeVAE(**_shapenet_fine_vae_hparams()), **hparams)
