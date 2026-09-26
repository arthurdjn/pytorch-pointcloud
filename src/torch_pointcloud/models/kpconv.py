"""KPConv classification and segmentation models.

{{ paper("1904.08889") }}
"""

from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union, overload

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn.pool import radius, radius_graph

import torch_pointcloud.transforms as T
from torch_pointcloud.datasets.s3dis import S3DIS_CLASSES
from torch_pointcloud.layers import PoolLike, create_pool
from torch_pointcloud.layers.act import create_act
from torch_pointcloud.layers.kpconv_blocks import KPConvBlock, KPResidualBlock
from torch_pointcloud.layers.norms import create_norm
from torch_pointcloud.layers.voxel_grid_pool import VoxelGridPool
from torch_pointcloud.utils.conversion import ensure_tuple, ensure_tuple_size
from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.types import OptTensor, PooledFeaturesDict

from ._base import ClassificationModel, SemanticSegmentationModel
from ._registry import WeightsDict, register_model
from .pointnet2 import PointNet2Decoder


class KPFCNNEncoderBlock(nn.Module):
    """One encoder stage: an optional grid subsampling followed by `depth` `KPResidualBlock` blocks.

    Neighborhoods are recomputed once at the stage entry, using `pool_radius` for the strided first block
    and `radius` for the remaining ones.

    Args:
        downsample: Grid pooling applied before the first block. Makes that block strided.
    """

    def __init__(
        self,
        *,
        depth: int,
        radius: float,
        pool_radius: Optional[float] = None,
        max_num_neighbors: int,
        spatial_dim: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        kp_radius: Union[float, Sequence[float]],
        kp_sigma: Union[float, Sequence[float]],
        kp_influence: str = "linear",
        fixed_position: Literal["none", "center", "vertical"] = "center",
        aggregation_mode: str = "sum",
        deformable: Union[bool, Sequence[bool]] = False,
        modulated: Union[bool, Sequence[bool]] = False,
        bias: bool = False,
        act: Union[str, Callable, None] = "leaky_relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        downsample: Optional[VoxelGridPool] = None,
    ):
        super().__init__()
        self.max_num_neighbors = max_num_neighbors
        self.radius = radius
        self.pool_radius = pool_radius if pool_radius is not None else radius
        self.downsample = downsample
        extra_msg = "Expected encoder `{param_name}` to be of length `depth`."
        kp_radius = ensure_tuple_size(kp_radius, size=depth, extra_msg=extra_msg.format(param_name="kp_radius"))
        kp_sigma = ensure_tuple_size(kp_sigma, size=depth, extra_msg=extra_msg.format(param_name="kp_sigma"))
        deformable = ensure_tuple_size(deformable, size=depth, extra_msg=extra_msg.format(param_name="deformable"))
        modulated = ensure_tuple_size(modulated, size=depth, extra_msg=extra_msg.format(param_name="modulated"))

        self.blocks = nn.ModuleList()
        for i in range(depth):
            strided = downsample is not None and i == 0
            block = KPResidualBlock(
                spatial_dim=spatial_dim,
                in_channels=in_channels if i == 0 or (downsample is not None and i == 1) else out_channels,
                out_channels=in_channels if strided else out_channels,
                kernel_size=kernel_size,
                kp_radius=kp_radius[i],
                kp_sigma=kp_sigma[i],
                kp_influence=kp_influence,
                fixed_position=fixed_position,
                aggregation_mode=aggregation_mode,
                deformable=deformable[i],
                modulated=modulated[i],
                strided=strided,
                act=act,
                act_kwargs=act_kwargs,
                act_first=act_first,
                norm=norm,
                norm_kwargs=norm_kwargs,
                bias=bias,
            )
            self.blocks.append(block)

    @overload
    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        batch: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]: ...

    @overload
    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        batch: Tensor,
        return_inverse: Literal[True] = True,
    ) -> Tuple[Tensor, Tensor, Tensor, OptTensor]: ...

    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        batch: Tensor,
        return_inverse: bool = False,
    ) -> Any:
        inv = None
        pos_down, batch_down = pos, batch
        if self.downsample is not None:
            _, pos_down, batch_down, inv = self.downsample(x, pos, batch, return_inverse=True)

        # Pre-computed neighbors edge indices, for both strided and non-strided blocks.
        # For strided block (first block after downsampling),
        # compute edge indices between downsampled coords and original coords.
        # For non-strided blocks, then downsampled coords is the same as original coords, and corresponds to the
        # `radius_graph(pos)` output.
        row, col = radius(
            pos,
            pos_down,
            r=self.pool_radius if self.downsample is not None else self.radius,
            batch_x=batch,
            batch_y=batch_down,
            max_num_neighbors=self.max_num_neighbors,
        )
        edge_index = torch.stack([col, row], dim=0)

        for i, block in enumerate(self.blocks):
            if i == 1 and self.downsample is not None:
                edge_index = radius_graph(
                    pos_down,
                    r=self.radius,
                    batch=batch_down,
                    max_num_neighbors=self.max_num_neighbors,
                    loop=True,
                )
                pos = pos_down

            x = block(x, (pos, pos_down), edge_index)

        if return_inverse:
            return x, pos_down, batch_down, inv
        return x, pos_down, batch_down


class KPFCNNEncoder(nn.Module):
    r"""KP-FCNN encoder: `KPFCNNEncoderBlock` stages from finest to coarsest, every stage but the first preceded by a
    `VoxelGridPool` subsampling.

    Args:
        in_channels: Number of channels entering the first stage.
        depths: Number of residual blocks in each stage.
        grid_sizes: Grid size of each downsampling step, one per stage transition.
        radii: Neighborhood search radius of each stage.
        channels: Output channels of each stage.
        max_num_neighbors: Maximum number of neighbors queried at each stage.
        kernel_size: Number of kernel points of every KPConv.
        kp_sigma: Kernel extent of each stage.
        kp_radius: Kernel point radius of each stage.
        kp_influence: Influence function: `"constant"`, `"linear"`, or `"gaussian"`.
        fixed_position: Which kernel point is pinned: `"none"`, `"center"`, or `"vertical"`.
        aggregation_mode: `"sum"` over all kernel points, or `"closest"` to keep only the nearest one.
        deformable: Whether each stage uses deformable kernels.
        modulated: Whether each stage uses modulated deformable kernels.
        act: Activation function.
        act_kwargs: Optional keyword arguments for the activation factory.
        act_first: Whether the activation comes before the normalization in the MLPs.
        norm: Normalization layer.
        norm_kwargs: Optional keyword arguments for the normalization factory.
        bias: Whether the convolutions and MLPs use a bias.
        spatial_dim: Spatial dimension of the input point cloud.

    Inputs:
        x: Point features of shape $(N, \text{in\_channels})$.
        pos: Point coordinates of shape $(N, D)$.
        batch: Batch indices of shape $(N,)$.

    Outputs:
        Features, coordinates and batch indices at the coarsest stage. With `return_intermediates=True`, also one
        skip per downsampled stage: the features, coordinates, batch indices and pooling inverse entering that stage.
    """

    def __init__(
        self,
        in_channels: int,
        *,
        depths: Sequence[int],
        grid_sizes: Sequence[float],
        radii: Sequence[float],
        channels: Sequence[int],
        max_num_neighbors: Sequence[int],
        kernel_size: int,
        kp_sigma: Union[float, Sequence[float]],
        kp_radius: Union[float, Sequence[float]],
        kp_influence: str = "linear",
        fixed_position: Literal["none", "center", "vertical"] = "center",
        aggregation_mode: str = "sum",
        deformable: Union[bool, Sequence] = False,
        modulated: Union[bool, Sequence] = False,
        act: Union[str, Callable, None] = "leaky_relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: bool = False,
        spatial_dim: int = 3,
    ):
        super().__init__()
        depths = ensure_tuple(depths)
        n = len(depths)
        extra_msg = "Expected `{param_name}` to be of length `depths`."
        channels = ensure_tuple_size(channels, size=n, extra_msg=extra_msg.format(param_name="channels"))
        max_num_neighbors = ensure_tuple_size(
            max_num_neighbors,
            size=n,
            extra_msg=extra_msg.format(param_name="max_num_neighbors"),
        )
        grid_sizes = ensure_tuple_size(grid_sizes, size=n - 1, extra_msg="Encoder length `grid_sizes` != `depths` - 1.")
        kp_radius = ensure_tuple_size(kp_radius, size=n, extra_msg=extra_msg.format(param_name="kp_radius"))
        kp_sigma = ensure_tuple_size(kp_sigma, size=n, extra_msg=extra_msg.format(param_name="kp_sigma"))
        deformable = ensure_tuple_size(deformable, size=n, extra_msg=extra_msg.format(param_name="deformable"))
        modulated = ensure_tuple_size(modulated, size=n, extra_msg=extra_msg.format(param_name="modulated"))

        self.blocks = nn.ModuleList()
        for i in range(n):
            downsample: Optional[VoxelGridPool] = None
            if i > 0:
                downsample = VoxelGridPool(grid_size=grid_sizes[i - 1], reduce="max")

            block = KPFCNNEncoderBlock(
                downsample=downsample,
                radius=radii[i],
                pool_radius=radii[i - 1] if i > 0 else None,
                max_num_neighbors=max_num_neighbors[i],
                spatial_dim=spatial_dim,
                depth=depths[i],
                in_channels=in_channels,
                out_channels=channels[i],
                kernel_size=kernel_size,
                kp_radius=([kp_radius[i - 1]] + [kp_radius[i]] * (depths[i] - 1)) if i > 0 else kp_radius[i],
                kp_sigma=([kp_sigma[i - 1]] + [kp_sigma[i]] * (depths[i] - 1)) if i > 0 else kp_sigma[i],
                kp_influence=kp_influence,
                fixed_position=fixed_position,
                aggregation_mode=aggregation_mode,
                deformable=deformable[i],
                modulated=modulated[i],
                act=act,
                act_kwargs=act_kwargs,
                act_first=act_first,
                norm=norm,
                norm_kwargs=norm_kwargs,
                bias=bias,
            )
            self.blocks.append(block)
            in_channels = channels[i]

    @overload
    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        batch: Tensor,
        return_intermediates: Literal[True],
    ) -> Tuple[Tensor, Tensor, Tensor, List[PooledFeaturesDict]]: ...

    @overload
    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        batch: Tensor,
        return_intermediates: Literal[False] = False,
    ) -> Tuple[Tensor, Tensor, Tensor]: ...

    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        batch: Tensor,
        return_intermediates: bool = False,
    ) -> Any:
        intermediates: List[PooledFeaturesDict] = []
        for i, block in enumerate(self.blocks):
            intermediate: PooledFeaturesDict = {"x": x, "pos": pos, "batch": batch}
            x, pos, batch, inv = block(x, pos, batch, return_inverse=True)

            if return_intermediates and i > 0:
                intermediate["pooling_inverse"] = inv
                intermediates.append(intermediate)

        if return_intermediates:
            return x, pos, batch, intermediates
        return x, pos, batch


class KPFCNNClassification(ClassificationModel):
    """KPConv Network for classification tasks as described in the paper
    :arxiv: [KPConv: Flexible and Efficient Convolution for Point Clouds](https://arxiv.org/abs/1904.08889)
    by Hugues Thomas, Charles R. Qi, Jean-Emmanuel Deschaud, Beatriz Marcotegui, François Goulette, Leonidas J. Guibas.

    KPConv introduces a novel point convolution operator that uses kernel points to define the spatial extent and weights
    of the convolution. The kernel points are arranged in space to define the convolution pattern, with weights determined
    by their spatial correlation with input points. This allows for flexible and efficient convolution on irregular point
    clouds while maintaining permutation invariance and translation invariance. The network uses a hierarchical architecture
    with strided convolutions for spatial pooling and feature aggregation.

    Note:
        The implementation is based on the original paper and the authors' code
        :github: [KPConv-PyTorch](https://github.com/HuguesTHOMAS/KPConv-PyTorch).

    Important:
        This implementation was completely rewritten to be compatible with
        :github: [`torch-geometric`](https://github.com/pyg-team/pytorch_geometric) library.

    Args:
        in_channels: Number of input channels.
        num_classes: Number of output classes.
        spatial_dim: Spatial dimension of the input point cloud.
        stem_channels: Number of channels in the stem layer.
        stem_type: Type of stem layer to use.
        encoder_depths: List of depths for each encoder block,
            i.e. corresponds to the number of residual blocks at each level.
        encoder_channels: List of channels for each encoder block.
        encoder_num_neighbors: List of maximum number of neighbors for each encoder block.
        grid_sizes: List of grid sizes for each downsampling block.
        radii: Search radius for each downsampling block.
        kernel_size: Size of the kernel for each KPConv block.
        kp_radius: List of kernel radius for KPConv blocks, at each level.
        kp_sigma: List of kernel extent for KPConv blocks, at each level.
        kp_influence: Influence function to use for KPConv blocks. Options are "constant", "linear", "gaussian".
        fixed_position: Whether to fix the position of the kernel points in KPConv blocks. Options are "none", "center", "vertical".
        aggregation_mode: Aggregation mode to use for the KPConv blocks. Options are "sum", "mean", "max".
        deformable: Whether to use a deformable kernel for the KPConv blocks.
        modulated: Whether to use a modulated kernel in KPConv operation.
        norm: Normalization to use for the KPConv blocks.
        act: Activation function to use for the KPConv blocks.
        bias: Whether to use a bias for the KPConv blocks.
        dropout: Dropout rate before the classification head.
        global_pool: Global pooling method to use before the classification head. Options are "max", "mean".

    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        *,
        spatial_dim: int = 3,
        stem_channels: Optional[int] = None,
        stem_type: Literal["linear", "kpconv"] = "kpconv",
        encoder_depths: Sequence[int],
        encoder_channels: Sequence[int],
        encoder_num_neighbors: Sequence[int],
        grid_sizes: Sequence[float],
        radii: Sequence[float],
        kernel_size: int,
        kp_radius: Union[float, Sequence[float]],
        kp_sigma: Union[float, Sequence[float]],
        kp_influence: str = "linear",
        fixed_position: Literal["none", "center", "vertical"] = "center",
        aggregation_mode: str = "sum",
        deformable: Union[bool, Sequence] = False,
        modulated: Union[bool, Sequence] = False,
        act: Union[str, Callable, None] = "leaky_relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: bool = False,
        dropout: float = 0.0,
        global_pool: PoolLike = "max",
    ):
        super().__init__(in_channels, num_classes)
        self.spatial_dim = spatial_dim
        self.stem_channels = stem_channels
        self.stem_type = stem_type
        self.encoder_depths = encoder_depths
        self.encoder_channels = encoder_channels
        self.encoder_num_neighbors = encoder_num_neighbors
        self.grid_sizes = grid_sizes
        self.radii = radii
        self.kernel_size = kernel_size
        self.kp_radius = ensure_tuple_size(kp_radius, size=len(encoder_depths))
        self.kp_sigma = ensure_tuple_size(kp_sigma, size=len(encoder_depths))
        self.kp_influence = kp_influence
        self.fixed_position = fixed_position
        self.aggregation_mode = aggregation_mode
        self.deformable = deformable
        self.modulated = modulated
        self.act = act
        self.act_kwargs = act_kwargs
        self.act_first = act_first
        self.norm = norm
        self.norm_kwargs = norm_kwargs
        self.bias = bias
        self.dropout = dropout

        self.stem = self.configure_stem()
        self.encoder = self.configure_encoder()
        self.global_pool = create_pool(global_pool)
        self.head = self.configure_head()

    def configure_stem(self) -> Optional[nn.Module]:
        """Build the stem lifting the input features to `stem_channels`, or `None` when `stem_channels` is unset."""
        if self.stem_channels is None:
            return None
        if self.stem_type == "kpconv":
            return KPConvBlock(
                spatial_dim=self.spatial_dim,
                in_channels=self.in_channels,
                out_channels=self.stem_channels,
                kernel_size=self.kernel_size,
                kp_radius=self.kp_radius[0],
                kp_sigma=self.kp_sigma[0],
                kp_influence=self.kp_influence,
                fixed_position=self.fixed_position,
                aggregation_mode=self.aggregation_mode,
                act=self.act,
                act_kwargs=self.act_kwargs,
                norm=self.norm,
                norm_kwargs=self.norm_kwargs,
                bias=self.bias,
            )
        act = create_act(self.act, **(self.act_kwargs or {})) or nn.Identity()
        norm = create_norm(self.norm, self.stem_channels, **(self.norm_kwargs or {})) or nn.Identity()
        return nn.Sequential(nn.Linear(self.in_channels, self.stem_channels), norm, act)

    def configure_encoder(self) -> KPFCNNEncoder:
        """Build the `KPFCNNEncoder` backbone."""
        return KPFCNNEncoder(
            in_channels=self.in_channels if self.stem_channels is None else self.stem_channels,
            depths=self.encoder_depths,
            channels=self.encoder_channels,
            grid_sizes=self.grid_sizes,
            radii=self.radii,
            max_num_neighbors=self.encoder_num_neighbors,
            spatial_dim=self.spatial_dim,
            kernel_size=self.kernel_size,
            kp_radius=self.kp_radius,
            kp_sigma=self.kp_sigma,
            kp_influence=self.kp_influence,
            fixed_position=self.fixed_position,
            aggregation_mode=self.aggregation_mode,
            deformable=self.deformable,
            modulated=self.modulated,
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
        return nn.Linear(self.num_features, self.num_classes)

    def reset_classifier(self, num_classes: int, global_pool: Optional[PoolLike] = None, **kwargs: Any) -> None:
        """Resets the classification head with new parameters.

        Note:
            To set an empty classification head, use `num_classes=0`.

        Args:
            num_classes: Number of output classes.
            global_pool: Pooling method to aggregate point features ("max" or "mean"). If `None`, keeps the current
                pooling.
            **kwargs: Additional keyword arguments to pass to the classification head.
        """
        self.num_classes = num_classes
        if global_pool is not None:
            self.global_pool = create_pool(global_pool)
        self.head = self.configure_head()

    @overload
    def forward_features(
        self,
        x: OptTensor,
        pos: Tensor,
        batch: Tensor,
        return_intermediates: Literal[True],
    ) -> Tuple[Tensor, Tensor, Tensor, List[PooledFeaturesDict]]: ...

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
        r"""Forward pass of the encoder, returning pre-pooling features.

        Args:
            x: Point features of shape $(N, C)$. If `None`, `pos` is used as features, which requires
                `in_channels` to match the dimension of `pos`.
            pos: Point coordinates of shape $(N, D)$.
            batch: Batch indices for each point of shape $(N,)$.
            return_intermediates: Whether to also return the per-stage intermediates.

        Returns:
            A tuple `(x, pos, batch)` at the coarsest level, where `x` has shape
                $(N', \text{encoder\_channels}[-1])$ and $N'$ is the number of downsampled points.
                If `return_intermediates=True`, the per-stage intermediates are appended to the tuple.
        """
        if x is None:
            if self.in_channels != pos.size(1):
                raise ValueError(
                    f"Got `x=None` but the model expects in_channels={self.in_channels} features while `pos` has "
                    f"{pos.size(1)} channels; pass `x` of shape (N, {self.in_channels})."
                )
            x = pos

        if self.stem is not None:
            if self.stem_type == "kpconv":
                edge_index = radius_graph(
                    pos,
                    r=self.radii[0],
                    batch=batch,
                    max_num_neighbors=self.encoder_num_neighbors[0],
                    loop=True,
                )
                x = self.stem(x, pos, edge_index)
            else:
                x = self.stem(x)

        return self.encoder(x, pos, batch, return_intermediates=return_intermediates)

    def forward_head(self, x: Tensor, batch: Tensor, pre_logits: bool = False) -> Tensor:
        r"""Forward pass of the classification head from pre-pooling features.

        Args:
            x: Pre-pooling features of shape $(N', \text{encoder\_channels}[-1])$ where $N'$ is the
                number of downsampled points.
            batch: Batch indices for each downsampled point of shape $(N',)$.
            pre_logits: Whether to return pre-logits. Defaults to False.

        Returns:
            Classification logits of shape $(B, \text{num\_classes})$.
        """
        x = self.global_pool(x, batch)
        if self.dropout:
            x = F.dropout(x, p=float(self.dropout), training=self.training)
        return x if pre_logits else self.head(x)

    def forward(self, x: OptTensor, pos: Tensor, batch: Tensor) -> Tensor:
        r"""Forward pass of the classification model.

        Args:
            x: Point features of shape $(N, C)$. If `None`, `pos` is used as features, which requires
                `in_channels` to match the dimension of `pos`.
            pos: Point coordinates of shape $(N, D)$.
            batch: Batch indices for each point of shape $(N,)$.

        Returns:
            Classification logits of shape $(B, \text{num\_classes})$.
        """
        x, _, batch = self.forward_features(x, pos, batch)
        return self.forward_head(x, batch, pre_logits=False)


class KPFCNNSegmentation(SemanticSegmentationModel):
    """KPConv Network for segmentation tasks as described in the paper
    :arxiv: [KPConv: Flexible and Efficient Convolution for Point Clouds](https://arxiv.org/abs/1904.08889)
    by Hugues Thomas, Charles R. Qi, Jean-Emmanuel Deschaud, Beatriz Marcotegui, François Goulette, Leonidas J. Guibas.

    KPConv introduces a novel point convolution operator that uses kernel points to define the spatial extent and weights
    of the convolution. The kernel points are arranged in space to define the convolution pattern, with weights determined
    by their spatial correlation with input points. This allows for flexible and efficient convolution on irregular point
    clouds while maintaining permutation invariance and translation invariance. The network uses a hierarchical architecture
    with strided convolutions for spatial pooling and feature aggregation.

    Note:
        The implementation is based on the original paper and the authors' code
        :github: [KPConv-PyTorch](https://github.com/HuguesTHOMAS/KPConv-PyTorch).

    Important:
        This implementation was completely rewritten to be compatible with
        :github: [`torch-geometric`](https://github.com/pyg-team/pytorch_geometric) library.

    Args:
        in_channels: Number of input channels.
        num_classes: Number of output classes.
        spatial_dim: Spatial dimension of the input point cloud.
        stem_channels: Number of channels in the stem layer.
        stem_type: Type of stem layer to use.
        encoder_depths: List of depths for each encoder block,
            i.e. corresponds to the number of residual blocks at each level.
        encoder_channels: List of channels for each encoder block.
        encoder_num_neighbors: List of maximum number of neighbors for each encoder block.
        fp_channels: List of channels for each feature propagation block.
        grid_sizes: List of grid sizes for each downsampling block.
        radii: Search radius for each downsampling block.
        kernel_size: Size of the kernel for each KPConv block.
        kp_radius: List of kernel radius for KPConv blocks, at each level.
        kp_sigma: List of kernel extent for KPConv blocks, at each level.
        kp_influence: Influence function to use for KPConv blocks. Options are "constant", "linear", "gaussian".
        fixed_position: Whether to fix the position of the kernel points in KPConv blocks. Options are "none", "center", "vertical".
        aggregation_mode: Aggregation mode to use for the KPConv blocks. Options are "sum", "mean", "max".
        deformable: Whether to use a deformable kernel for the KPConv blocks.
        modulated: Whether to use a modulated kernel in KPConv operation.
        norm: Normalization to use for the KPConv blocks.
        act: Activation function to use for the KPConv blocks.
        bias: Whether to use a bias for the KPConv blocks.
        dropout: Dropout rate before the classification head.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        *,
        spatial_dim: int = 3,
        stem_channels: Optional[int] = None,
        stem_type: Literal["linear", "kpconv"] = "kpconv",
        encoder_depths: Sequence[int],
        encoder_channels: Sequence[int],
        encoder_num_neighbors: Sequence[int],
        fp_channels: Sequence[Sequence[int]],
        grid_sizes: Sequence[float],
        radii: Sequence[float],
        kernel_size: int,
        kp_radius: Union[float, Sequence[float]],
        kp_sigma: Union[float, Sequence[float]],
        kp_influence: str = "linear",
        fixed_position: Literal["none", "center", "vertical"] = "center",
        aggregation_mode: str = "sum",
        deformable: Union[bool, Sequence] = False,
        modulated: Union[bool, Sequence] = False,
        act: Union[str, Callable, None] = "leaky_relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: bool = False,
        dropout: float = 0.0,
        head_channels: Optional[Sequence[int]] = None,
    ):
        super().__init__(in_channels, num_classes)
        self.spatial_dim = spatial_dim
        self.stem_channels = stem_channels
        self.stem_type = stem_type
        self.encoder_depths = encoder_depths
        self.encoder_channels = encoder_channels
        self.encoder_num_neighbors = encoder_num_neighbors
        self.fp_channels = fp_channels
        self.head_channels = list(head_channels) if head_channels else []
        self.grid_sizes = grid_sizes
        self.radii = radii
        self.kernel_size = kernel_size
        self.kp_radius = ensure_tuple_size(kp_radius, size=len(encoder_depths))
        self.kp_sigma = ensure_tuple_size(kp_sigma, size=len(encoder_depths))
        self.kp_influence = kp_influence
        self.fixed_position = fixed_position
        self.aggregation_mode = aggregation_mode
        self.deformable = deformable
        self.modulated = modulated
        self.act = act
        self.act_kwargs = act_kwargs
        self.act_first = act_first
        self.norm = norm
        self.norm_kwargs = norm_kwargs
        self.bias = bias
        self.dropout = dropout

        self.stem = self.configure_stem()
        self.encoder = self.configure_encoder()
        self.decoder = self.configure_decoder()
        self.head = self.configure_head()

    def configure_stem(self) -> Optional[nn.Module]:
        """Build the stem lifting the input features to `stem_channels`, or `None` when `stem_channels` is unset."""
        if self.stem_channels is None:
            return None
        if self.stem_type == "kpconv":
            return KPConvBlock(
                spatial_dim=self.spatial_dim,
                in_channels=self.in_channels,
                out_channels=self.stem_channels,
                kernel_size=self.kernel_size,
                kp_radius=self.kp_radius[0],
                kp_sigma=self.kp_sigma[0],
                kp_influence=self.kp_influence,
                fixed_position=self.fixed_position,
                aggregation_mode=self.aggregation_mode,
                act=self.act,
                act_kwargs=self.act_kwargs,
                norm=self.norm,
                norm_kwargs=self.norm_kwargs,
                bias=self.bias,
            )
        act = create_act(self.act, **(self.act_kwargs or {})) or nn.Identity()
        norm = create_norm(self.norm, self.stem_channels, **(self.norm_kwargs or {})) or nn.Identity()
        return nn.Sequential(nn.Linear(self.in_channels, self.stem_channels), norm, act)

    def configure_encoder(self) -> KPFCNNEncoder:
        """Build the `KPFCNNEncoder` backbone."""
        return KPFCNNEncoder(
            in_channels=self.in_channels if self.stem_channels is None else self.stem_channels,
            depths=self.encoder_depths,
            channels=self.encoder_channels,
            grid_sizes=self.grid_sizes,
            radii=self.radii,
            max_num_neighbors=self.encoder_num_neighbors,
            spatial_dim=self.spatial_dim,
            kernel_size=self.kernel_size,
            kp_radius=self.kp_radius,
            kp_sigma=self.kp_sigma,
            kp_influence=self.kp_influence,
            fixed_position=self.fixed_position,
            aggregation_mode=self.aggregation_mode,
            deformable=self.deformable,
            modulated=self.modulated,
            act=self.act,
            act_kwargs=self.act_kwargs,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
            bias=self.bias,
        )

    def configure_decoder(self) -> PointNet2Decoder:
        """Build the `PointNet2Decoder` upsampling the coarsest features back through the encoder skips."""
        stem_channels = self.in_channels if self.stem_channels is None else self.stem_channels
        all_skip_channels = [stem_channels, *self.encoder_channels[:-1]]
        skip_channels = all_skip_channels[-len(self.fp_channels) :][::-1]
        return PointNet2Decoder(
            in_channels=self.encoder_channels[-1],
            skip_channels=skip_channels,
            fp_channels=self.fp_channels,
            bias=self.bias,
            act=self.act,
            act_kwargs=self.act_kwargs,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
            k=1,
        )

    @property
    def num_features(self) -> int:
        """Feature dimension $C$ of the decoder output."""
        return self.fp_channels[-1][-1]

    def configure_head(self) -> nn.Module:
        if self.num_classes == 0:
            return nn.Identity()
        if not self.head_channels:
            return nn.Linear(self.num_features, self.num_classes)
        head_act = create_act(self.act, **(self.act_kwargs or {})) or nn.Identity()
        layers: List[nn.Module] = []
        ch_in = self.num_features
        for ch in self.head_channels:
            layers.append(nn.Sequential(nn.Linear(ch_in, ch, bias=True), head_act))
            ch_in = ch
        layers.append(nn.Linear(ch_in, self.num_classes))
        return nn.Sequential(*layers)

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
    ) -> Tuple[Tensor, Tensor, Tensor, List[PooledFeaturesDict]]: ...

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
        if x is None:
            if self.in_channels != pos.size(1):
                raise ValueError(
                    f"Got `x=None` but the model expects in_channels={self.in_channels} features while `pos` has "
                    f"{pos.size(1)} channels; pass `x` of shape (N, {self.in_channels})."
                )
            x = pos

        if self.stem is not None:
            if self.stem_type == "kpconv":
                edge_index = radius_graph(
                    pos,
                    r=self.radii[0],
                    batch=batch,
                    max_num_neighbors=self.encoder_num_neighbors[0],
                    loop=True,
                )
                x = self.stem(x, pos, edge_index)
            else:
                x = self.stem(x)

        if not return_intermediates:
            return self.encoder(x, pos, batch)

        skip: PooledFeaturesDict = {"x": x, "pos": pos, "batch": batch}
        x, pos, batch, intermediates = self.encoder(x, pos, batch, return_intermediates=True)
        return x, pos, batch, [skip, *intermediates]

    def forward_decoder(
        self,
        x: Tensor,
        pos: Tensor,
        batch: Tensor,
        intermediates: List[PooledFeaturesDict],
    ) -> Tensor:
        x, _, _ = self.decoder(x, pos, batch, intermediates)
        return x

    def forward_head(self, x: Tensor, pre_logits: bool = False) -> Tensor:
        if self.dropout:
            x = F.dropout(x, p=float(self.dropout), training=self.training)
        return x if pre_logits else self.head(x)

    def forward(self, x: OptTensor, pos: Tensor, batch: Tensor) -> Tensor:
        x, pos, batch, intermediates = self.forward_features(x, pos, batch, return_intermediates=True)
        x = self.forward_decoder(x, pos, batch, intermediates)
        return self.forward_head(x)


@register_model(
    "kpfcnn.modelnet40",
    task="classification",
    hparams=dict(
        in_channels=6,
        num_classes=40,
        stem_channels=32,
        stem_type="kpconv",
        encoder_depths=[1, 3, 3, 3],
        encoder_channels=[64, 128, 256, 512],
        encoder_num_neighbors=[20, 35, 40, 40],
        grid_sizes=[0.08, 0.16, 0.32],
        radii=[0.1, 0.2, 0.4, 0.8],
        kernel_size=15,
        kp_radius=[0.1, 0.2, 0.4, 0.8],
        kp_sigma=[0.05, 0.1, 0.2, 0.4],
        act="leaky_relu",
        norm="batch_norm",
        norm_kwargs={"momentum": 0.05},
    ),
)
def kpfcnn_modelnet40_clf(**hparams: Any) -> KPFCNNClassification:
    return KPFCNNClassification(**hparams)


_BASE_S3DIS_TRANSFORMS = T.Compose(
    [
        # Original implementation of KPConv uses custom C++ code for grid subsampling,
        # which behaves differently from the PyG implementation, but is close enough.
        # The main difference is that labels are reduced using the most frequent value per voxel.
        # NOTE: tensors are automatically converted to float before reduction (if other than "first")
        T.CopyItems(
            keys=[DataKeys.POS, DataKeys.SEGMENT],
            dst_keys=[DataKeys.ORIGIN_POS, DataKeys.ORIGIN_SEGMENT],
            allow_missing_keys=True,
        ),
        T.Voxelize(
            pos_key=DataKeys.POS,
            pos_reduce="mean",
            keys=[DataKeys.COLOR, DataKeys.SEGMENT],
            reduce=["mean", "first"],
            size=0.03,
            method="grid",
            dst_inverse_key=DataKeys.INVERSE,
        ),
        T.Scale(keys=DataKeys.COLOR, scale=1.0 / 255),
        T.AxisMinOffset(keys=DataKeys.POS, dst_keys="height", axis=2),
        T.OnesLike(keys="height", dst_keys="ones"),
        T.Cat(keys=["ones", DataKeys.COLOR, "height"], dst_key=DataKeys.X),
        T.RenameItems(keys=[DataKeys.SEGMENT], dst_keys=[DataKeys.LABEL]),
        T.KeepItems(
            keys=[
                DataKeys.X,
                DataKeys.POS,
                DataKeys.LABEL,
                DataKeys.ORIGIN_POS,
                DataKeys.ORIGIN_SEGMENT,
                DataKeys.INVERSE,
            ]
        ),
    ]
)


@register_model(
    "kpfcnn-base-sm.s3dis-area5.hugues-thomas",
    task="semantic-segmentation",
    transform=_BASE_S3DIS_TRANSFORMS,
    weights=WeightsDict(
        url="hf://torch-pointcloud/kpfcnn-base-sm.s3dis-area5.hugues-thomas/resolve/2162b440c28ddcf97e130a3cdfbf6cf740db5388/model.safetensors",
        dataset="s3dis-area5",
        metrics={"mIoU": 65.27, "OA": 88.93},
        classes=S3DIS_CLASSES,
        author="hugues-thomas",
        license="MIT",
    ),
    hparams=dict(
        in_channels=5,
        num_classes=13,
        stem_channels=64,
        stem_type="kpconv",
        encoder_depths=[1, 2, 2, 3, 3],
        encoder_channels=[128, 256, 512, 1024, 2048],
        encoder_num_neighbors=[128, 128, 128, 128, 128],
        fp_channels=[[1024], [512], [256], [128]],
        head_channels=[128],
        grid_sizes=[0.06, 0.12, 0.24, 0.48],
        radii=[0.075, 0.15, 0.3, 0.6, 1.2],
        kernel_size=15,
        kp_radius=[0.075, 0.15, 0.3, 0.6, 1.2],
        kp_sigma=[0.036, 0.072, 0.144, 0.288, 0.576],
        kp_influence="linear",
        fixed_position="center",
        aggregation_mode="sum",
        deformable=False,
        modulated=False,
        bias=False,
        act="leaky_relu",
        act_kwargs={"negative_slope": 0.1},
        norm="batch_norm",
        norm_kwargs={"momentum": 0.02},
    ),
)
def kpfcnn_base_sm_seg(**hparams: Any) -> KPFCNNSegmentation:
    return KPFCNNSegmentation(**hparams)


@register_model(
    "kpfcnn-base.s3dis-area5.hugues-thomas",
    task="semantic-segmentation",
    transform=_BASE_S3DIS_TRANSFORMS,
    weights=WeightsDict(
        url="hf://torch-pointcloud/kpfcnn-base.s3dis-area5.hugues-thomas/resolve/7b241427afada125eb8e0dcedd30d8352e2e7d7a/model.safetensors",
        dataset="s3dis-area5",
        metrics={"mIoU": 66.60, "OA": 89.63},
        classes=S3DIS_CLASSES,
        author="hugues-thomas",
        license="MIT",
    ),
    hparams=dict(
        in_channels=5,
        num_classes=13,
        stem_channels=64,
        stem_type="kpconv",
        encoder_depths=[1, 3, 3, 3, 3],
        encoder_channels=[128, 256, 512, 1024, 2048],
        encoder_num_neighbors=[128, 128, 128, 128, 128],
        fp_channels=[[1024], [512], [256], [128]],
        head_channels=[128],
        grid_sizes=[0.06, 0.12, 0.24, 0.48],
        radii=[0.075, 0.15, 0.3, 0.6, 1.2],
        kernel_size=15,
        kp_radius=[0.075, 0.15, 0.3, 0.6, 1.2],
        kp_sigma=[0.036, 0.072, 0.144, 0.288, 0.576],
        kp_influence="linear",
        fixed_position="center",
        aggregation_mode="sum",
        deformable=False,
        modulated=False,
        bias=False,
        act="leaky_relu",
        act_kwargs={"negative_slope": 0.1},
        norm="batch_norm",
        norm_kwargs={"momentum": 0.02},
    ),
)
def kpfcnn_base_seg(**hparams: Any) -> KPFCNNSegmentation:
    return KPFCNNSegmentation(**hparams)


@register_model(
    "kpfcnn-base-deform.s3dis-area5.hugues-thomas",
    task="semantic-segmentation",
    transform=_BASE_S3DIS_TRANSFORMS,
    weights=WeightsDict(
        url="hf://torch-pointcloud/kpfcnn-base-deform.s3dis-area5.hugues-thomas/resolve/bbeeb89dc1b1f5bc72ed713d47e90a84f263aa59/model.safetensors",
        dataset="s3dis-area5",
        metrics={"mIoU": 67.05, "OA": 89.93},
        classes=S3DIS_CLASSES,
        author="hugues-thomas",
        license="MIT",
    ),
    hparams=dict(
        in_channels=5,
        num_classes=13,
        stem_channels=64,
        stem_type="kpconv",
        encoder_depths=[1, 3, 3, 3, 3],
        encoder_channels=[128, 256, 512, 1024, 2048],
        encoder_num_neighbors=[128, 128, 1024, 1024, 1024],
        fp_channels=[[1024], [512], [256], [128]],
        head_channels=[128],
        grid_sizes=[0.06, 0.12, 0.24, 0.48],
        radii=[0.075, 0.15, 0.72, 1.44, 2.88],
        kernel_size=15,
        kp_radius=[0.075, 0.15, 0.3, 0.6, 1.2],
        kp_sigma=[0.036, 0.072, 0.144, 0.288, 0.576],
        kp_influence="linear",
        fixed_position="center",
        aggregation_mode="sum",
        deformable=[False, False, [False, True, True], True, True],
        modulated=False,
        bias=False,
        act="leaky_relu",
        act_kwargs={"negative_slope": 0.1},
        norm="batch_norm",
        norm_kwargs={"momentum": 0.02},
    ),
)
def kpfcnn_base_deform_seg(**hparams: Any) -> KPFCNNSegmentation:
    return KPFCNNSegmentation(**hparams)


@register_model(
    "kpfcnn-base-sm-deform.s3dis-area5.hugues-thomas",
    task="semantic-segmentation",
    transform=_BASE_S3DIS_TRANSFORMS,
    weights=WeightsDict(
        url="hf://torch-pointcloud/kpfcnn-base-sm-deform.s3dis-area5.hugues-thomas/resolve/ebb67bd500167d5ec7249a63a68eae2b76b72f53/model.safetensors",
        dataset="s3dis-area5",
        metrics={"mIoU": 66.01, "OA": 89.53},
        classes=S3DIS_CLASSES,
        author="hugues-thomas",
        license="MIT",
    ),
    hparams=dict(
        in_channels=5,
        num_classes=13,
        stem_channels=64,
        stem_type="kpconv",
        encoder_depths=[1, 3, 3, 3, 3],
        encoder_channels=[128, 256, 512, 1024, 2048],
        encoder_num_neighbors=[128, 128, 128, 1024, 1024],
        fp_channels=[[1024], [512], [256], [128]],
        head_channels=[128],
        grid_sizes=[0.06, 0.12, 0.24, 0.48],
        radii=[0.075, 0.15, 0.3, 1.2, 2.4],
        kernel_size=15,
        kp_radius=[0.075, 0.15, 0.3, 0.6, 1.2],
        kp_sigma=[0.036, 0.072, 0.144, 0.288, 0.576],
        kp_influence="linear",
        fixed_position="center",
        aggregation_mode="sum",
        deformable=[False, False, False, [False, True, True], True],
        modulated=False,
        bias=False,
        act="leaky_relu",
        act_kwargs={"negative_slope": 0.1},
        norm="batch_norm",
        norm_kwargs={"momentum": 0.02},
    ),
)
def kpfcnn_base_sm_deform_seg(**hparams: Any) -> KPFCNNSegmentation:
    return KPFCNNSegmentation(**hparams)
