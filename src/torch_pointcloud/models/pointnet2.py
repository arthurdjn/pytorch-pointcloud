from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union, overload

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import MLP

from torch_pointcloud.layers import PoolLike, create_cls_head, create_pool
from torch_pointcloud.utils.conversion import ensure_list, ensure_tuple, ensure_tuple_size, is_iterable
from torch_pointcloud.utils.imports import optional_import
from torch_pointcloud.utils.ops import knn_interpolate
from torch_pointcloud.utils.types import OptTensor

if TYPE_CHECKING:
    from torch_cluster import fps, radius


fps, _ = optional_import("torch_cluster", "fps")
radius, _ = optional_import("torch_cluster", "radius")


# TODO: add a `spatial_dim` argument
class SAModule(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: Sequence[Union[int, Sequence[int]]],
        ratio: float,
        radii: Union[float, Sequence[float]],
        num_neighbors: Union[int, Sequence[int]],
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: bool = False,
        use_coords: bool = True,
        normalize_coords: bool = True,
        pool: PoolLike = "max",
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.channels = ensure_list(channels, recursive=True)
        self.ratio = ratio
        self.use_coords = use_coords
        self.normalize_coords = normalize_coords

        # Wrap parameters in list of lists to be compatible with Multi-Scale Grouping (MSG) mode
        self.channels = [self.channels] if not isinstance(self.channels[0], list) else self.channels
        sizes = [len(channels) for channels in self.channels]

        extra_msg = f"The parameter `{{param}}` must be a sequence matching the number of scales {len(sizes)}."
        self.radii = ensure_tuple_size(radii, size=len(sizes), extra_msg=extra_msg.format(param="radii"))
        self.num_neighbors = ensure_tuple_size(
            num_neighbors,
            size=len(sizes),
            extra_msg=extra_msg.format(param="num_neighbors"),
        )

        in_channels = in_channels + 3 if use_coords else in_channels
        self.mlps = nn.ModuleList()
        for i in range(len(self.channels)):
            mlp = MLP(
                [in_channels, *self.channels[i]],
                act=act,
                act_kwargs=act_kwargs,
                act_first=act_first,
                norm=norm,
                norm_kwargs=norm_kwargs,
                bias=bias,
                plain_last=False,
            )
            self.mlps.append(mlp)

        self.pool = create_pool(pool)

    def forward(
        self,
        features: Tensor,
        coords: Tensor,
        batch: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        idx = fps(coords, batch, ratio=self.ratio)
        new_coords = coords[idx]
        new_batch = batch[idx]
        msg_features = []

        for r, k, mlp in zip(self.radii, self.num_neighbors, self.mlps):
            row, col = radius(coords, new_coords, r, batch, new_batch, max_num_neighbors=k)
            # row: Tensor of shape (N,) containing neighbor indices in the the new point cloud
            # col: Tensor of shape (N,) containing neighbor indices in the original point cloud
            rel_coords = coords[col] - new_coords[row]
            if self.normalize_coords:
                rel_coords = rel_coords / r

            new_features = features[col]
            if self.use_coords:
                new_features = torch.cat([new_features, rel_coords], dim=1)

            new_features = mlp(new_features)
            new_features = self.pool(new_features, row)
            msg_features.append(new_features)

        return torch.cat(msg_features, dim=1), new_coords, new_batch


class GlobalSAModule(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: Sequence[int],
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: bool = False,
        pool: PoolLike = "max",
    ) -> None:
        super().__init__()
        channels = list(ensure_tuple(channels))
        self.mlp = MLP(
            [in_channels, *channels],
            act=act,
            act_kwargs=act_kwargs,
            act_first=act_first,
            norm=norm,
            norm_kwargs=norm_kwargs,
            bias=bias,
            plain_last=False,
        )
        self.pool = create_pool(pool)

    def forward(self, features: Tensor, coords: Tensor, batch: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        features = self.mlp(features)
        features = self.pool(features, batch)
        coords = coords.new_zeros((features.size(0), 3))
        batch = torch.arange(features.size(0), device=batch.device)
        return features, coords, batch


class FPModule(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: Sequence[int],
        k: int,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.k = k
        self.mlp = MLP(
            [in_channels, *channels],
            act=act,
            act_kwargs=act_kwargs,
            act_first=act_first,
            norm=norm,
            norm_kwargs=norm_kwargs,
            bias=bias,
            plain_last=False,
        )

    def forward(
        self,
        features: Tensor,
        coords: Tensor,
        batch: Tensor,
        features_skip: OptTensor,
        coords_skip: Tensor,
        batch_skip: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        features = knn_interpolate(features, coords, coords_skip, batch, batch_skip, k=self.k)
        if features_skip is not None:
            features = torch.cat([features, features_skip], dim=1)

        features = self.mlp(features)
        return features, coords_skip, batch_skip


def create_sa_blocks(
    in_channels: int,
    channels: Sequence[Sequence[Sequence[int]]],
    ratios: Sequence[float],
    radii: Sequence[Union[float, Sequence[float]]],
    num_neighbors: Sequence[Union[int, Sequence[int]]],
    act: Union[str, Callable, None] = "relu",
    act_kwargs: Optional[Dict[str, Any]] = None,
    act_first: bool = False,
    norm: Union[str, Callable, None] = "batch_norm",
    norm_kwargs: Optional[Dict[str, Any]] = None,
    bias: bool = False,
    use_coords: bool = True,
    pool: PoolLike = "max",
) -> nn.ModuleList:
    num_blocks = len(channels)
    extra_msg = f"The parameter `{{param}}` must be a sequence matching the number of blocks {num_blocks}."
    ratios = ensure_tuple_size(ratios, size=num_blocks, extra_msg=extra_msg.format(param="ratios"))
    radii = ensure_tuple_size(radii, size=num_blocks, extra_msg=extra_msg.format(param="radii"))
    num_neighbors = ensure_tuple_size(num_neighbors, size=num_blocks, extra_msg=extra_msg.format(param="k"))

    blocks = nn.ModuleList()
    for i in range(num_blocks):
        block = SAModule(
            in_channels=in_channels,
            channels=channels[i],
            ratio=ratios[i],
            radii=radii[i],
            num_neighbors=num_neighbors[i],
            act=act,
            act_kwargs=act_kwargs,
            act_first=act_first,
            norm=norm,
            norm_kwargs=norm_kwargs,
            bias=bias,
            use_coords=use_coords,
            pool=pool,
        )
        blocks.append(block)
        in_channels = sum([c[-1] for c in channels[i]])

    return blocks


def create_fp_blocks(
    in_channels: int,
    skip_channels: Sequence[int],
    fp_channels: Sequence[Sequence[int]],
    act: Union[str, Callable, None] = "relu",
    act_kwargs: Optional[Dict[str, Any]] = None,
    act_first: bool = False,
    norm: Union[str, Callable, None] = "batch_norm",
    norm_kwargs: Optional[Dict[str, Any]] = None,
    bias: bool = False,
    k: int = 3,
) -> nn.ModuleList:
    if len(skip_channels) != len(fp_channels):
        raise ValueError(
            f"The number of skip channels ({len(skip_channels)}) must match "
            f"the number of feature propagation channels ({len(fp_channels)})."
        )

    blocks = nn.ModuleList()
    num_blocks = len(fp_channels)

    for i in range(num_blocks):
        in_channels = in_channels if i == 0 else fp_channels[i - 1][-1]
        block = FPModule(
            in_channels=in_channels + skip_channels[i],
            channels=fp_channels[i],
            k=1 if i == 0 else k,
            act=act,
            act_kwargs=act_kwargs,
            act_first=act_first,
            norm=norm,
            norm_kwargs=norm_kwargs,
            bias=bias,
        )
        blocks.append(block)

    return blocks


def ensure_msg_list(items: Sequence[Any], extra_msg: str = "") -> List[List[List[Any]]]:
    """Utility function to ensure that items are converted in nested lists compatible
    with Multi-Scale Grouping (MSG) mode.
    This function will convert a list of list into a list of list of list.

    Example:
        Let's say we have designed a network where the first two SA blocks are
        not using MSG mode, but the last SA block is using MSG mode.

        Calling `ensure_msg_list` will make sure the provided channels are compliant
        with the MSG mode.

        >>> sa_channels = [[32, 64], [128, 256], [[256, 512, 512], [256, 512, 1024]]]
        >>> ensure_msg_list(sa_channels)
        [[[32, 64]], [[128, 256]], [[256, 512, 512], [256, 512, 1024]]]
    """
    items = ensure_list(items, recursive=True)

    result = []
    if not is_iterable(items):
        raise ValueError(f"Expected a sequence, got {type(items).__name__}. {extra_msg}")

    for i, item in enumerate(items):
        if not is_iterable(item):
            raise ValueError(f"Expected a sequence, got {type(item).__name__} at index {i} from {items}. {extra_msg}")

        # Check if the item is already a list of lists
        if all(is_iterable(subitem) for subitem in item):
            result.append(item)
        elif all(not is_iterable(subitem) for subitem in item):
            result.append([item])
        else:
            raise ValueError(
                "Expected either all items to be iterable or non-iterable, "
                f"got a mix of both at index {i} from {items}. {extra_msg}"
            )

    return result


class PointNet2Classification(nn.Module):
    """PointNet++ classification model from the paper
    [PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space](https://arxiv.org/abs/1706.02413)
    by Charles R. Qi, Li Yi, Hao Su, Leonidas J. Guibas.

    This network is a hierarchical point cloud classification model. It processes raw point clouds through
    multiple Set Abstraction (SA) blocks that progressively downsample the points while learning local features
    using radius-based grouping. Each SA block can optionally use Multi-Scale Grouping (MSG) to capture features
    at different scales. The network starts with an optional linear stem layer, followed by SA blocks, and ends
    with aggregation layers and a classification head. The architecture supports flexible configuration of
    channels, sampling ratios, search radiuses and neighborhood sizes at each level.

    Args:
        in_channels: Number of input channels (features per point).
        num_classes: Number of output classes.
        stem_channels: Optional number of channels for initial linear projection.
        sa_channels: List of channel configurations for Set Abstraction (SA) blocks.
            Each element defines the MLP channels for one SA block.
            For Multi-Scale Grouping (MSG), provide nested lists of channels.
        aggr_channels: Channel sizes for the final MLP after global pooling.
        ratios: Sampling ratios for each SA block (between 0 and 1).
        radii: Search radiuses for each SA block's neighborhood.
            For MSG, provide a list of radii per block.
        num_neighbors: Max number of neighbors for each SA block.
            For MSG, provide a list of neighbor counts per block.
        act: Activation function type or callable.
        act_kwargs: Additional keyword arguments for the activation function.
        act_first: If ``True``, activation is applied before normalization.
        norm: Normalization layer type or callable.
        norm_kwargs: Additional keyword arguments for the normalization layer.
        bias: Whether to use bias in linear layers.
        use_coords: Whether to use point coordinates as features.
        pool: Pooling operation for SA blocks.
        dropout: Dropout rate for classification head.
        global_pool: Global pooling operation.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        *,
        stem_channels: Optional[int] = None,
        sa_channels: Sequence[Sequence[Union[int, Sequence[int]]]],
        aggr_channels: Optional[Union[int, Sequence[int]]] = None,
        ratios: Sequence[float],
        radii: Sequence[Union[float, Sequence[float]]],
        num_neighbors: Sequence[Union[int, Sequence[int]]],
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: bool = False,
        use_coords: bool = True,
        pool: PoolLike = "max",
        dropout: float = 0.0,
        global_pool: PoolLike = "max",
    ) -> None:
        super().__init__()
        sa_channels = ensure_msg_list(
            sa_channels,
            extra_msg="The parameter `sa_channels` must be a sequence compliant with the Multi-Scale Grouping (MSG) mode.",
        )

        self.in_channels = in_channels
        self.num_classes = num_classes

        self.stem: Optional[nn.Module] = None
        if stem_channels is not None:
            self.stem = nn.Linear(in_channels, stem_channels)
            in_channels = stem_channels

        self.sa_blocks = create_sa_blocks(
            in_channels=in_channels,
            channels=sa_channels,
            ratios=ratios,
            radii=radii,
            num_neighbors=num_neighbors,
            act=act,
            act_kwargs=act_kwargs,
            act_first=act_first,
            norm=norm,
            norm_kwargs=norm_kwargs,
            bias=bias,
            use_coords=use_coords,
            pool=pool,
        )

        in_channels = sum([c[-1] for c in sa_channels[-1]])
        aggr_channels = ensure_tuple(aggr_channels) if aggr_channels else None
        self.aggr = (
            MLP(
                [in_channels, *aggr_channels],
                act=act,
                act_kwargs=act_kwargs,
                act_first=act_first,
                norm=norm,
                norm_kwargs=norm_kwargs,
                plain_last=False,
            )
            if aggr_channels
            else None
        )

        self.global_pool = create_pool(global_pool)
        self.dropout = dropout
        self.embedding_dim = aggr_channels[-1] if aggr_channels else in_channels
        self.head = create_cls_head(self.embedding_dim, num_classes)

    def reset_classifier(self, num_classes: int, global_pool: PoolLike = "max", **kwargs: Any) -> None:
        self.num_classes = num_classes
        self.global_pool = create_pool(global_pool)
        self.head = create_cls_head(num_features=self.embedding_dim, num_classes=self.num_classes, **kwargs)

    @overload
    def forward_features(
        self,
        features: OptTensor,
        coords: Tensor,
        batch: Tensor,
        return_intermediates: Literal[True],
    ) -> Tuple[Tensor, Tensor, Tensor, List[Dict[str, Tensor]]]: ...

    @overload
    def forward_features(
        self,
        features: OptTensor,
        coords: Tensor,
        batch: Tensor,
        return_intermediates: Literal[False] = False,
    ) -> Tuple[Tensor, Tensor, Tensor]: ...

    def forward_features(
        self,
        features: OptTensor,
        coords: Tensor,
        batch: Tensor,
        return_intermediates: bool = False,
    ) -> Any:
        features = features if features is not None else coords
        if self.stem is not None:
            features = self.stem(features)

        # NOTE: We only store the intermediate results if specified with `return_intermediates=True`
        intermediates = [{"features": features, "pos": coords, "batch": batch}] if return_intermediates else []
        for i, block in enumerate(self.sa_blocks):
            features, coords, batch = block(features, coords, batch)
            if return_intermediates and i < len(self.sa_blocks) - 1:
                # NOTE: Do not store the last result, as it will be the returned output.
                # TODO: We could move this up, before the forward block call,
                # TODO: to avoid having the condition i < len(self.sa_blocks) - 1 etc.
                intermediates.append({"features": features, "pos": coords, "batch": batch})

        if self.aggr is not None:
            features = self.aggr(features)

        if return_intermediates:
            return features, coords, batch, intermediates
        return features, coords, batch

    def forward_head(self, x: Tensor, batch: Tensor, pre_logits: bool = False) -> Tensor:
        x = self.global_pool(x, batch)
        if self.dropout:
            x = F.dropout(x, p=float(self.dropout), training=self.training)
        return x if pre_logits else self.head(x)

    def forward(self, features: OptTensor, coords: Tensor, batch: Tensor) -> Tensor:
        features, _, batch = self.forward_features(features, coords, batch)
        return self.forward_head(features, batch)


# TODO: Update the docstring (remove classification part)
class PointNet2Segmentation(nn.Module):
    """PointNet++ segmentation model from the paper
    [PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space](https://arxiv.org/abs/1706.02413)
    by Charles R. Qi, Li Yi, Hao Su, Leonidas J. Guibas.

    This network is a hierarchical point cloud segmentation model. It processes raw point clouds through
    multiple Set Abstraction (SA) blocks that progressively downsample the points while learning local features
    using radius-based grouping. Each SA block can optionally use Multi-Scale Grouping (MSG) to capture features
    at different scales. The network starts with an optional linear stem layer, followed by SA blocks and optionally
    a GlobalSA block. The network ends with a Feature Propagation (FP) block for each SA block and finally a
    classification head. The architecture supports flexible configuration of channels, sampling ratios, search
    radiuses and neighborhood sizes at each level.

    Args:
        in_channels: Number of input channels (features per point).
        num_classes: Number of output classes.
        stem_channels: Optional number of channels for initial linear projection.
        sa_channels: List of channel configurations for Set Abstraction (SA) blocks.
            Each element defines the MLP channels for one SA block.
            For Multi-Scale Grouping (MSG), provide nested lists of channels.
        aggr_channels: Channel sizes for the final MLP after global pooling.
        fp_channels: List of channel configurations for Feature Propagation (FP) blocks.
            Each element defines the MLP channels for one FP block.
        ratios: Sampling ratios for each SA block (between 0 and 1).
        radii: Search radiuses for each SA block's neighborhood.
            For MSG, provide a list of radii per block.
        num_neighbors: Max number of neighbors for each SA block.
            For MSG, provide a list of neighbor counts per block.
        act: Activation function type or callable.
        act_kwargs: Additional keyword arguments for the activation function.
        act_first: If ``True``, activation is applied before normalization.
        norm: Normalization layer type or callable.
        norm_kwargs: Additional keyword arguments for the normalization layer.
        bias: Whether to use bias in linear layers.
        use_coords: Whether to use point coordinates as features.
        pool: Pooling operation for SA blocks.
        dropout: Dropout rate for classification head.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        *,
        stem_channels: Optional[int] = None,
        sa_channels: Sequence[Sequence[Union[int, Sequence[int]]]],
        aggr_channels: Optional[Union[int, Sequence[int]]] = None,
        fp_channels: Sequence[Sequence[int]],
        ratios: Sequence[float],
        radii: Sequence[Union[float, Sequence[float]]],
        num_neighbors: Sequence[Union[int, Sequence[int]]],
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = False,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: bool = False,
        use_coords: bool = True,
        pool: PoolLike = "max",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        sa_channels = ensure_msg_list(
            sa_channels,
            extra_msg="The parameter `sa_channels` must be a sequence compliant "
            f"with Multi-Scale Grouping (MSG) mode, but got {type(sa_channels)}.",
        )

        self.in_channels = in_channels
        self.num_classes = num_classes

        self.stem = nn.Linear(in_channels, stem_channels) if stem_channels else None
        in_channels = stem_channels if stem_channels else in_channels
        skip_channels = [in_channels]

        self.sa_blocks = create_sa_blocks(
            in_channels=in_channels,
            channels=sa_channels,
            ratios=ratios,
            radii=radii,
            num_neighbors=num_neighbors,
            act=act,
            act_kwargs=act_kwargs,
            act_first=act_first,
            norm=norm,
            norm_kwargs=norm_kwargs,
            bias=bias,
            use_coords=use_coords,
            pool=pool,
        )

        # Store the output channels of each SA (MSG) block for ease of use
        sa_out_channels = []
        for i in range(len(sa_channels)):
            sa_out_channels.append(sum([c[-1] for c in sa_channels[i]]))

        # The skip channels are the output of each SA (MSG) block except the last one
        skip_channels.extend(sa_out_channels[:-1])
        in_channels = sa_out_channels[-1]

        aggr_channels = ensure_tuple(aggr_channels) if aggr_channels else None
        self.aggr = (
            MLP(
                [in_channels, *aggr_channels],
                act=act,
                act_kwargs=act_kwargs,
                act_first=act_first,
                norm=norm,
                norm_kwargs=norm_kwargs,
                plain_last=False,
            )
            if aggr_channels
            else None
        )

        self.fp_blocks = create_fp_blocks(
            in_channels=aggr_channels[-1] if aggr_channels is not None else in_channels,
            skip_channels=skip_channels[::-1],
            fp_channels=fp_channels,
            bias=bias,
            act=act,
            act_kwargs=act_kwargs,
            act_first=act_first,
            norm=norm,
            norm_kwargs=norm_kwargs,
        )

        self.dropout = dropout
        self.embedding_dim = fp_channels[-1][-1]
        self.head = create_cls_head(self.embedding_dim, num_classes)

    def reset_classifier(self, num_classes: int, **kwargs: Any) -> None:
        self.num_classes = num_classes
        self.head = create_cls_head(num_features=self.embedding_dim, num_classes=self.num_classes, **kwargs)

    @overload
    def forward_features(
        self,
        features: OptTensor,
        coords: Tensor,
        batch: Tensor,
        return_intermediates: Literal[True],
    ) -> Tuple[Tensor, Tensor, Tensor, List[Dict[str, Tensor]]]: ...

    @overload
    def forward_features(
        self,
        features: OptTensor,
        coords: Tensor,
        batch: Tensor,
        return_intermediates: Literal[False] = False,
    ) -> Tuple[Tensor, Tensor, Tensor]: ...

    def forward_features(
        self,
        features: OptTensor,
        coords: Tensor,
        batch: Tensor,
        return_intermediates: bool = False,
    ) -> Any:
        features = features if features is not None else coords
        if self.stem is not None:
            features = self.stem(features)

        # NOTE: We only store the intermediate results if specified with `return_intermediates=True`
        intermediates = [{"features": features, "pos": coords, "batch": batch}] if return_intermediates else []
        for i, block in enumerate(self.sa_blocks):
            features, coords, batch = block(features, coords, batch)
            if return_intermediates and i < len(self.sa_blocks) - 1:
                # NOTE: Do not store the last result, as it will be the returned output.
                intermediates.append({"features": features, "pos": coords, "batch": batch})

        if self.aggr is not None:
            features = self.aggr(features)

        if return_intermediates:
            return features, coords, batch, intermediates
        return features, coords, batch

    def forward_decoder(
        self,
        features: Tensor,
        coords: Tensor,
        batch: Tensor,
        intermediates: List[Dict[str, Tensor]],
    ) -> Tensor:
        for block, intermediate in zip(self.fp_blocks, reversed(intermediates)):
            features_skip = intermediate["features"]
            coords_skip = intermediate["pos"]
            batch_skip = intermediate["batch"]

            features, coords, batch = block(features, coords, batch, features_skip, coords_skip, batch_skip)
        return features

    def forward_head(self, x: Tensor, pre_logits: bool = False) -> Tensor:
        if self.dropout:
            x = F.dropout(x, p=float(self.dropout), training=self.training)
        return x if pre_logits else self.head(x)

    def forward(self, features: OptTensor, coords: Tensor, batch: Tensor) -> Tensor:
        features, coords, batch, intermediates = self.forward_features(
            features, coords, batch, return_intermediates=True
        )
        features = self.forward_decoder(features, coords, batch, intermediates)
        return self.forward_head(features)
