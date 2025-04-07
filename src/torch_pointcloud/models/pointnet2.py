from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Sequence, Tuple, Union, overload

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torch_pointcloud.layers import MLP, ActLike, NormLike, PoolLike, create_cls_head, create_pool
from torch_pointcloud.utils.conversion import ensure_tuple, ensure_tuple_size
from torch_pointcloud.utils.imports import optional_import
from torch_pointcloud.utils.ops import knn_interpolate
from torch_pointcloud.utils.types import OptTensor

if TYPE_CHECKING:
    from torch_cluster import fps, radius


fps, _ = optional_import("torch_cluster", "fps")
radius, _ = optional_import("torch_cluster", "radius")


class SAModule(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: Sequence[Union[int, Sequence[int]]],
        ratio: float,
        radii: Union[float, Sequence[float]],
        num_neighbors: Union[int, Sequence[int]],
        act: ActLike = "relu",
        norm: NormLike = "batch_norm1d",
        bias: bool = False,
        order: str = "lan",
        use_coords: bool = True,
        normalize_coords: bool = True,
        pool: PoolLike = "max",
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.channels = ensure_tuple(channels, recursive=True)
        self.ratio = ratio
        self.order = order
        self.use_coords = use_coords
        self.normalize_coords = normalize_coords

        # Wrap parameters in list of lists to be compatible with Multi-Scale Grouping (MSG) mode
        self.channels = [self.channels] if not isinstance(self.channels[0], tuple) else self.channels
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
                channels=self.channels[i],
                in_channels=in_channels,
                act=act,
                norm=norm,
                bias=bias,
                dropout=None,
                order=order,
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
            row, col = radius(coords, new_coords, r, batch, batch[idx], max_num_neighbors=k)
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
        act: ActLike = "relu",
        norm: NormLike = "batch_norm1d",
        bias: bool = False,
        order: str = "lan",
        pool: PoolLike = "max",
    ) -> None:
        super().__init__()
        channels = list(ensure_tuple(channels))
        self.mlp = MLP(channels, in_channels=in_channels, act=act, norm=norm, bias=bias, dropout=None, order=order)
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
        act: ActLike = "relu",
        norm: NormLike = "batch_norm1d",
        bias: bool = False,
        order: str = "lan",
    ) -> None:
        super().__init__()
        self.k = k
        self.mlp = MLP(channels, in_channels=in_channels, act=act, norm=norm, bias=bias, dropout=None, order=order)

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
    act: ActLike = "relu",
    norm: NormLike = "batch_norm1d",
    bias: bool = False,
    order: str = "land",
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
            norm=norm,
            bias=bias,
            order=order,
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
    act: ActLike = "relu",
    norm: NormLike = "batch_norm1d",
    bias: bool = False,
    order: str = "lan",
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
            k=1 if i == 0 else 3,
            act=act,
            norm=norm,
            bias=bias,
            order=order,
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

    def is_iterable(item: Any) -> bool:
        return isinstance(item, Iterable) and not isinstance(item, (str, bytes))

    result = []
    if not is_iterable(items):
        raise ValueError(f"Expected a sequence, got {type(items).__name__}. {extra_msg}")

    for i, item in enumerate(items):
        if not is_iterable(item):
            raise ValueError(f"Expected a sequence, got {type(item).__name__} at index {i} from {items}. {extra_msg}")

        # Check if the item is already a list of lists
        if any(is_iterable(subitem) for subitem in item):
            result.append(item)
        else:
            result.append([item])

    return result


class PointNet2Classification(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        *,
        stem_channels: Optional[int] = None,
        sa_channels: Sequence[Sequence[Union[int, Sequence[int]]]],
        aggr_channels: Sequence[int],
        ratios: Sequence[float],
        radii: Sequence[Union[float, Sequence[float]]],
        num_neighbors: Sequence[Union[int, Sequence[int]]],
        act: ActLike = "relu",
        norm: NormLike = "batch_norm1d",
        bias: bool = False,
        order: str = "lan",
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
            norm=norm,
            bias=bias,
            order=order,
            use_coords=use_coords,
            pool=pool,
        )

        in_channels = sum([c[-1] for c in sa_channels[-1]])
        self.aggr = GlobalSAModule(
            in_channels=in_channels,
            channels=aggr_channels,
            act=act,
            norm=norm,
            bias=bias,
            order=order,
            pool=global_pool,
        )

        self.dropout = dropout
        self.embedding_dim = aggr_channels[-1]
        self.head = create_cls_head(self.embedding_dim, num_classes)

    def reset_classifier(self, num_classes: int, global_pool: str = "max", **kwargs: Any) -> None:
        self.num_classes = num_classes
        self.aggr.pool = create_pool(global_pool) if isinstance(global_pool, str) else global_pool
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
        intermediates = [{"features": features, "coords": coords, "batch": batch}] if return_intermediates else []
        for block in self.sa_blocks:
            features, coords, batch = block(features, coords, batch)
            if return_intermediates:
                intermediates.append({"features": features, "coords": coords, "batch": batch})

        features, coords, batch = self.aggr(features, coords, batch)

        if return_intermediates:
            return features, coords, batch, intermediates
        return features, coords, batch

    def forward_head(self, x: Tensor, pre_logits: bool = False) -> Tensor:
        if self.dropout:
            x = F.dropout(x, p=float(self.dropout), training=self.training)
        return x if pre_logits else self.head(x)

    def forward(self, features: OptTensor, coords: Tensor, batch: Tensor) -> Tensor:
        features, _, _ = self.forward_features(features, coords, batch)
        return self.forward_head(features)


class PointNet2Segmentation(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        *,
        stem_channels: Optional[int] = None,
        sa_channels: Sequence[Sequence[Union[int, Sequence[int]]]],
        aggr_channels: Optional[Sequence[int]] = None,
        fp_channels: Sequence[Sequence[int]],
        ratios: Sequence[float],
        radii: Sequence[Union[float, Sequence[float]]],
        num_neighbors: Sequence[Union[int, Sequence[int]]],
        act: ActLike = "relu",
        norm: NormLike = "batch_norm1d",
        bias: bool = False,
        order: str = "lan",
        use_coords: bool = True,
        pool: PoolLike = "max",
        dropout: float = 0.0,
        global_pool: PoolLike = "max",
    ) -> None:
        super().__init__()
        sa_channels = ensure_msg_list(
            sa_channels,
            extra_msg="The parameter `sa_channels` must be a sequence compliant "
            f"with Multi-Scale Grouping (MSG) mode, but got {type(sa_channels)}.",
        )

        self.in_channels = in_channels
        self.num_classes = num_classes

        skip_channels = [in_channels]

        self.stem: Optional[nn.Module] = None
        if stem_channels is not None:
            self.stem = nn.Linear(in_channels, stem_channels)
            in_channels = stem_channels
            skip_channels = [stem_channels]

        self.sa_blocks = create_sa_blocks(
            in_channels=in_channels,
            channels=sa_channels,
            ratios=ratios,
            radii=radii,
            num_neighbors=num_neighbors,
            act=act,
            norm=norm,
            bias=bias,
            order=order,
            use_coords=use_coords,
            pool=pool,
        )

        for i in range(len(sa_channels)):
            skip_channels.append(sum([c[-1] for c in sa_channels[i]]))

        self.aggr = None
        if aggr_channels is not None:
            self.aggr = GlobalSAModule(
                in_channels=skip_channels[-1],
                channels=aggr_channels,
                act=act,
                norm=norm,
                bias=bias,
                order=order,
                pool=global_pool,
            )

        self.fp_blocks = create_fp_blocks(
            in_channels=aggr_channels[-1] if aggr_channels is not None else skip_channels.pop(-1),
            skip_channels=skip_channels[::-1],
            fp_channels=fp_channels,
            bias=bias,
            act=act,
            norm=norm,
            order=order,
        )

        self.dropout = dropout
        self.embedding_dim = fp_channels[-1][-1]
        self.head = create_cls_head(self.embedding_dim, num_classes)

    def reset_classifier(self, num_classes: int, global_pool: str = "max", **kwargs: Any) -> None:
        self.num_classes = num_classes
        self.global_pool = create_pool(global_pool) if isinstance(global_pool, str) else global_pool
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
        intermediates = [{"features": features, "coords": coords, "batch": batch}] if return_intermediates else []
        for block in self.sa_blocks:
            features, coords, batch = block(features, coords, batch)
            if return_intermediates:
                intermediates.append({"features": features, "coords": coords, "batch": batch})

        if self.aggr is not None:
            features, coords, batch = self.aggr(features, coords, batch)
        else:
            # In case the GlobalSAModule is not specified (`aggr_channels=None`), then the last intermediate features
            # is in fact the final result and not an intermediate result, so pop it
            intermediates.pop(-1)

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
            coords_skip = intermediate["coords"]
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
