"""PointNet classification and segmentation models.

{{ paper("1612.00593") }}
"""

import warnings
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union, overload

import torch
import torch.nn as nn
from torch_geometric.nn import MLP

import torch_pointcloud.transforms as T
from torch_pointcloud.datasets.modelnet import MODELNET40_CLASSES
from torch_pointcloud.datasets.s3dis import S3DIS_CLASSES
from torch_pointcloud.datasets.scanobjectnn import SCANOBJECTNN_CLASSES
from torch_pointcloud.layers import (
    PoolLike,
    TNet,
    create_act,
    create_norm,
    create_pool,
)
from torch_pointcloud.utils.data import DataKeys

from ._base import ClassificationModel, SemanticSegmentationModel
from ._registry import WeightsDict, register_model


class PointNetEncoder(nn.Module):
    """PointNet encoder module that processes point clouds to extract global feature vectors as described in the original PointNet paper
    :arxiv: [PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](https://arxiv.org/pdf/1612.00593).

    The encoder follows the PointNet architecture by:

    1. Applying a spatial transformer network (T-Net) to align input point coordinates,
    2. Processing points through the first MLP to extract per-point features,
    3. Optionally applying a feature transformer network to align feature space,
    4. Processing through the second MLP to extract higher-level features.

    Abstract:
        This is the core feature extraction component of PointNet. The global features can be used
        for classification tasks, while the combination of global and point features can be used
        for segmentation tasks.

    Tip:
        To get actual global features, you should apply your own pooling operation on the output of this module,
        like:

        ```python
        import torch
        from torch_geometric.utils import scatter

        from torch_pointcloud.models.pointnet import PointNetEncoder

        encoder = PointNetEncoder()
        pos = torch.rand(2048, 3)
        batch = torch.arange(2).repeat_interleave(1024)
        out = encoder(None, pos, batch)
        global_features = scatter(out, batch, dim=0, reduce="max")
        ```

    Args:
        spatial_dim: Dimension of point coordinates.
        in_channels: Dimension of additional point features.
        mlp1_channels: Dimensions of the first MLP.
        mlp2_channels: Dimensions of the second MLP.
        act: Activation function to use.
        act_kwargs: Keyword arguments for the activation function.
        norm: Normalization to use.
        norm_kwargs: Keyword arguments for the normalization layers.
        use_features_transform: Whether to use the feature transformer network.
        tnet_mlp1_dims: Dimensions of T-Net first MLP.
        tnet_mlp2_dims: Dimensions of T-Net second MLP.
        tnet_act: Activation function for T-Net.
        tnet_act_kwargs: Keyword arguments for the T-Net activation function.
        tnet_norm: Normalization for T-Net.
        tnet_norm_kwargs: Keyword arguments for the T-Net normalization layers.
        act_first: Whether to apply the activation before the normalization in every layer.
        act_last: Whether the last layer of the second MLP applies the activation; `False` returns its normalized
            linear output. Unlike PyG's `plain_last`, the normalization is kept.
        input_transform_uses_features: Whether the input T-Net predicts its transform from the coordinates
            concatenated with the point features, the coordinates alone otherwise.
    """

    def __init__(
        self,
        spatial_dim: int = 3,
        in_channels: int = 0,
        mlp1_channels: Sequence[int] = (64,),
        mlp2_channels: Sequence[int] = (128, 1024),
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        use_features_transform: bool = True,
        tnet_mlp1_dims: Sequence[int] = (64, 128, 1024),
        tnet_mlp2_dims: Sequence[int] = (512, 256),
        tnet_act: Union[str, Callable, None] = "relu",
        tnet_act_kwargs: Optional[Dict[str, Any]] = None,
        tnet_norm: Union[str, Callable, None] = "batch_norm",
        tnet_norm_kwargs: Optional[Dict[str, Any]] = None,
        act_first: bool = True,
        act_last: bool = True,
        input_transform_uses_features: bool = False,
    ) -> None:
        super().__init__()
        self.act_first = act_first
        self.input_transform_uses_features = input_transform_uses_features
        mlp1_channels = [spatial_dim + in_channels] + list(mlp1_channels)
        mlp2_channels = [mlp1_channels[-1]] + list(mlp2_channels)

        self.stnet = TNet(
            local_channels=tnet_mlp1_dims,
            global_channels=tnet_mlp2_dims,
            k=spatial_dim,
            in_channels=mlp1_channels[0] if input_transform_uses_features else None,
            act=tnet_act,
            act_kwargs=tnet_act_kwargs,
            act_first=act_first,
            norm=tnet_norm,
            norm_kwargs=tnet_norm_kwargs,
        )

        self.ftnet = None
        if use_features_transform:
            self.ftnet = TNet(
                local_channels=tnet_mlp1_dims,
                global_channels=tnet_mlp2_dims,
                k=mlp1_channels[-1],
                act=tnet_act,
                act_kwargs=tnet_act_kwargs,
                act_first=act_first,
                norm=tnet_norm,
                norm_kwargs=tnet_norm_kwargs,
            )

        self.mlp1 = MLP(
            mlp1_channels,
            act=act,
            act_kwargs=act_kwargs,
            norm=norm,
            norm_kwargs=norm_kwargs,
            act_first=act_first,
            plain_last=False,
        )
        self.mlp2 = MLP(
            mlp2_channels,
            act=act,
            act_kwargs=act_kwargs,
            norm=norm,
            norm_kwargs=norm_kwargs,
            act_first=act_first,
            plain_last=True,
        )
        self.out_norm = create_norm(norm, mlp2_channels[-1], **(norm_kwargs or {})) or nn.Identity()
        self.out_act = (create_act(act, **(act_kwargs or {})) if act_last else None) or nn.Identity()
        self.register_load_state_dict_pre_hook(_rename_former_out_norm_keys)  # type: ignore[no-untyped-call]

    @overload
    def forward(
        self,
        x: Optional[torch.Tensor],
        pos: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor: ...

    @overload
    def forward(
        self,
        x: Optional[torch.Tensor],
        pos: torch.Tensor,
        batch: torch.Tensor,
        return_point_features: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]: ...

    def forward(
        self,
        x: Optional[torch.Tensor],
        pos: torch.Tensor,
        batch: torch.Tensor,
        return_point_features: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        r"""Forward pass of the PointNet encoder.

        Args:
            x: Additional point features of shape $(N, C)$.
            pos: Point coordinates of shape $(N, D)$.
            batch: Batch indices for each point of shape $(N,)$.
            return_point_features: Whether to return per-point features.

        Returns:
            (Tensor): If `return_point_features=False`, per-point features of shape $(N, C_2)$
                where $N$ is the number of points and $C_2$ is the last dimension of the second MLP.
            (Tuple[Tensor, Tensor]): If `return_point_features=True`, a tuple of:

                - `x` of shape $(N, C_2)$ where $C_2$ is the last dimension of the second MLP.
                - `point_features` of shape $(N, C_1)$ where $C_1$ is the last dimension of the first MLP.
        """

        xp = self.stnet(pos, batch, x if self.input_transform_uses_features else None)
        x = xp if x is None else torch.cat([xp, x], dim=1)

        point_features = self.mlp1(x)

        if self.ftnet is not None:
            point_features = self.ftnet(point_features, batch)

        x = self.mlp2(point_features)
        if self.act_first:
            x = self.out_act(x)
            x = self.out_norm(x)
        else:
            x = self.out_norm(x)
            x = self.out_act(x)

        if return_point_features:
            return x, point_features
        return x


def _rename_former_out_norm_keys(module: nn.Module, state_dict: Dict[str, Any], prefix: str, *args: Any) -> None:
    assert isinstance(module, PointNetEncoder)
    former = f"{prefix}mlp2.norms.{module.mlp2.num_layers - 1}."
    keys = [key for key in state_dict if key.startswith(former)]
    if keys:
        warnings.warn(
            f"The `{former}*` keys of `PointNetEncoder` were renamed to `{prefix}out_norm.*` in 0.0.19 and will stop "
            "loading in 0.0.20; save the loaded model's `state_dict` again to update the checkpoint.",
            FutureWarning,
            stacklevel=2,
        )
    for key in keys:
        state_dict[f"{prefix}out_norm.{key[len(former) :]}"] = state_dict.pop(key)


class PointNetClassification(ClassificationModel):
    r"""PointNet architecture for 3D point cloud classification tasks as described in the original PointNet paper
    :arxiv: [PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](https://arxiv.org/pdf/1612.00593).

    This model implements the complete PointNet classification network as described in the
    original paper. It consists of a PointNet encoder to extract global features from point clouds,
    followed by a classification head to predict class probabilities.

    Abstract:
        This implementation follows the official PointNet architecture for classification,
        achieving invariance to point permutation through max pooling and robustness to
        geometric transformations through the T-Net modules.

    Tip:
        To set an empty classification head, set `num_classes=0`.

    Tip:
        You can control the activations, normalization, and dropout rate of the encoder and head.
        To skip them, set them to `None`.

    Tip:
        The registered checkpoints use `act_first=False` and `act_last=False`. Both layouts share their `state_dict`
        keys, so a strict load does not tell them apart: build a checkpoint with `create_model`, which sets them from
        the registry.

    Args:
        in_channels: Dimension of additional point features.
        num_classes: Number of output classes.
        spatial_dim: Dimension of point coordinates.
        dropout: Dropout rate applied within the classification head.
        global_pool: Pooling method to aggregate point features (`"max"` or `"mean"`).
        mlp1_channels: Dimensions of encoder's first MLP.
        mlp2_channels: Dimensions of encoder's second MLP.
        act: Activation function to use.
        act_kwargs: Keyword arguments for the activation function.
        norm: Normalization to use.
        norm_kwargs: Keyword arguments for the normalization layers.
        use_features_transform: Whether to use feature transformation.
        tnet_mlp1_dims: Dimensions of T-Net first MLP.
        tnet_mlp2_dims: Dimensions of T-Net second MLP.
        tnet_act: Activation function for T-Net.
        tnet_act_kwargs: Keyword arguments for the T-Net activation function.
        tnet_norm: Normalization for T-Net.
        tnet_norm_kwargs: Keyword arguments for the T-Net normalization layers.
        head_channels: Hidden dimensions of the classification head MLP.
        act_first: Whether to apply the activation before the normalization in every layer.
        act_last: Whether the last encoder layer applies the activation; `False` pools the normalized linear
            output. Unlike PyG's `plain_last`, the normalization is kept.
        input_transform_uses_features: Whether the input T-Net predicts its transform from the coordinates
            concatenated with the point features, the coordinates alone otherwise.

    Shape:
        - Input: features of shape $(N, \text{in\_channels})$ (optional), points of shape $(N, \text{spatial\_dim})$
        - Output: logits of shape $(B, \text{num\_classes})$
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        *,
        spatial_dim: int = 3,
        dropout: float = 0.0,
        global_pool: PoolLike = "max",
        mlp1_channels: Sequence[int] = (64,),
        mlp2_channels: Sequence[int] = (128, 1024),
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        use_features_transform: bool = True,
        tnet_mlp1_dims: Sequence[int] = (64, 128, 1024),
        tnet_mlp2_dims: Sequence[int] = (512, 256),
        tnet_act: Union[str, Callable, None] = "relu",
        tnet_act_kwargs: Optional[Dict[str, Any]] = None,
        tnet_norm: Union[str, Callable, None] = "batch_norm",
        tnet_norm_kwargs: Optional[Dict[str, Any]] = None,
        head_channels: Sequence[int] = (512, 256),
        act_first: bool = True,
        act_last: bool = True,
        input_transform_uses_features: bool = False,
    ) -> None:
        super().__init__(in_channels=in_channels, num_classes=num_classes)
        self.dropout = dropout
        self.spatial_dim = spatial_dim
        self.mlp1_channels = mlp1_channels
        self.mlp2_channels = mlp2_channels
        self.act = act
        self.act_kwargs = act_kwargs
        self.norm = norm
        self.norm_kwargs = norm_kwargs
        self.use_features_transform = use_features_transform
        self.tnet_mlp1_dims = tnet_mlp1_dims
        self.tnet_mlp2_dims = tnet_mlp2_dims
        self.tnet_act = tnet_act
        self.tnet_act_kwargs = tnet_act_kwargs
        self.tnet_norm = tnet_norm
        self.tnet_norm_kwargs = tnet_norm_kwargs
        self.head_channels = list(head_channels)
        self.act_first = act_first
        self.act_last = act_last
        self.input_transform_uses_features = input_transform_uses_features

        self.encoder = self.configure_encoder()
        self.global_pool = create_pool(global_pool)
        self.head = self.configure_head()

    def configure_encoder(self) -> PointNetEncoder:
        """Build the `PointNetEncoder` backbone."""
        return PointNetEncoder(
            spatial_dim=self.spatial_dim,
            in_channels=self.in_channels,
            mlp1_channels=self.mlp1_channels,
            mlp2_channels=self.mlp2_channels,
            act=self.act,
            act_kwargs=self.act_kwargs,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
            use_features_transform=self.use_features_transform,
            tnet_mlp1_dims=self.tnet_mlp1_dims,
            tnet_mlp2_dims=self.tnet_mlp2_dims,
            tnet_act=self.tnet_act,
            tnet_act_kwargs=self.tnet_act_kwargs,
            tnet_norm=self.tnet_norm,
            tnet_norm_kwargs=self.tnet_norm_kwargs,
            act_first=self.act_first,
            act_last=self.act_last,
            input_transform_uses_features=self.input_transform_uses_features,
        )

    @property
    def num_features(self) -> int:
        """Feature dimension $C$ of the encoder output."""
        return self.mlp2_channels[-1]

    def configure_head(self) -> nn.Module:
        if self.num_classes == 0:
            return nn.Identity()
        return MLP(
            [self.num_features, *self.head_channels, self.num_classes],
            act=self.act,
            act_kwargs=self.act_kwargs,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
            dropout=self.dropout,
            act_first=self.act_first,
            plain_last=True,
        )

    def reset_classifier(
        self,
        num_classes: int,
        global_pool: Optional[PoolLike] = None,
        **kwargs: Any,
    ) -> None:
        """Resets the classification head with new parameters.

        Args:
            num_classes: Number of output classes.
            global_pool: Pooling method to aggregate point features ("max" or "mean"); `None` keeps the
                current pooling.
            **kwargs: Additional keyword arguments to pass to the classification head.
        """
        self.num_classes = num_classes
        if global_pool is not None:
            self.global_pool = create_pool(global_pool)
        self.head = self.configure_head()

    def forward_features(
        self,
        x: Optional[torch.Tensor],
        pos: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        r"""Forward pass of the PointNet encoder, returning pre-pooling features.

        Args:
            x: Additional point features of shape $(N, \text{in\_channels})$.
            pos: Point coordinates of shape $(N, \text{spatial\_dim})$.
            batch: Batch indices for each point of shape $(N,)$.

        Returns:
            Pre-pooling features of shape $(N, \text{mlp2\_dims}[-1])$ where $N$ is the number of points.
        """
        return self.encoder(x, pos, batch)

    def forward_head(self, x: torch.Tensor, batch: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        r"""Forward pass of the classification head from pre-pooling features.

        Args:
            x: Pre-pooling features of shape $(N, \text{mlp2\_dims}[-1])$ where $N$ is the number of points.
            batch: Batch indices for each point of shape $(N,)$.
            pre_logits: Whether to return pre-logits. Defaults to False.

        Returns:
            Classification logits of shape $(B, \text{num\_classes})$.
        """
        x = self.global_pool(x, batch)
        return x if pre_logits else self.head(x)

    def forward(
        self,
        x: Optional[torch.Tensor],
        pos: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        r"""Forward pass of the PointNet classification network.

        Args:
            x: Additional point features of shape $(N, \text{in\_channels})$.
            pos: Point coordinates of shape $(N, \text{spatial\_dim})$.
            batch: Batch indices for each point of shape $(N,)$.

        Returns:
            Classification logits of shape $(B, \text{num\_classes})$.
        """
        x = self.forward_features(x, pos, batch)
        return self.forward_head(x, batch)


class PointNetSegmentation(SemanticSegmentationModel):
    r"""PointNet architecture for point cloud segmentation tasks as described in the original PointNet paper
    :arxiv: [PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](https://arxiv.org/pdf/1612.00593).

    This model implements the segmentation variant of PointNet as described in the original paper.
    It extracts both local point features and global shape features, combines them for each point,
    and predicts per-point semantic labels. This architecture enables the network to consider
    both local geometry and global context for segmentation.

    Abstract:
        The key innovation in the segmentation variant is the concatenation of global features
        with per-point features, allowing each point's classification to be informed by both
        local geometry and the global shape context. This enables part segmentation that is
        aware of the overall object structure.

    Tip:
        The registered checkpoints use `act_first=False` and `act_last=False`. Both layouts share their `state_dict`
        keys, so a strict load does not tell them apart: build a checkpoint with `create_model`, which sets them from
        the registry.

    Args:
        in_channels: Dimension of additional point features.
        num_classes: Number of segmentation classes.
        spatial_dim: Dimension of point coordinates.
        dropout: Dropout rate applied within the segmentation head.
        mlp1_channels: Dimensions of encoder's first MLP.
        mlp2_channels: Dimensions of encoder's second MLP.
        act: Activation function to use.
        act_kwargs: Keyword arguments for the activation function.
        norm: Normalization to use.
        norm_kwargs: Keyword arguments for the normalization layers.
        global_pool: Pooling method for global features ("max" or "mean").
        use_features_transform: Whether to use feature transformation.
        tnet_mlp1_dims: Dimensions of T-Net first MLP.
        tnet_mlp2_dims: Dimensions of T-Net second MLP.
        tnet_act: Activation function for T-Net.
        tnet_act_kwargs: Keyword arguments for the T-Net activation function.
        tnet_norm: Normalization for T-Net.
        tnet_norm_kwargs: Keyword arguments for the T-Net normalization layers.
        seg_head_dims: Dimensions of segmentation head MLPs.
        act_first: Whether to apply the activation before the normalization in every layer.
        act_last: Whether the last encoder layer applies the activation; `False` pools the normalized linear
            output. Unlike PyG's `plain_last`, the normalization is kept.
        input_transform_uses_features: Whether the input T-Net predicts its transform from the coordinates
            concatenated with the point features, the coordinates alone otherwise.

    Shape:
        - Input: features of shape $(N, \text{in\_channels})$ (optional), points of shape $(N, \text{spatial\_dim})$
        - Output: logits of shape $(N, \text{num\_classes})$
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        *,
        spatial_dim: int = 3,
        dropout: float = 0.3,
        mlp1_channels: Sequence[int] = (64,),
        mlp2_channels: Sequence[int] = (128, 1024),
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        global_pool: PoolLike = "max",
        use_features_transform: bool = True,
        tnet_mlp1_dims: Sequence[int] = (64, 128, 1024),
        tnet_mlp2_dims: Sequence[int] = (512, 256),
        tnet_act: Union[str, Callable, None] = "relu",
        tnet_act_kwargs: Optional[Dict[str, Any]] = None,
        tnet_norm: Union[str, Callable, None] = "batch_norm",
        tnet_norm_kwargs: Optional[Dict[str, Any]] = None,
        seg_head_dims: Sequence[int] = (512, 256, 128),
        act_first: bool = True,
        act_last: bool = True,
        input_transform_uses_features: bool = False,
    ) -> None:
        super().__init__(in_channels=in_channels, num_classes=num_classes)
        self.dropout = dropout
        self.spatial_dim = spatial_dim
        self.mlp1_channels = mlp1_channels
        self.mlp2_channels = mlp2_channels
        self.act = act
        self.act_kwargs = act_kwargs
        self.norm = norm
        self.norm_kwargs = norm_kwargs
        self.use_features_transform = use_features_transform
        self.tnet_mlp1_dims = tnet_mlp1_dims
        self.tnet_mlp2_dims = tnet_mlp2_dims
        self.tnet_act = tnet_act
        self.tnet_act_kwargs = tnet_act_kwargs
        self.tnet_norm = tnet_norm
        self.tnet_norm_kwargs = tnet_norm_kwargs
        self.seg_head_dims = list(seg_head_dims)
        self.act_first = act_first
        self.act_last = act_last
        self.input_transform_uses_features = input_transform_uses_features

        self.encoder = self.configure_encoder()
        self.global_pool = create_pool(global_pool)

        # Calculate input dimension for segmentation head
        self.head = self.configure_head()

    def configure_encoder(self) -> PointNetEncoder:
        """Build the `PointNetEncoder` backbone."""
        return PointNetEncoder(
            spatial_dim=self.spatial_dim,
            in_channels=self.in_channels,
            mlp1_channels=self.mlp1_channels,
            mlp2_channels=self.mlp2_channels,
            act=self.act,
            act_kwargs=self.act_kwargs,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
            use_features_transform=self.use_features_transform,
            tnet_mlp1_dims=self.tnet_mlp1_dims,
            tnet_mlp2_dims=self.tnet_mlp2_dims,
            tnet_act=self.tnet_act,
            tnet_act_kwargs=self.tnet_act_kwargs,
            tnet_norm=self.tnet_norm,
            tnet_norm_kwargs=self.tnet_norm_kwargs,
            act_first=self.act_first,
            act_last=self.act_last,
            input_transform_uses_features=self.input_transform_uses_features,
        )

    @property
    def num_features(self) -> int:
        """Channel count $C$ entering the head: per-point features concatenated with the global feature."""
        return self.mlp1_channels[-1] + self.mlp2_channels[-1]

    def configure_head(self) -> nn.Module:
        if self.num_classes == 0:
            return nn.Identity()
        return MLP(
            [self.num_features, *self.seg_head_dims, self.num_classes],
            act=self.act,
            act_kwargs=self.act_kwargs,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
            dropout=self.dropout,
            act_first=self.act_first,
            plain_last=True,
        )

    def reset_classifier(
        self,
        num_classes: int,
        global_pool: Optional[PoolLike] = None,
        **kwargs: Any,
    ) -> None:
        """Resets the segmentation head with new parameters.

        Args:
            num_classes: Number of output classes.
            global_pool: Pooling method for global features ("max" or "mean"); `None` keeps the current
                pooling.
            **kwargs: Additional keyword arguments to pass to the segmentation head.
        """
        self.num_classes = num_classes
        if global_pool is not None:
            self.global_pool = create_pool(global_pool)
        if "dims" in kwargs:
            self.seg_head_dims = list(kwargs["dims"])
        self.head = self.configure_head()

    def forward_features(
        self,
        x: Optional[torch.Tensor],
        pos: torch.Tensor,
        batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Forward pass of the PointNet encoder, returning pre-pooling features.

        Args:
            x: Additional point features of shape $(N, \text{in\_channels})$.
            pos: Point coordinates of shape $(N, \text{spatial\_dim})$.
            batch: Batch indices for each point of shape $(N,)$.

        Returns:
            A tuple of `(pre_pooling_features, point_features)`
                where `pre_pooling_features` is of shape $(N, \text{mlp2\_dims}[-1])$
                and `point_features` is of shape $(N, \text{mlp1\_dims}[-1])$.
        """
        # TODO: Actually do the global pooling in the encoder
        x, point_features = self.encoder(x, pos, batch, return_point_features=True)
        return x, point_features

    def forward_head(
        self,
        x: torch.Tensor,
        point_features: torch.Tensor,
        batch: torch.Tensor,
        pre_logits: bool = False,
    ) -> torch.Tensor:
        r"""Forward pass of the segmentation head from pre-pooling features.

        Args:
            x: Pre-pooling features of shape $(N, \text{mlp2\_dims}[-1])$ where $N$ is the number of points.
            point_features: Point features of shape $(N, \text{mlp1\_dims}[-1])$.
            batch: Batch indices for each point of shape $(N,)$.
            pre_logits: Whether to return pre-logits. Defaults to False.

        Returns:
            Per-point segmentation logits of shape $(N, \text{num\_classes})$.
        """
        x_global = self.global_pool(x, batch)
        # Expand global features to match point features
        x_global = x_global[batch]

        x = torch.cat([point_features, x_global], dim=1)
        return x if pre_logits else self.head(x)

    def forward(
        self,
        x: Optional[torch.Tensor],
        pos: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        r"""Forward pass of the PointNet segmentation network.

        Args:
            x: Additional point features of shape $(N, \text{in\_channels})$.
            pos: Point coordinates of shape $(N, \text{spatial\_dim})$.
            batch: Batch indices for each point of shape $(N,)$.

        Returns:
            Per-point segmentation logits of shape $(N, \text{num\_classes})$.
        """
        x, point_features = self.forward_features(x, pos, batch)
        return self.forward_head(x, point_features, batch)


@register_model(
    "pointnet.modelnet40",
    task="classification",
    weights=None,
    hparams=dict(in_channels=0, num_classes=40, dropout=0.3),
    transform=T.Compose(
        [
            T.CopyItems(keys=DataKeys.POS, dst_keys=DataKeys.ORIGIN_POS),
            T.FarthestPointSample(
                pos_key=DataKeys.POS,
                keys=[DataKeys.NORMAL],
                num_samples=1024,
                dst_index_key=DataKeys.INDEX,
            ),
            T.Rescale(keys=DataKeys.POS, method="centroid"),
        ]
    ),
)
def pointnet_modelnet40(**hparams: Any) -> PointNetClassification:
    return PointNetClassification(**hparams)


@register_model(
    "pointnet.s3dis-area5",
    task="semantic-segmentation",
    weights=None,
    hparams=dict(in_channels=9, num_classes=13),
    transform=T.Compose(
        [
            T.Cat(keys=[DataKeys.POS, DataKeys.COLOR, DataKeys.NORM_POS], dst_key=DataKeys.X),
        ]
    ),
)
def pointnet_s3dis_area5(**hparams: Any) -> PointNetSegmentation:
    return PointNetSegmentation(**hparams)


@register_model(
    "pointnet.shapenetpart",
    task="semantic-segmentation",
    weights=None,
    hparams=dict(in_channels=0, num_classes=50),
    transform=T.Rescale(keys=DataKeys.POS),
)
def pointnet_shapenetpart(**hparams: Any) -> PointNetSegmentation:
    return PointNetSegmentation(**hparams)


@register_model(
    "pointnet.modelnet40.jiawei-ren",
    task="classification",
    weights=WeightsDict(
        url="hf://torch-pointcloud/pointnet.modelnet40.jiawei-ren/resolve/0000000000000000000000000000000000000000/model.safetensors",
        dataset="modelnet40",
        metrics={"OA": 90.68, "mAcc": 86.54},
        classes=MODELNET40_CLASSES,
        author="jiawei-ren",
        license="Apache-2.0",
    ),
    hparams=dict(in_channels=0, num_classes=40, dropout=0.3, act_first=False, act_last=False),
    transform=T.Compose(
        [
            T.CopyItems(keys=DataKeys.POS, dst_keys=DataKeys.ORIGIN_POS),
            T.Slice(
                keys=[DataKeys.POS, DataKeys.NORMAL],
                stop=1024,
                dst_index_key=DataKeys.INDEX,
                allow_missing_keys=True,
            ),
        ]
    ),
)
def pointnet_jiawei_ren_modelnet40(**hparams: Any) -> PointNetClassification:
    # from the repo: https://github.com/jiawei-ren/ModelNet-C (SimpleView training code, DGCNN protocol)
    return PointNetClassification(**hparams)


@register_model(
    "pointnet.scanobjectnn-hardest.openpoints",
    task="classification",
    weights=WeightsDict(
        url="hf://torch-pointcloud/pointnet.scanobjectnn-hardest.openpoints/resolve/0000000000000000000000000000000000000000/model.safetensors",
        dataset="scanobjectnn-hardest",
        metrics={"OA": 75.16, "mAcc": 71.44},
        classes=SCANOBJECTNN_CLASSES,
        author="openpoints",
        license="MIT",
    ),
    hparams=dict(
        in_channels=1,
        num_classes=15,
        dropout=0.5,
        mlp1_channels=[64, 64],
        mlp2_channels=[64, 128, 1024],
        act_first=False,
        act_last=False,
        input_transform_uses_features=True,
    ),
    transform=T.Compose(
        [
            T.CopyItems(keys=DataKeys.POS, dst_keys=DataKeys.ORIGIN_POS),
            T.FarthestPointSample(pos_key=DataKeys.POS, num_samples=1024, dst_index_key=DataKeys.INDEX),
            T.Slice(keys=DataKeys.POS, start=1, stop=2, dim=1, dst_keys=DataKeys.X),
            T.Shift(keys=DataKeys.X, method="min"),
            T.Rescale(keys=DataKeys.POS, method="centroid"),
        ]
    ),
)
def pointnet_openpoints_scanobjectnn(**hparams: Any) -> PointNetClassification:
    # from the repo: https://github.com/guochengqian/openpoints (the height is the only point feature)
    return PointNetClassification(**hparams)


@register_model(
    "pointnet.s3dis-area5.xu-yan",
    task="semantic-segmentation",
    weights=WeightsDict(
        url="hf://torch-pointcloud/pointnet.s3dis-area5.xu-yan/resolve/0000000000000000000000000000000000000000/model.safetensors",
        dataset="s3dis-area5",
        metrics={"mIoU": 43.88, "OA": 79.39},
        classes=S3DIS_CLASSES,
        author="xu-yan",
        license="MIT",
    ),
    hparams=dict(
        in_channels=6,
        num_classes=13,
        dropout=0.0,
        act_first=False,
        act_last=False,
        input_transform_uses_features=True,
    ),
    transform=T.Cat(keys=[DataKeys.COLOR, DataKeys.NORM_POS], dst_key=DataKeys.X),
)
def pointnet_yanx27_s3dis_area5(**hparams: Any) -> PointNetSegmentation:
    # from the repo: https://github.com/yanx27/Pointnet_Pointnet2_pytorch (sem_seg, S3DIS)
    return PointNetSegmentation(**hparams)
