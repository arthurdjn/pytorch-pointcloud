from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Sequence, Tuple, Union, overload

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MLP

from torch_pointcloud.layers import (
    PoolLike,
    TNet,
    create_pool,
)
from torch_pointcloud.utils.imports import optional_import

from ._base import ClassificationModel, SegmentationModel

if TYPE_CHECKING:
    from torch_scatter import scatter

scatter, _ = optional_import("torch_scatter", "scatter")


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
        encoder = PointNetEncoder()
        out = encoder(x, pos, batch)
        global_features = scatter(out, batch, dim=0, reduce="max")
        ```

    Args:
        spatial_dim: Dimension of point coordinates.
        in_channels: Dimension of additional point features.
        mlp1_dims: Dimensions of the first MLP.
        mlp2_dims: Dimensions of the second MLP.
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

    """

    def __init__(
        self,
        spatial_dim: int = 3,
        in_channels: int = 0,
        mlp1_dims: Sequence[int] = (64,),
        mlp2_dims: Sequence[int] = (128, 1024),
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
    ) -> None:
        super().__init__()
        mlp1_dims = [spatial_dim + in_channels] + list(mlp1_dims)
        mlp2_dims = [mlp1_dims[-1]] + list(mlp2_dims)

        self.stnet = TNet(
            local_channels=tnet_mlp1_dims,
            global_channels=tnet_mlp2_dims,
            k=spatial_dim,
            act=tnet_act,
            act_kwargs=tnet_act_kwargs,
            act_first=True,
            norm=tnet_norm,
            norm_kwargs=tnet_norm_kwargs,
        )

        self.ftnet = None
        if use_features_transform:
            self.ftnet = TNet(
                local_channels=tnet_mlp1_dims,
                global_channels=tnet_mlp2_dims,
                k=mlp1_dims[-1],
                act=tnet_act,
                act_kwargs=tnet_act_kwargs,
                act_first=True,
                norm=tnet_norm,
                norm_kwargs=tnet_norm_kwargs,
            )

        self.mlp1 = MLP(
            mlp1_dims,
            act=act,
            act_kwargs=act_kwargs,
            norm=norm,
            norm_kwargs=norm_kwargs,
            act_first=True,
            plain_last=False,
        )
        self.mlp2 = MLP(
            mlp2_dims,
            act=act,
            act_kwargs=act_kwargs,
            norm=norm,
            norm_kwargs=norm_kwargs,
            act_first=True,
            plain_last=False,
        )

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
            (Tensor): If `return_point_features=False`, returns global features of shape $(N, C_1)$
                where $N$ is the batch size and $C_1$ is the last dimension of the first MLP.
            (Tuple[Tensor, Tensor]): If `return_point_features=True`, returns a tuple of:

                - $\mathbf{x}_{\text{global}}$ is of shape $(N, C_1)$ where $C_1$ is the last dimension of the first MLP.
                - $\mathbf{x}$ is of shape $(N, C_2)$ where $C_2$ is the last dimension of the second MLP.
        """

        xp = self.stnet(pos, batch)
        x = xp if x is None else torch.cat([xp, x], dim=1)

        point_features = self.mlp1(x)

        if self.ftnet is not None:
            point_features = self.ftnet(point_features, batch)

        x = self.mlp2(point_features)

        if return_point_features:
            return x, point_features
        return x


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

    Args:
        num_classes: Number of output classes.
        spatial_dim: Dimension of point coordinates.
        in_channels: Dimension of additional point features.
        dropout: Dropout rate applied before classification head.
        global_pool: Pooling method to aggregate point features (`"max"` or `"mean"`).
        mlp1_dims: Dimensions of encoder's first MLP.
        mlp2_dims: Dimensions of encoder's second MLP.
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

    Shape:
        - Input: features of shape $(N, \text{in\_channels})$ (optional), points of shape $(N, \text{spatial\_dim})$
        - Output: logits of shape $(B, \text{num\_classes})$
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 0,
        spatial_dim: int = 3,
        dropout: float = 0.0,
        global_pool: PoolLike = "max",
        mlp1_dims: Sequence[int] = (64,),
        mlp2_dims: Sequence[int] = (128, 1024),
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
    ) -> None:
        super().__init__(in_channels=in_channels, num_classes=num_classes)
        self.dropout = dropout

        self.encoder = PointNetEncoder(
            spatial_dim=spatial_dim,
            in_channels=in_channels,
            mlp1_dims=mlp1_dims,
            mlp2_dims=mlp2_dims,
            act=act,
            act_kwargs=act_kwargs,
            norm=norm,
            norm_kwargs=norm_kwargs,
            use_features_transform=use_features_transform,
            tnet_mlp1_dims=tnet_mlp1_dims,
            tnet_mlp2_dims=tnet_mlp2_dims,
            tnet_act=tnet_act,
            tnet_act_kwargs=tnet_act_kwargs,
            tnet_norm=tnet_norm,
            tnet_norm_kwargs=tnet_norm_kwargs,
        )

        self.num_features = mlp2_dims[-1]

        self.global_pool = create_pool(global_pool)
        self.head = nn.Identity() if num_classes == 0 else nn.Linear(self.num_features, num_classes)

    def reset_classifier(
        self,
        num_classes: int,
        global_pool: PoolLike = "max",
        **kwargs: Any,
    ) -> None:
        """Resets the classification head with new parameters.

        Args:
            num_classes: Number of output classes.
            global_pool: Pooling method to aggregate point features ("max" or "mean").
            **kwargs: Additional keyword arguments to pass to the classification head.
        """
        self.num_classes = num_classes
        self.global_pool = create_pool(global_pool)
        self.head = nn.Identity() if self.num_classes == 0 else nn.Linear(self.num_features, self.num_classes)

    def forward_features(
        self,
        x: Optional[torch.Tensor],
        pos: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of the PointNet encoder, returning pre-pooling features.

        Args:
            x: Additional point features of shape $(N, in_channels)$.
            pos: Point coordinates of shape $(N, spatial_dim)$.
            batch: Batch indices for each point of shape $(N,)$.

        Returns:
            Pre-pooling features of shape $(N, mlp2_dims[-1])$ where $N$ is the batch size.
        """
        return self.encoder(x, pos, batch)

    def forward_head(self, x: torch.Tensor, batch: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        """Forward pass of the classification head from pre-pooling features.

        Args:
            x: Pre-pooling features of shape $(N, mlp2_dims[-1])$ where $N$ is the batch size.
            batch: Batch indices for each point of shape $(N,)$.
            pre_logits: Whether to return pre-logits. Defaults to False.

        Returns:
            Classification logits of shape $(B, num_classes)$.
        """
        x = self.global_pool(x, batch)
        if self.dropout:
            x = F.dropout(x, p=float(self.dropout), training=self.training)
        return x if pre_logits else self.head(x)

    def forward(
        self,
        x: Optional[torch.Tensor],
        pos: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of the PointNet classification network.

        Args:
            x: Additional point features of shape $(N, in_channels)$.
            pos: Point coordinates of shape $(N, spatial_dim)$.
            batch: Batch indices for each point of shape $(N,)$.

        Returns:
            Classification logits of shape $(B, num_classes)$.
        """
        x = self.forward_features(x, pos, batch)
        return self.forward_head(x, batch)


class PointNetSegmentation(SegmentationModel):
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

    Args:
        num_classes: Number of segmentation classes.
        spatial_dim: Dimension of point coordinates.
        in_channels: Dimension of additional point features.
        dropout: Dropout rate applied before segmentation head.
        mlp1_dims: Dimensions of encoder's first MLP.
        mlp2_dims: Dimensions of encoder's second MLP.
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

    Shape:
        - Input: features of shape $(N, \text{in\_channels})$ (optional), points of shape $(N, \text{spatial\_dim})$
        - Output: logits of shape $(N, \text{num\_classes})$
    """

    def __init__(
        self,
        num_classes: int,
        spatial_dim: int = 3,
        in_channels: int = 0,
        dropout: float = 0.3,
        mlp1_dims: Sequence[int] = (64,),
        mlp2_dims: Sequence[int] = (128, 1024),
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
    ) -> None:
        super().__init__(in_channels=in_channels, num_classes=num_classes)
        self.dropout = dropout

        self.encoder = PointNetEncoder(
            spatial_dim=spatial_dim,
            in_channels=in_channels,
            mlp1_dims=mlp1_dims,
            mlp2_dims=mlp2_dims,
            act=act,
            act_kwargs=act_kwargs,
            norm=norm,
            norm_kwargs=norm_kwargs,
            use_features_transform=use_features_transform,
            tnet_mlp1_dims=tnet_mlp1_dims,
            tnet_mlp2_dims=tnet_mlp2_dims,
            tnet_act=tnet_act,
            tnet_act_kwargs=tnet_act_kwargs,
            tnet_norm=tnet_norm,
            tnet_norm_kwargs=tnet_norm_kwargs,
        )

        self.global_pool = create_pool(global_pool)

        # Calculate input dimension for segmentation head
        point_feat_dim = mlp1_dims[-1]
        global_feat_dim = mlp2_dims[-1]
        self.num_features = point_feat_dim + global_feat_dim

        self.act = act
        self.act_kwargs = act_kwargs
        self.norm = norm
        self.norm_kwargs = norm_kwargs
        self.seg_head_dims = list(seg_head_dims)
        self.head = self.configure_head()

    def configure_head(self) -> nn.Module:
        if self.num_classes == 0:
            return nn.Identity()
        dims = [self.num_features, *self.seg_head_dims]
        return MLP(
            [*dims[:-1], self.num_classes],
            act=self.act,
            act_kwargs=self.act_kwargs,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
            dropout=self.dropout,
            act_first=True,
            plain_last=True,
        )

    def reset_classifier(
        self,
        num_classes: int,
        global_pool: PoolLike = "max",
        **kwargs: Any,
    ) -> None:
        """Resets the segmentation head with new parameters.

        Args:
            num_classes: Number of output classes.
            global_pool: Pooling method for global features ("max" or "mean").
            **kwargs: Additional keyword arguments to pass to the segmentation head.
        """
        self.num_classes = num_classes
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
        """Forward pass of the PointNet encoder, returning pre-pooling features.

        Args:
            x: Additional point features of shape $(N, in_channels)$.
            pos: Point coordinates of shape $(N, spatial_dim)$.
            batch: Batch indices for each point of shape $(N,)$.

        Returns:
            A tuple of $(pre_pooling_features, point_features)$
                where `pre_pooling_features` is of shape $(N, mlp2_dims[-1])$
                and `point_features` is of shape $(N, mlp1_dims[-1])$.
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
        """Forward pass of the segmentation head from pre-pooling features.

        Args:
            x: Pre-pooling features of shape $(N, mlp2_dims[-1])$ where $N$ is the batch size.
            point_features: Point features of shape $(N, mlp1_dims[-1])$.
            batch: Batch indices for each point of shape $(N,)$.
            pre_logits: Whether to return pre-logits. Defaults to False.

        Returns:
            Per-point segmentation logits of shape $(N, num_classes)$.
        """
        x_global = self.global_pool(x, batch)
        # Expand global features to match point features
        x_global = x_global[batch]

        x = torch.cat([point_features, x_global], dim=1)

        if self.dropout:
            x = F.dropout(x, p=float(self.dropout), training=self.training)
        return x if pre_logits else self.head(x)

    def forward(
        self,
        x: Optional[torch.Tensor],
        pos: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of the PointNet segmentation network.

        Args:
            x: Additional point features of shape $(N, in_channels)$.
            pos: Point coordinates of shape $(N, spatial_dim)$.
            batch: Batch indices for each point of shape $(N,)$.

        Returns:
            Per-point segmentation logits of shape $(N, num_classes)$.
        """
        x, point_features = self.forward_features(x, pos, batch)
        return self.forward_head(x, point_features, batch)
