import itertools
from typing import TYPE_CHECKING, Any, Optional, Sequence, Tuple, Union, overload

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_pointcloud.layers import (
    ActLike,
    NormLike,
    PoolLike,
    create_cls_head,
    create_pool,
    create_seg_head,
    linear_block,
)
from torch_pointcloud.utils.imports import optional_import

if TYPE_CHECKING:
    from torch_scatter import scatter

scatter, _ = optional_import("torch_scatter", "scatter")


class TNet(nn.Module):
    """Transformation Network (T-Net) module as described in PointNet paper
    [PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](https://arxiv.org/pdf/1612.00593).

    T-Net predicts an affine transformation matrix that helps align input point clouds
    or feature spaces to a canonical space. This network acts as a mini-PointNet that
    takes points/features as input and outputs a transformation matrix.

    There are two instances of T-Net in PointNet:
    1. Input transform network: Operates on raw point coordinates (k=3)
    2. Feature transform network: Operates on point features (k=64 typically)

    Note:
        The transformation matrix is initialized as an identity matrix and
        adds a residual connection to help with optimization stability.

    Args:
        k: Dimension of input features to transform. Default: 3 for spatial transform.
        mlp1_dims: Dimensions of the first MLP. Default: (64, 128, 1024).
        mlp2_dims: Dimensions of the second MLP after pooling. Default: (512, 256).
        act: Activation function to use. Default: "relu".
        norm: Normalization to use. Default: "batch_norm1d".
        global_pool: Pooling method to use ("max" or "mean"). Default: "max".

    Returns:
        Transformation matrix of shape $(N, k, k)$ where $N$ is the batch size.
    """

    def __init__(
        self,
        k: int = 3,
        mlp1_dims: Sequence[int] = (64, 128, 1024),
        mlp2_dims: Sequence[int] = (512, 256),
        act: ActLike = "relu",
        norm: NormLike = "batch_norm1d",
        global_pool: str = "max",
    ) -> None:
        super().__init__()
        self.k = k
        self.global_pool = global_pool

        mlp1_dims = list(mlp1_dims)
        mlp2_dims = list(mlp2_dims)

        blocks = []
        for in_features, out_features in itertools.pairwise([k] + mlp1_dims):
            block = linear_block(in_features, out_features, act=act, norm=norm, dropout=None, order="lan")
            blocks.append(block)
        self.mlp1 = nn.Sequential(*blocks)

        blocks = []
        for in_features, out_features in itertools.pairwise([mlp1_dims[-1]] + mlp2_dims):
            block = linear_block(in_features, out_features, act=act, norm=norm, dropout=None, order="lan")
            blocks.append(block)
        self.mlp2 = nn.Sequential(*blocks)

        self.transform = nn.Linear(mlp2_dims[-1], k * k)
        nn.init.zeros_(self.transform.weight)
        nn.init.eye_(self.transform.bias.view(k, k))

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """Forward pass of the T-Net.

        Args:
            x: Input tensor of shape $(N, k, *)$ where $N$ is the batch size, $k$ is the dimension of the input features, and $*$ means any number of additional dimensions.
            batch: Batch indices of shape $(N)$ where $N$ is the batch size.

        Returns:
            Transformation matrix of shape $(N, k, k)$ where $N$ is the batch size.
        """

        x = self.mlp1(x)
        x = scatter(x, batch, dim=0, reduce=self.global_pool)
        x = self.mlp2(x)

        x = self.transform(x)
        iden = torch.eye(self.k, dtype=x.dtype, device=x.device)
        x = x.view(-1, self.k, self.k) + iden

        return x[batch]


class PointNetEncoder(nn.Module):
    """PointNet encoder module that processes point clouds to extract global feature vectors as described in PointNet paper
    [PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](https://arxiv.org/pdf/1612.00593).

    The encoder follows the PointNet architecture by:
    1. Applying a spatial transformer network (T-Net) to align input point coordinates
    2. Processing points through the first MLP to extract per-point features
    3. Optionally applying a feature transformer network to align feature space
    4. Processing through the second MLP to extract higher-level features

    Note:
        This is the core feature extraction component of PointNet. The global features can be used
        for classification tasks, while the combination of global and point features can be used
        for segmentation tasks.

    Note:
        To get actual global features, you should apply your own pooling operation on the output of this module,
        like:

        ```python
        model = PointNetClassification(num_classes=10)
        x = model(coords, features, batch)
        global_features = self.global_pool(x, batch)
        ```

    Args:
        coords_dim: Dimension of point coordinates.
        features_dim: Dimension of additional point features.
        mlp1_dims: Dimensions of the first MLP.
        mlp2_dims: Dimensions of the second MLP.
        act: Activation function to use.
        norm: Normalization to use.
        use_features_transform: Whether to use the feature transformer network.
        tnet_mlp1_dims: Dimensions of T-Net first MLP.
        tnet_mlp2_dims: Dimensions of T-Net second MLP.
        tnet_act: Activation function for T-Net.
        tnet_norm: Normalization for T-Net.

    Returns:
        Pre-global features of shape $(N, mlp2_dims[-1])$ where $N$ is the batch size.
    """

    def __init__(
        self,
        coords_dim: int = 3,
        features_dim: int = 0,
        mlp1_dims: Sequence[int] = (64,),
        mlp2_dims: Sequence[int] = (128, 1024),
        act: ActLike = "relu",
        norm: NormLike = "batch_norm1d",
        use_features_transform: bool = True,
        tnet_mlp1_dims: Sequence[int] = (64, 128, 1024),
        tnet_mlp2_dims: Sequence[int] = (512, 256),
        tnet_act: ActLike = "relu",
        tnet_norm: NormLike = "batch_norm1d",
    ) -> None:
        super().__init__()
        mlp1_dims = [coords_dim + features_dim] + list(mlp1_dims)
        mlp2_dims = [mlp1_dims[-1]] + list(mlp2_dims)

        self.stnet = TNet(
            k=coords_dim,
            mlp1_dims=tnet_mlp1_dims,
            mlp2_dims=tnet_mlp2_dims,
            act=tnet_act,
            norm=tnet_norm,
        )

        self.ftnet = None
        if use_features_transform:
            self.ftnet = TNet(
                k=mlp1_dims[-1],
                mlp1_dims=tnet_mlp1_dims,
                mlp2_dims=tnet_mlp2_dims,
                act=tnet_act,
                norm=tnet_norm,
            )

        blocks = []
        for in_features, out_features in itertools.pairwise(mlp1_dims):
            block = linear_block(in_features, out_features, act=act, norm=norm, dropout=None, order="lan")
            blocks.append(block)
        self.mlp1 = nn.Sequential(*blocks)

        blocks = []
        for in_features, out_features in itertools.pairwise(mlp2_dims):
            block = linear_block(in_features, out_features, act=act, norm=norm, dropout=None, order="lan")
            blocks.append(block)
        self.mlp2 = nn.Sequential(*blocks)

    @overload
    def forward(
        self,
        coords: torch.Tensor,
        features: Optional[torch.Tensor],
        batch: torch.Tensor,
    ) -> torch.Tensor: ...

    @overload
    def forward(
        self,
        coords: torch.Tensor,
        features: Optional[torch.Tensor],
        batch: torch.Tensor,
        return_point_features: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]: ...

    def forward(
        self,
        coords: torch.Tensor,
        features: Optional[torch.Tensor],
        batch: torch.Tensor,
        return_point_features: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass of the PointNet encoder.

        Args:
            coords: Point coordinates of shape $(N, coords_dim)$ where $N$ is the batch size.
            features: Additional point features of shape $(N, features_dim)$.
            batch: Batch indices for each point of shape $(N,)$.
            return_point_features: Whether to return per-point features.

        Returns:
            If `return_point_features=False`, returns global features of shape $(N, mlp2_dims[-1])$
                where $N$ is the batch size.
            If `return_point_features=True`, returns a tuple of $(global_features, point_features)$
                where `point_features` is of shape $(N, mlp1_dims[-1])$.
        """

        xt = self.stnet(coords, batch)
        x = torch.bmm(coords.unsqueeze(1), xt).squeeze(1)

        if features is not None:
            x = torch.cat([x, features], dim=1)

        point_features = self.mlp1(x)

        if self.ftnet is not None:
            xt = self.ftnet(point_features, batch)
            point_features = torch.bmm(point_features.unsqueeze(1), xt).squeeze(1)

        x = self.mlp2(point_features)

        if return_point_features:
            return x, point_features
        return x


class PointNetClassification(nn.Module):
    r"""PointNet architecture for 3D point cloud classification tasks as described in PointNet paper
    [PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](https://arxiv.org/pdf/1612.00593).

    This model implements the complete PointNet classification network as described in the
    original paper. It consists of a PointNet encoder to extract global features from point clouds,
    followed by a classification head to predict class probabilities.

    Note:
        This implementation follows the official PointNet architecture for classification,
        achieving invariance to point permutation through max pooling and robustness to
        geometric transformations through the T-Net modules.

    Note:
        To set an empty classification head, use `num_classes=0`.

    Note:
        You can control the activations, normalization, and dropout rate of the encoder and head.
        To skip them, set them to `None`.

    Args:
        num_classes: Number of output classes.
        coords_dim: Dimension of point coordinates.
        features_dim: Dimension of additional point features.
        dropout: Dropout rate applied before classification head.
        global_pool: Pooling method to aggregate point features ("max" or "mean").
        mlp1_dims: Dimensions of encoder's first MLP.
        mlp2_dims: Dimensions of encoder's second MLP.
        act: Activation function to use.
        norm: Normalization to use.
        use_features_transform: Whether to use feature transformation.
        tnet_mlp1_dims: Dimensions of T-Net first MLP.
        tnet_mlp2_dims: Dimensions of T-Net second MLP.
        tnet_act: Activation function for T-Net.
        tnet_norm: Normalization for T-Net.

    Shape:
        - Input: points of shape $(N, \text{coords_dim})$ and optionally features of shape $(N, \text{features_dim})$
        - Output: logits of shape $(N, \text{num_classes})$
    """

    def __init__(
        self,
        num_classes: int,
        coords_dim: int = 3,
        features_dim: int = 0,
        dropout: float = 0.0,
        global_pool: PoolLike = "max",
        mlp1_dims: Sequence[int] = (64,),
        mlp2_dims: Sequence[int] = (128, 1024),
        act: ActLike = "relu",
        norm: NormLike = "batch_norm1d",
        use_features_transform: bool = True,
        tnet_mlp1_dims: Sequence[int] = (64, 128, 1024),
        tnet_mlp2_dims: Sequence[int] = (512, 256),
        tnet_act: ActLike = "relu",
        tnet_norm: NormLike = "batch_norm1d",
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.dropout = dropout

        self.encoder = PointNetEncoder(
            coords_dim=coords_dim,
            features_dim=features_dim,
            mlp1_dims=mlp1_dims,
            mlp2_dims=mlp2_dims,
            act=act,
            norm=norm,
            use_features_transform=use_features_transform,
            tnet_mlp1_dims=tnet_mlp1_dims,
            tnet_mlp2_dims=tnet_mlp2_dims,
            tnet_act=tnet_act,
            tnet_norm=tnet_norm,
        )

        self.num_features = mlp2_dims[-1]

        self.global_pool = create_pool(global_pool)
        self.head = create_cls_head(num_features=self.num_features, num_classes=num_classes)

    def reset_classifier(
        self,
        num_classes: int,
        global_pool: PoolLike = "max",
        **kwargs: Any,
    ) -> None:
        """Resets the classification head with new parameters.

        Note:
            To set an empty classification head, use `num_classes=0`.

        Args:
            num_classes: Number of output classes.
            global_pool: Pooling method to aggregate point features ("max" or "mean").
            **kwargs: Additional keyword arguments to pass to the classification head.
        """
        self.num_classes = num_classes
        self.global_pool = create_pool(global_pool)
        self.head = create_cls_head(num_features=self.num_features, num_classes=self.num_classes, **kwargs)

    def forward_features(
        self,
        coords: torch.Tensor,
        features: Optional[torch.Tensor],
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of the PointNet encoder, returning pre-pooling features.

        Args:
            coords: Point coordinates of shape $(N, coords_dim)$.
            features: Additional point features of shape $(N, features_dim)$.
            batch: Batch indices for each point of shape $(N,)$.

        Returns:
            Pre-pooling features of shape $(N, mlp2_dims[-1])$ where $N$ is the batch size.
        """
        return self.encoder(coords, features, batch)

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
        coords: torch.Tensor,
        features: Optional[torch.Tensor],
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of the PointNet classification network.

        Args:
            coords: Point coordinates of shape $(N, coords_dim)$.
            features: Additional point features of shape $(N, features_dim)$.
            batch: Batch indices for each point of shape $(N,)$.

        Returns:
            Classification logits of shape $(B, num_classes)$.
        """
        x = self.forward_features(coords, features, batch)
        return self.forward_head(x, batch)


class PointNetSegmentation(nn.Module):
    r"""PointNet architecture for point cloud segmentation tasks as described in PointNet paper
    [PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](https://arxiv.org/pdf/1612.00593).

    This model implements the segmentation variant of PointNet as described in the original paper.
    It extracts both local point features and global shape features, combines them for each point,
    and predicts per-point semantic labels. This architecture enables the network to consider
    both local geometry and global context for segmentation.

    Note:
        The key innovation in the segmentation variant is the concatenation of global features
        with per-point features, allowing each point's classification to be informed by both
        local geometry and the global shape context. This enables part segmentation that is
        aware of the overall object structure.

    Args:
        num_classes: Number of segmentation classes.
        coords_dim: Dimension of point coordinates.
        features_dim: Dimension of additional point features.
        dropout: Dropout rate applied before segmentation head.
        mlp1_dims: Dimensions of encoder's first MLP.
        mlp2_dims: Dimensions of encoder's second MLP.
        act: Activation function to use.
        norm: Normalization to use.
        global_pool: Pooling method for global features ("max" or "mean").
        use_features_transform: Whether to use feature transformation.
        tnet_mlp1_dims: Dimensions of T-Net first MLP.
        tnet_mlp2_dims: Dimensions of T-Net second MLP.
        tnet_act: Activation function for T-Net.
        tnet_norm: Normalization for T-Net.
        seg_head_dims: Dimensions of segmentation head MLPs.

    Shape:
        - Input: points of shape $(N, \text{coords_dim})$ and optionally features of shape $(N, \text{features_dim})$
        - Output: logits of shape $(N, \text{num_classes})$
    """

    def __init__(
        self,
        num_classes: int,
        coords_dim: int = 3,
        features_dim: int = 0,
        dropout: float = 0.3,
        mlp1_dims: Sequence[int] = (64,),
        mlp2_dims: Sequence[int] = (128, 1024),
        act: ActLike = "relu",
        norm: NormLike = "batch_norm1d",
        global_pool: PoolLike = "max",
        use_features_transform: bool = True,
        tnet_mlp1_dims: Sequence[int] = (64, 128, 1024),
        tnet_mlp2_dims: Sequence[int] = (512, 256),
        tnet_act: ActLike = "relu",
        tnet_norm: NormLike = "batch_norm1d",
        seg_head_dims: Sequence[int] = (512, 256, 128),
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.dropout = dropout

        self.encoder = PointNetEncoder(
            coords_dim=coords_dim,
            features_dim=features_dim,
            mlp1_dims=mlp1_dims,
            mlp2_dims=mlp2_dims,
            act=act,
            norm=norm,
            use_features_transform=use_features_transform,
            tnet_mlp1_dims=tnet_mlp1_dims,
            tnet_mlp2_dims=tnet_mlp2_dims,
            tnet_act=tnet_act,
            tnet_norm=tnet_norm,
        )

        self.global_pool = create_pool(global_pool)

        # Calculate input dimension for segmentation head
        point_feat_dim = mlp1_dims[-1]
        global_feat_dim = mlp2_dims[-1]
        self.num_features = point_feat_dim + global_feat_dim

        self.head = create_seg_head(
            [self.num_features] + list(seg_head_dims),
            num_classes,
            act=act,
            norm=norm,
            dropout=dropout,
            order="land",
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
        dims = kwargs.get("dims", [])
        self.head = create_seg_head([self.num_features] + list(dims), num_classes, **kwargs)

    def forward_features(
        self,
        coords: torch.Tensor,
        features: Optional[torch.Tensor],
        batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the PointNet encoder, returning pre-pooling features.

        Args:
            coords: Point coordinates of shape $(N, coords_dim)$.
            features: Additional point features of shape $(N, features_dim)$.
            batch: Batch indices for each point of shape $(N,)$.

        Returns:
            A tuple of $(pre_pooling_features, point_features)$
                where `pre_pooling_features` is of shape $(N, mlp2_dims[-1])$
                and `point_features` is of shape $(N, mlp1_dims[-1])$.
        """
        # TODO: Actually do the global pooling in the encoder
        x, point_features = self.encoder(coords, features, batch, return_point_features=True)
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
        global_features = self.global_pool(x, batch)
        # Expand global features to match point features
        global_features = global_features[batch]

        x = torch.cat([point_features, global_features], dim=1)

        if self.dropout:
            x = F.dropout(x, p=float(self.dropout), training=self.training)
        return x if pre_logits else self.head(x)

    def forward(
        self,
        coords: torch.Tensor,
        features: Optional[torch.Tensor],
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of the PointNet segmentation network.

        Args:
            coords: Point coordinates of shape $(N, coords_dim)$.
            features: Additional point features of shape $(N, features_dim)$.
            batch: Batch indices for each point of shape $(N,)$.

        Returns:
            Per-point segmentation logits of shape $(N, num_classes)$.
        """
        x, point_features = self.forward_features(coords, features, batch)
        return self.forward_head(x, point_features, batch)
