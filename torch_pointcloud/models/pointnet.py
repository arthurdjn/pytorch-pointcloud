import itertools
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter

from torch_pointcloud.layers.activations import ActLike
from torch_pointcloud.layers.blocks import linear_block
from torch_pointcloud.layers.classifier import create_classifier
from torch_pointcloud.layers.norms import NormLike


class TNet(nn.Module):
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

    def forward(self, x: torch.Tensor, batch_idxs: torch.Tensor) -> torch.Tensor:
        x = self.mlp1(x)
        x = scatter(x, batch_idxs, dim=0, reduce=self.global_pool)
        x = self.mlp2(x)

        x = self.transform(x)
        iden = torch.eye(self.k, dtype=x.dtype, device=x.device)
        x = x.view(-1, self.k, self.k) + iden

        return x[batch_idxs]


class PointNetEncoder(nn.Module):
    def __init__(
        self,
        coords_dim: int = 3,
        features_dim: int = 0,
        mlp1_dims: Sequence[int] = (64,),
        mlp2_dims: Sequence[int] = (128, 1024),
        act: ActLike = "relu",
        norm: NormLike = "batch_norm1d",
        global_pool: str = "max",
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

        self.global_pool = global_pool

    def forward(
        self,
        coords: torch.Tensor,
        features: Optional[torch.Tensor],
        batch_idxs: torch.Tensor,
    ) -> torch.Tensor:
        xt = self.stnet(coords, batch_idxs)
        x = torch.bmm(coords.unsqueeze(1), xt).squeeze(1)

        if features is not None:
            x = torch.cat([x, features], dim=1)

        x = self.mlp1(x)

        if self.ftnet is not None:
            xt = self.ftnet(x, batch_idxs)
            x = torch.bmm(x.unsqueeze(1), xt).squeeze(1)

        x = self.mlp2(x)

        return x


class PointNetClassification(nn.Module):
    def __init__(
        self,
        num_classes: int,
        coords_dim: int = 3,
        features_dim: int = 0,
        dropout: float = 0.0,
        global_pool: str = "max",
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
        self.global_pool, self.head = create_classifier(self.num_features, self.num_classes, global_pool)

    def reset_classifier(self, num_classes: int, global_pool: str = "max") -> None:
        self.num_classes = num_classes
        self.global_pool, self.head = create_classifier(self.num_features, self.num_classes, global_pool)

    def forward_features(
        self,
        coords: torch.Tensor,
        features: Optional[torch.Tensor],
        batch_idxs: torch.Tensor,
    ) -> torch.Tensor:
        return self.encoder(coords, features, batch_idxs)

    def forward_head(self, x: torch.Tensor, batch_idxs: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        x = self.global_pool(x, batch_idxs)
        if self.dropout:
            x = F.dropout(x, p=float(self.drop_rate), training=self.training)
        return x if pre_logits else self.head(x)

    def forward(
        self,
        coords: torch.Tensor,
        features: Optional[torch.Tensor],
        batch_idxs: torch.Tensor,
    ) -> torch.Tensor:
        x = self.forward_features(coords, features, batch_idxs)
        x = self.forward_head(x, batch_idxs)
        return x


class PointNetSegmentation(nn.Module):
    def __init__(
        self,
        num_classes: int,
        coords_dim: int = 3,
        features_dim: int = 0,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes

        self.encoder = PointNetEncoder(
            coords_dim=coords_dim,
            features_dim=features_dim,
            mlp1_dims=(64,),  # First MLP: 3 -> 64
            mlp2_dims=(128, 1024),  # Second MLP: 64 -> 128 -> 1024
            use_features_transform=True,
            tnet_mlp1_dims=(64, 128, 1024),
            tnet_mlp2_dims=(512, 256),
        )

        # Segmentation head (as per original paper)
        # Input: point features (128) concatenated with global features (1024)
        self.segmentation_head = nn.Sequential(
            linear_block(1024 + 1024, 512, dropout=dropout, order="land"),
            linear_block(512, 256, dropout=dropout, order="land"),
            linear_block(256, 128, dropout=dropout, order="land"),
            nn.Linear(128, num_classes),
        )

    def forward(
        self,
        coords: torch.Tensor,
        features: Optional[torch.Tensor],
        batch_idxs: torch.Tensor,
    ) -> torch.Tensor:
        # Get point features and global features from encoder
        point_feat, global_feat = self.encoder(coords, features, batch_idxs)

        # Expand global features to match point features
        global_feat = global_feat[batch_idxs]

        # Concatenate point and global features
        x = torch.cat([point_feat, global_feat], dim=1)  # (N, 128 + 1024)

        # Get per-point predictions
        logits = self.segmentation_head(x)

        return logits
