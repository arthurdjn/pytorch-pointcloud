from numbers import Number, Real
from typing import Any, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import FloatTensor, LongTensor, Tensor

from torch_pointcloud.layers.mlp import SharedMLP, shared_mlp2d
from torch_pointcloud.ops import knn


def decimation_indices(ptr: LongTensor, decimation_factor: Union[int, float]) -> Tuple[Tensor, LongTensor]:
    """Get indices which downsample each point cloud by a decimation factor.

    Decimation happens separately for each cloud to prevent emptying smaller
    point clouds. Empty clouds are prevented: clouds will have a least
    one node after decimation.

    Args:
        ptr (LongTensor): indices of samples in the batch.
        decimation_factor (Number): value to divide number of nodes with.
            Should be higher than 1 for downsampling.

    :rtype: (:class:`Tensor`, :class:`LongTensor`): indices for downsampling
        and resulting updated ptr.

    """
    if decimation_factor < 1:
        raise ValueError(
            "Argument `decimation_factor` should be higher than (or equal to) "
            f"1 for downsampling. (Current value: {decimation_factor})"
        )

    batch_size = ptr.size(0) - 1
    bincount = ptr[1:] - ptr[:-1]
    decimated_bincount = torch.div(bincount, decimation_factor, rounding_mode="floor")
    # Decimation should not empty clouds completely.
    decimated_bincount = torch.max(torch.ones_like(decimated_bincount), decimated_bincount)
    idx_decim = torch.cat(
        [(ptr[i] + torch.randperm(bincount[i], device=ptr.device)[: decimated_bincount[i]]) for i in range(batch_size)],
        dim=0,
    )
    # Get updated ptr (e.g. for future decimations)
    ptr_decim = ptr.clone()
    for i in range(batch_size):
        ptr_decim[i + 1] = ptr_decim[i] + decimated_bincount[i]

    return idx_decim, ptr_decim


def decimate(tensors: List[Tensor], ptr: Tensor, decimation_factor: int) -> Tuple[Tuple[Tensor, ...], LongTensor]:
    """Decimate each element of the given tuple of tensors."""
    idx_decim, ptr_decim = decimation_indices(ptr, decimation_factor)
    tensors_decim = tuple(tensor[idx_decim] for tensor in tensors)
    return tensors_decim, ptr_decim


class SharedMLP2(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        transpose: bool = False,
        padding_mode: str = "zeros",
        bn: bool = False,
        activation_fn: Any = None,
    ):
        super().__init__()

        conv_fn = nn.ConvTranspose2d if transpose else nn.Conv2d

        self.conv = conv_fn(in_channels, out_channels, kernel_size, stride=stride, padding_mode=padding_mode)
        self.batch_norm = nn.BatchNorm2d(out_channels, eps=1e-6, momentum=0.99) if bn else None
        self.activation_fn = activation_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Forward pass of the network

        Parameters
        ----------
        input: torch.Tensor, shape (B, d_in, N, K)

        Returns
        -------
        torch.Tensor, shape (B, d_out, N, K)
        """
        x = self.conv(x)
        if self.batch_norm:
            x = self.batch_norm(x)
        if self.activation_fn:
            x = self.activation_fn(x)
        return x


class LocalSpatialEncoding(nn.Module):
    r"""
    Parameters
    ----------
    coords: torch.Tensor, shape (B, N, 3)
        coordinates of the point cloud
    features: torch.Tensor, shape (B, d, N, 1)
        features of the point cloud
    neighbors: tuple

    Returns
    -------
    torch.Tensor, shape (B, 2*d, N, K)
    """

    def __init__(self, num_channels: int, num_neighbors: int) -> None:
        super().__init__()
        self.num_neighbors = num_neighbors
        self.mlp = shared_mlp2d([10, num_channels], act="relu", bn=True)

    # https://github.com/isl-org/Open3D-ML/blob/fcf97c07bf7a113a47d0fcf63760b245c2a2784e/ml3d/torch/models/randlanet.py
    # def gather_neighbor(self, coords, neighbor_indices):
    #     """Gather features based on neighbor indices.

    #     Args:
    #         coords: torch.Tensor of shape (B, N, d)
    #         neighbor_indices: torch.Tensor of shape (B, N, K)

    #     Returns:
    #         gathered neighbors of shape (B, dim, N, K)

    #     """
    #     B, N, K = neighbor_indices.size()
    #     dim = coords.shape[2]

    #     extended_indices = neighbor_indices.unsqueeze(1).expand(B, dim, N, K)
    #     extended_coords = coords.transpose(-2, -1).unsqueeze(-1).expand(
    #         B, dim, N, K)
    #     neighbor_coords = torch.gather(extended_coords, 2,
    #                                    extended_indices)  # (B, dim, N, K)

    #     return neighbor_coords

    def forward(
        self,
        xyz: torch.Tensor,
        features: torch.Tensor,
        dists: torch.Tensor,
        idxs: torch.Tensor,
    ) -> torch.Tensor:
        B, N, K = idxs.size()
        # idx(B, N, K), coords(B, N, 3)
        extended_idx = idxs.unsqueeze(1).expand(B, 3, N, K)  # (B, 3, N, K)
        xyz = xyz.transpose(-2, -1).unsqueeze(-1).expand(B, 3, N, K)  # (B, 3, N, K)
        xyz_neighbors = torch.gather(xyz, 2, extended_idx)  # (B, 3, N, K)

        # relative point position encoding
        concat = torch.cat((xyz, xyz_neighbors, xyz - xyz_neighbors, dists.unsqueeze(-3)), dim=-3).to(xyz.device)
        out_features = self.mlp(concat)
        return torch.cat((out_features, features.expand(B, -1, N, K)), dim=-3)


class AttentivePooling(nn.Module):
    r"""
    Forward pass

    Parameters
    ----------
    x: torch.Tensor, shape (B, d_in, N, K)

    Returns
    -------
    torch.Tensor, shape (B, d_out, N, 1)
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.attn = nn.Sequential(nn.Linear(in_channels, in_channels, bias=False), nn.Softmax(dim=-2))
        self.mlp = shared_mlp2d([in_channels, out_channels], bn=True, act="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.attn(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = torch.sum(x * attn, dim=-1, keepdim=True)  # (B, d_in, N, 1)
        return self.mlp(x)


class LocalFeatureAggregation(nn.Module):
    r"""
    Forward pass

    Parameters
    ----------
    coords: torch.Tensor, shape (B, N, 3)
        coordinates of the point cloud
    features: torch.Tensor, shape (B, d_in, N, 1)
        features of the point cloud

    Returns
    -------
    torch.Tensor, shape (B, 2*d_out, N, 1)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_neighbors: int,
        act: nn.Module = nn.LeakyReLU(),
    ) -> None:
        super().__init__()

        self.num_neighbors = num_neighbors
        self.act = act

        self.mlp1 = shared_mlp2d([in_channels, out_channels // 2], act=nn.LeakyReLU(0.2))
        self.mlp2 = shared_mlp2d([out_channels, 2 * out_channels], act=None)
        self.shortcut = shared_mlp2d([in_channels, 2 * out_channels], bn=True, act=None)

        self.lse1 = LocalSpatialEncoding(out_channels // 2, num_neighbors)
        self.lse2 = LocalSpatialEncoding(out_channels // 2, num_neighbors)

        self.pool1 = AttentivePooling(out_channels, out_channels // 2)
        self.pool2 = AttentivePooling(out_channels, out_channels)

    def forward(self, xyz: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        dists, idxs = knn(xyz, xyz, k=self.num_neighbors)

        skip_features = self.shortcut(features)
        features = self.mlp1(features)

        features = self.lse1(xyz, features, dists, idxs)
        features = self.pool1(features)

        features = self.lse2(xyz, features, dists, idxs)
        features = self.pool2(features)

        return self.act(self.mlp2(features) + skip_features)


class RandLANet(nn.Module):
    r"""
    Forward pass

    Parameters
    ----------
    input: torch.Tensor, shape (B, N, d_in)
    input points

    Returns
    -------
    torch.Tensor, shape (B, num_classes, N)
    segmentation scores for each point
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        num_neighbors: int = 16,
        decimation: int = 4,
    ):
        super(RandLANet, self).__init__()
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_neighbors = num_neighbors
        self.decimation = decimation

        # Authors use 8, which is a bottleneck
        # for the final MLP, and also when num_classes>8
        # or num_features>8.
        num_features_bottleneck = max(32, num_classes, num_features)

        # encoder
        self.fc0 = nn.Linear(num_features, 8)  # d_bottleneck
        self.block1 = LocalFeatureAggregation(8, 16, num_neighbors)  # (num_neighbors, d_bottleneck, 32)
        self.block2 = LocalFeatureAggregation(32, 64, num_neighbors)  # (num_neighbors, 32, 128)
        self.block3 = LocalFeatureAggregation(128, 128, num_neighbors)  # (num_neighbors, 128, 256)
        self.block4 = LocalFeatureAggregation(256, 256, num_neighbors)  # (num_neighbors, 256, 512)
        self.mlp_summit = shared_mlp2d([512, 512], act="relu", bn=True)
        # decoder
        decoder_kwargs = dict(transpose=True, bn=True, activation_fn=nn.ReLU())
        self.fp4 = shared_mlp2d([1024, 256], act="relu", bn=True, plain_last=True)  # [512 + 256, 256]
        self.fp3 = shared_mlp2d([512, 128], act="relu", bn=True, plain_last=True)  # [256 + 128, 128]
        self.fp2 = shared_mlp2d([256, 32], act="relu", bn=True, plain_last=True)  # [128 + 32, 32]
        self.fp1 = shared_mlp2d([64, 8], act="relu", bn=True, plain_last=True)  # [32 + 32, d_bottleneck]
        # head
        self.mlp_classif = SharedMLP([d_bottleneck, 64, 32], dropout=[0.0, 0.5])
        self.fc_classif = Linear(32, num_classes)

        # final semantic prediction
        self.head = nn.Sequential(
            shared_mlp2d([8, 64], bn=True, act=nn.ReLU()),
            shared_mlp2d([64, 32], bn=True, act=nn.ReLU()),
            nn.Dropout(),
            shared_mlp2d([32, num_classes], bn=False, act=None),
        )

    # def forward(self, x, pos, batch, ptr):
    #     x = x if x is not None else pos

    #     b1_out = self.block1(self.fc0(x), pos, batch)
    #     b1_out_decimated, ptr1 = decimate(b1_out, ptr, self.decimation)

    #     b2_out = self.block2(*b1_out_decimated)
    #     b2_out_decimated, ptr2 = decimate(b2_out, ptr1, self.decimation)

    #     b3_out = self.block3(*b2_out_decimated)
    #     b3_out_decimated, ptr3 = decimate(b3_out, ptr2, self.decimation)

    #     b4_out = self.block4(*b3_out_decimated)
    #     b4_out_decimated, _ = decimate(b4_out, ptr3, self.decimation)

    #     mlp_out = (
    #         self.mlp_summit(b4_out_decimated[0]),
    #         b4_out_decimated[1],
    #         b4_out_decimated[2],
    #     )

    #     fp4_out = self.fp4(*mlp_out, *b3_out_decimated)
    #     fp3_out = self.fp3(*fp4_out, *b2_out_decimated)
    #     fp2_out = self.fp2(*fp3_out, *b1_out_decimated)
    #     fp1_out = self.fp1(*fp2_out, *b1_out)

    #     x = self.mlp_classif(fp1_out[0])
    #     logits = self.fc_classif(x)

    #     if self.return_logits:
    #         return logits

    #     probas = logits.log_softmax(dim=-1)
    #     return probas

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:

        N = xyz.size(1)
        d = self.decimation

        coords = xyz[..., :3].clone().cpu()
        x = self.fc_start(xyz).transpose(-2, -1).unsqueeze(-1)
        x = self.bn_start(x)  # shape (B, d, N, 1)

        decimation_ratio = 1

        # <<<<<<<<<< ENCODER
        x_stack = []

        permutation = torch.randperm(N)
        coords = coords[:, permutation]
        x = x[:, :, permutation]

        for lfa in self.encoder:
            # at iteration i, x.shape = (B, N//(d**i), d_in)
            x = lfa(coords[:, : N // decimation_ratio], x)
            x_stack.append(x.clone())
            decimation_ratio *= d
            x = x[:, :, : N // decimation_ratio]

        # # >>>>>>>>>> ENCODER

        x = self.mlp(x)

        # <<<<<<<<<< DECODER
        for mlp in self.decoder:
            neighbors, _ = knn(
                coords[:, : N // decimation_ratio].cpu().contiguous(),  # original set
                coords[:, : d * N // decimation_ratio].cpu().contiguous(),  # upsampled set
                1,
            )  # shape (B, N, 1)
            neighbors = neighbors.to(self.device)

            extended_neighbors = neighbors.unsqueeze(1).expand(-1, x.size(1), -1, 1)

            x_neighbors = torch.gather(x, -2, extended_neighbors)

            x = torch.cat((x_neighbors, x_stack.pop()), dim=1)

            x = mlp(x)

            decimation_ratio //= d

        # >>>>>>>>>> DECODER
        # inverse permutation
        x = x[:, :, torch.argsort(permutation)]

        scores = self.fc_end(x)

        return scores.squeeze(-1)
