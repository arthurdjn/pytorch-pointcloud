"""ForAINet forest panoptic segmentation model, and the SegmentAnyTree checkpoint trained with it.

{{ paper("2312.15084") }}

{{ paper("2401.15739") }}
"""

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, TypedDict, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import MLP
from torch_geometric.utils import scatter

import torch_pointcloud.transforms as T
from torch_pointcloud.datasets.forinstance import FORINSTANCE_LEARNING_MAP, FORINSTANCE_TREE_LEARNING_MAP
from torch_pointcloud.layers import SparseResNetBlock, StridedSparseConv3d, StridedSparseConvTranspose3d
from torch_pointcloud.layers.act import create_act
from torch_pointcloud.layers.norms import create_norm
from torch_pointcloud.models._base import InstanceSegmentationModel
from torch_pointcloud.models._registry import WeightsDict, register_model
from torch_pointcloud.ops.cluster import mean_shift, region_grow
from torch_pointcloud.utils.conversion import convert_to_spconv_tensor, ensure_tuple
from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.imports import _SPCONV_GITHUB_URL, optional_import
from torch_pointcloud.utils.types import InstanceSegmentation3D

if TYPE_CHECKING:
    import spconv.pytorch as spconv
    from spconv.pytorch import SparseConvTensor
else:
    spconv, _ = optional_import("spconv.pytorch", url=_SPCONV_GITHUB_URL)


class ForAINetOutput(TypedDict):
    r"""Raw ForAINet output over the $V$ voxels of a batch.

    Attributes:
        semantic_logits: Per-voxel semantic logits, shape $(V, C_\text{sem})$.
        offset: Per-voxel offset to the centre of its instance, in the units of `pos`, shape $(V, 3)$.
        embedding: Per-voxel instance embedding, shape $(V, E)$.
        features: Per-voxel backbone features, the input of the proposal scorer, shape $(V, F)$.
        pos: Voxel positions, shape $(V, 3)$.
        pos_grid: Voxel grid coordinates, shape $(V, 3)$.
        batch: Voxel batch index, shape $(V,)$.
    """

    semantic_logits: Tensor
    offset: Tensor
    embedding: Tensor
    features: Tensor
    pos: Tensor
    pos_grid: Tensor
    batch: Tensor


# inspired by: https://github.com/SmartForest-no/SegmentAnyTree/blob/a3561ed8447bbb7938f059ba65a3e9c97d6e2ee9/torch_points3d/modules/MinkowskiEngine/api_modules.py#L235
class ForAINetEncoderBlock(nn.Module):
    r"""One level of the ForAINet sparse U-Net encoder: a convolution, then residual blocks.

    The $3^3$ convolution is a submanifold one at stride 1, where it sets the width to `out_channels`, and a
    `StridedSparseConv3d` onto the parent voxels at stride 2, where it keeps `in_channels`; normalization and
    activation follow it, then `num_blocks` `SparseResNetBlock`s reach `out_channels`.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        stride: Stride of the convolution, $1$ or $2$.
        num_blocks: Number of residual blocks.
        indice_key: spconv index key of the submanifold convolutions of the output level.
        act: Activation passed to `create_act`.
        act_kwargs: Extra keyword arguments for the activation.
        norm: Normalization passed to `create_norm`.
        norm_kwargs: Extra keyword arguments for the normalization.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int,
        num_blocks: int,
        indice_key: str,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        width = in_channels if stride > 1 else out_channels
        if stride > 1:
            self.conv: nn.Module = StridedSparseConv3d(in_channels, width, kernel_size=3, stride=stride)
        else:
            self.conv = spconv.SubMConv3d(in_channels, width, 3, padding=1, bias=False, indice_key=indice_key)
        self.norm = create_norm(norm, width, **(norm_kwargs or {})) or nn.Identity()
        self.act = create_act(act, **(act_kwargs or {})) or nn.Identity()
        self.blocks = spconv.SparseSequential(
            *[
                SparseResNetBlock(
                    width if i == 0 else out_channels,
                    out_channels,
                    indice_key=indice_key,
                    act=act,
                    act_kwargs=act_kwargs,
                    norm=norm,
                    norm_kwargs=norm_kwargs,
                )
                for i in range(num_blocks)
            ]
        )

    def forward(self, x: "SparseConvTensor") -> "SparseConvTensor":
        out = self.conv(x)
        out = out.replace_feature(self.act(self.norm(out.features)))
        return self.blocks(out)


class ForAINetDecoderBlock(nn.Module):
    r"""One level of the ForAINet sparse U-Net decoder: a convolution, then residual blocks.

    The layout of `ForAINetEncoderBlock`, the strided convolution replaced by a `StridedSparseConvTranspose3d` onto
    the voxels of `target`, the next finer level.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        stride: Stride of the convolution, $1$ or $2$.
        num_blocks: Number of residual blocks.
        indice_key: spconv index key of the submanifold convolutions of the output level.
        act: Activation passed to `create_act`.
        act_kwargs: Extra keyword arguments for the activation.
        norm: Normalization passed to `create_norm`.
        norm_kwargs: Extra keyword arguments for the normalization.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int,
        num_blocks: int,
        indice_key: str,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.stride = stride
        width = in_channels if stride > 1 else out_channels
        if stride > 1:
            self.conv: nn.Module = StridedSparseConvTranspose3d(in_channels, width, kernel_size=3, stride=stride)
        else:
            self.conv = spconv.SubMConv3d(in_channels, width, 3, padding=1, bias=False, indice_key=indice_key)
        self.norm = create_norm(norm, width, **(norm_kwargs or {})) or nn.Identity()
        self.act = create_act(act, **(act_kwargs or {})) or nn.Identity()
        self.blocks = spconv.SparseSequential(
            *[
                SparseResNetBlock(
                    width if i == 0 else out_channels,
                    out_channels,
                    indice_key=indice_key,
                    act=act,
                    act_kwargs=act_kwargs,
                    norm=norm,
                    norm_kwargs=norm_kwargs,
                )
                for i in range(num_blocks)
            ]
        )

    def forward(self, x: "SparseConvTensor", target: "SparseConvTensor") -> "SparseConvTensor":
        r"""Convolve `x` onto the voxels of `target`, the voxels of `x` itself at stride 1."""
        out = self.conv(x, target) if self.stride > 1 else self.conv(x)
        out = out.replace_feature(self.act(self.norm(out.features)))
        return self.blocks(out)


class ForAINetUNet(nn.Module):
    r"""ForAINet sparse U-Net: encoder levels down, decoder levels back up to the input voxels.

    Every strided encoder block halves the resolution. Each decoder block but the first takes the output of the
    encoder block at its resolution as a skip (every encoder output but the deepest, deepest first), concatenated
    after its input; a strided decoder block writes onto the voxels of the next finer level. The output holds the
    input voxels, in their order.

    Args:
        in_channels: Number of input feature channels.
        channels: Output width of every encoder block.
        strides: Stride of every encoder block.
        decoder_channels: Output width of every decoder block; as many blocks as in the encoder.
        decoder_strides: Stride of every decoder block; as many strided blocks as in the encoder.
        num_blocks: Residual blocks per encoder and decoder block.
        act: Activation passed to `create_act`.
        act_kwargs: Extra keyword arguments for the activation.
        norm: Normalization passed to `create_norm`.
        norm_kwargs: Extra keyword arguments for the normalization.

    Shape:
        - Input: `SparseConvTensor` with `in_channels` features.
        - Output: `SparseConvTensor` on the same voxels with `decoder_channels[-1]` features.
    """

    def __init__(
        self,
        in_channels: int,
        channels: Sequence[int],
        strides: Sequence[int],
        decoder_channels: Sequence[int],
        decoder_strides: Sequence[int],
        *,
        num_blocks: int = 2,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        channels, strides = ensure_tuple(channels), ensure_tuple(strides)
        decoder_channels, decoder_strides = ensure_tuple(decoder_channels), ensure_tuple(decoder_strides)
        if len(strides) != len(channels) or len(decoder_strides) != len(decoder_channels):
            raise ValueError("Every encoder and decoder block needs a width and a stride.")

        if len(decoder_channels) != len(channels):
            raise ValueError(
                f"The decoder needs one block per encoder block, got {len(decoder_channels)} and {len(channels)}."
            )

        if sum(s > 1 for s in decoder_strides) != sum(s > 1 for s in strides):
            raise ValueError("The decoder must upsample as many times as the encoder downsamples.")

        super().__init__()
        self.channels = channels
        self.strides = strides
        self.decoder_channels = decoder_channels
        self.decoder_strides = decoder_strides

        self.encoder = nn.ModuleList()
        level = 0
        for i, (width, stride) in enumerate(zip(channels, strides)):
            level += stride > 1
            self.encoder.append(
                ForAINetEncoderBlock(
                    channels[i - 1] if i > 0 else in_channels,
                    width,
                    stride=stride,
                    num_blocks=num_blocks,
                    indice_key=f"subm{level}",
                    act=act,
                    act_kwargs=act_kwargs,
                    norm=norm,
                    norm_kwargs=norm_kwargs,
                )
            )

        self.decoder = nn.ModuleList()
        for j, (width, stride) in enumerate(zip(decoder_channels, decoder_strides)):
            level -= stride > 1
            self.decoder.append(
                ForAINetDecoderBlock(
                    decoder_channels[j - 1] + channels[-1 - j] if j > 0 else channels[-1],
                    width,
                    stride=stride,
                    num_blocks=num_blocks,
                    indice_key=f"subm{level}",
                    act=act,
                    act_kwargs=act_kwargs,
                    norm=norm,
                    norm_kwargs=norm_kwargs,
                )
            )

    def forward(self, x: "SparseConvTensor") -> "SparseConvTensor":
        levels = [x]
        skips = []
        out = x
        for encoder_block, stride in zip(self.encoder, self.strides):
            out = encoder_block(out)
            skips.append(out)
            if stride > 1:
                levels.append(out)

        level = len(levels) - 1
        for j, (decoder_block, stride) in enumerate(zip(self.decoder, self.decoder_strides)):
            if j > 0:
                out = out.replace_feature(torch.cat([out.features, skips[-1 - j].features], dim=1))
            level -= stride > 1
            out = decoder_block(out, levels[level])
        return out


# inspired by: https://github.com/SmartForest-no/SegmentAnyTree/blob/a3561ed8447bbb7938f059ba65a3e9c97d6e2ee9/torch_points3d/models/panoptic/PointGroup3heads.py#L22
class ForAINetInstanceSegmentation(InstanceSegmentationModel):
    r"""ForAINet: semantic segmentation and individual trees of forest point clouds.

    Reference: :github: [prs-eth/ForAINet](https://github.com/prs-eth/ForAINet).
    A PointGroup network: a `ForAINetUNet` encodes the voxels, then three MLP heads predict per voxel the semantic
    logits, an offset to the centre of its instance and an instance embedding. `decode` groups the voxels of every
    class but `stuff_classes` into proposals twice, by growing regions over the voxels moved by their offsets and
    by mean shift over the embeddings, and scores every proposal with a second, smaller `ForAINetUNet` over the
    backbone features of its voxels, max-pooled and mapped to a confidence by a linear layer.

    The instances have a single class, so `decode` labels every one $0$: instance labels do not index the semantic
    classes (the registry's `classes`), which `num_semantic_classes` sets.

    `forward` takes the voxel features `x`, their positions `pos` (the units of the offsets and of
    `cluster_radius`), their integer grid coordinates `pos_grid`, non-negative, and the voxel `batch`. It predicts
    over the voxels, so `decode` returns masks over the voxels; `ops.superpoint.point_mask_index(mask_index, inverse)`
    maps them to the points.

    Args:
        in_channels: Number of input voxel feature channels.
        num_semantic_classes: Number of semantic classes of the semantic head, the class count to change for
            fine-tuning (the model takes no `num_classes`: its instances have one class).
        stuff_classes: Semantic classes that hold no instance.
        channels: Output width of every backbone encoder block.
        strides: Stride of every backbone encoder block.
        decoder_channels: Output width of every backbone decoder block.
        decoder_strides: Stride of every backbone decoder block.
        num_blocks: Residual blocks per U-Net block.
        embedding_channels: Width $E$ of the instance embedding.
        scorer_channels: Output width of every scorer encoder block, all strided.
        scorer_decoder_channels: Output width of every scorer decoder block, all strided.
        act: U-Net activation passed to `create_act`.
        act_kwargs: Extra keyword arguments for the U-Net activation.
        norm: U-Net normalization passed to `create_norm`.
        norm_kwargs: Extra keyword arguments for the U-Net normalization.
        head_act: Activation of the hidden layer of the semantic, offset and embedding heads.
        head_act_kwargs: Extra keyword arguments for the head activation.
        cluster_radius: Link radius of the region growing, in the units of `pos`.
        cluster_max_neighbors: Number of links of a voxel in the region growing.
        min_cluster_size: Fewest voxels of a grown region.
        bandwidth: Bandwidth of the mean shift.
        min_mean_shift_size: Fewest voxels of a scene's instance classes for the mean shift to run on it.

    Shape:
        - Input: voxel features $(V, \text{in\_channels})$, voxel positions $(V, 3)$, voxel grid coordinates
          $(V, 3)$ and voxel batch $(V,)$.
        - Output: `ForAINetOutput`, decoded by `decode` into proposal masks over the $V$ voxels.
    """

    def __init__(
        self,
        in_channels: int,
        num_semantic_classes: int,
        *,
        stuff_classes: Sequence[int] = (0,),
        channels: Sequence[int] = (16, 32, 48, 64, 80, 96, 112),
        strides: Sequence[int] = (1, 2, 2, 2, 2, 2, 2),
        decoder_channels: Sequence[int] = (96, 80, 64, 48, 32, 16, 16),
        decoder_strides: Sequence[int] = (2, 2, 2, 2, 2, 2, 1),
        num_blocks: int = 2,
        embedding_channels: int = 5,
        scorer_channels: Sequence[int] = (32, 64),
        scorer_decoder_channels: Sequence[int] = (32, 16),
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        head_act: Union[str, Callable, None] = "leaky_relu",
        head_act_kwargs: Optional[Dict[str, Any]] = None,
        cluster_radius: float = 0.3,
        cluster_max_neighbors: int = 200,
        min_cluster_size: int = 10,
        bandwidth: float = 0.6,
        min_mean_shift_size: int = 4,
    ) -> None:
        super().__init__(in_channels=in_channels, num_classes=1)
        self.num_semantic_classes = num_semantic_classes
        self.stuff_classes = ensure_tuple(stuff_classes)
        self.channels = ensure_tuple(channels)
        self.strides = ensure_tuple(strides)
        self.decoder_channels = ensure_tuple(decoder_channels)
        self.decoder_strides = ensure_tuple(decoder_strides)
        self.num_blocks = num_blocks
        self.embedding_channels = embedding_channels
        self.scorer_channels = ensure_tuple(scorer_channels)
        self.scorer_decoder_channels = ensure_tuple(scorer_decoder_channels)
        self.act = act
        self.act_kwargs = act_kwargs
        self.norm = norm
        self.norm_kwargs = norm_kwargs
        self.head_act = head_act
        self.head_act_kwargs = head_act_kwargs
        self.cluster_radius = cluster_radius
        self.cluster_max_neighbors = cluster_max_neighbors
        self.min_cluster_size = min_cluster_size
        self.bandwidth = bandwidth
        self.min_mean_shift_size = min_mean_shift_size

        self.backbone = ForAINetUNet(
            in_channels,
            self.channels,
            self.strides,
            self.decoder_channels,
            self.decoder_strides,
            num_blocks=num_blocks,
            act=act,
            act_kwargs=act_kwargs,
            norm=norm,
            norm_kwargs=norm_kwargs,
        )
        self.semantic_head = self.configure_head(num_semantic_classes)
        self.offset_head = self.configure_head(3)
        self.embedding_head = self.configure_head(embedding_channels)
        self.scorer = ForAINetUNet(
            self.num_features,
            self.scorer_channels,
            [2] * len(self.scorer_channels),
            self.scorer_decoder_channels,
            [2] * len(self.scorer_decoder_channels),
            num_blocks=num_blocks,
            act=act,
            act_kwargs=act_kwargs,
            norm=norm,
            norm_kwargs=norm_kwargs,
        )
        self.score_head = nn.Linear(self.scorer_decoder_channels[-1], 1)

    def configure_head(self, out_channels: int) -> MLP:
        """Build a per-voxel head of `out_channels` outputs.

        A hidden linear layer without bias, normalization and activation, then a linear layer to `out_channels`.
        """
        return MLP(
            [self.num_features, self.num_features, out_channels],
            act=self.head_act,
            act_kwargs=self.head_act_kwargs,
            norm="batch_norm",
            bias=[False, True],
        )

    @property
    def num_features(self) -> int:
        """Channel count $C$ of the backbone features entering the heads."""
        return self.decoder_channels[-1]

    def forward_features(self, x: Tensor, pos_grid: Tensor, batch: Tensor) -> Tensor:
        r"""Encode the voxels with the backbone U-Net.

        Args:
            x: Voxel features, shape $(V, C_\text{in})$.
            pos_grid: Non-negative voxel grid coordinates, shape $(V, 3)$.
            batch: Voxel batch index, shape $(V,)$.

        Returns:
            The backbone features of every voxel, shape $(V, C)$.
        """
        sparse_x = convert_to_spconv_tensor(x, pos_grid, batch, padding=1)
        return self.backbone(sparse_x).features

    def forward_head(self, x: Tensor, pos: Tensor, pos_grid: Tensor, batch: Tensor) -> ForAINetOutput:
        r"""Predict the semantic logits, offsets and embeddings of every voxel.

        Args:
            x: Backbone features, shape $(V, C)$.
            pos: Voxel positions, shape $(V, 3)$.
            pos_grid: Voxel grid coordinates, shape $(V, 3)$.
            batch: Voxel batch index, shape $(V,)$.

        Returns:
            The `ForAINetOutput` of the batch.
        """
        return {
            "semantic_logits": self.semantic_head(x),
            "offset": self.offset_head(x),
            "embedding": self.embedding_head(x),
            "features": x,
            "pos": pos,
            "pos_grid": pos_grid,
            "batch": batch,
        }

    def forward(self, x: Tensor, pos: Tensor, pos_grid: Tensor, batch: Tensor) -> ForAINetOutput:
        features = self.forward_features(x, pos_grid, batch)
        return self.forward_head(features, pos, pos_grid, batch)

    def score(self, features: Tensor, pos_grid: Tensor, mask_index: Tensor, num_proposals: int) -> Tensor:
        r"""Confidence of every proposal, from the backbone features of its voxels.

        Each proposal is one scene of the scorer U-Net; its output features are max-pooled over the proposal.

        Args:
            features: Backbone features of every voxel, shape $(V, C)$.
            pos_grid: Voxel grid coordinates, shape $(V, 3)$.
            mask_index: Proposal and voxel index of every voxel of a proposal, shape $(2, M)$.
            num_proposals: Number of proposals $K$.

        Returns:
            The confidence in $(0, 1)$ of every proposal, shape $(K,)$.
        """
        proposal, voxel = mask_index
        sparse_x = convert_to_spconv_tensor(
            features[voxel],
            pos_grid[voxel],
            proposal,
            padding=1,
            batch_size=num_proposals,
        )
        pooled = scatter(self.scorer(sparse_x).features, proposal, dim=0, dim_size=num_proposals, reduce="max")
        return self.score_head(pooled).squeeze(1).sigmoid()

    @torch.no_grad()
    def decode(self, output: ForAINetOutput) -> InstanceSegmentation3D:
        r"""Group the voxels into scored instance proposals.

        The voxels whose predicted class is not a stuff class grow into regions after moving by their offsets
        (links within `cluster_radius`, regions of at least `min_cluster_size` voxels), and are clustered by mean
        shift over their embeddings in the scenes holding at least `min_mean_shift_size` of them. Every region and
        every cluster is a proposal, scored by `score`; the two groupings overlap, so proposals do too.

        Args:
            output: The `forward` output.

        Returns:
            The proposals: `mask_index` $(2, M)$, the (proposal, voxel) pairs of their masks over the voxels of the
            batch, sorted by proposal then voxel; `labels`, `scores` and `batch` $(K,)$; and the `semantic_logits`
            $(V, C_\text{sem})$ of the voxels.
        """
        batch = output["batch"]
        stuff = torch.tensor(self.stuff_classes, device=batch.device)
        thing = torch.isin(output["semantic_logits"].argmax(dim=1), stuff, invert=True)
        index = thing.nonzero().squeeze(1)
        thing_batch = batch[index]

        regions = region_grow(
            (output["pos"] + output["offset"])[index],
            self.cluster_radius,
            thing_batch,
            max_num_neighbors=self.cluster_max_neighbors,
            min_cluster_size=self.min_cluster_size,
        )
        counts = torch.bincount(thing_batch)
        shifted = index[counts[thing_batch] >= self.min_mean_shift_size] if index.numel() else index
        modes = mean_shift(output["embedding"][shifted], self.bandwidth, batch[shifted], bin_seeding=True)

        proposals: List[Tensor] = []
        voxels: List[Tensor] = []
        num_proposals = 0
        for members, labels in ((index, regions), (shifted, modes)):
            kept = labels >= 0
            proposals.append(labels[kept] + num_proposals)
            voxels.append(members[kept])
            num_proposals += int(labels.max()) + 1 if bool(kept.any()) else 0
        proposal, voxel = torch.cat(proposals), torch.cat(voxels)
        order = torch.argsort(proposal * batch.numel() + voxel)
        mask_index = torch.stack([proposal[order], voxel[order]])

        proposal_batch = scatter(batch[voxel], proposal, dim=0, dim_size=num_proposals, reduce="max")
        scores = (
            self.score(output["features"], output["pos_grid"], mask_index, num_proposals)
            if num_proposals
            else output["features"].new_zeros(0)
        )
        return {
            "mask_index": mask_index,
            "labels": torch.zeros_like(proposal_batch),
            "scores": scores.float(),
            "batch": proposal_batch,
            "semantic_logits": output["semantic_logits"],
        }


def _forainet_transform(learning_map: Dict[int, int], height: T.DictTransform) -> T.Compose:
    """Single-pass preprocessing of a FOR-instance-style scene in float64 world coordinates: `height` writes the
    fourth feature, positions are centred and cast, 0.2 m voxels keep one random point each."""
    return T.Compose(
        [
            T.Relabel(keys=DataKeys.SEGMENT, labels=learning_map, default=-1, allow_missing_keys=True),
            height,
            T.Shift(keys=DataKeys.POS, method="centroid"),
            T.ToFloat(keys=[DataKeys.POS, DataKeys.HEIGHT]),
            T.Cat(keys=[DataKeys.POS, DataKeys.HEIGHT], dst_key=DataKeys.X, dim=1),
            T.Voxelize(
                pos_key=DataKeys.POS,
                pos_reduce="first",
                size=0.2,
                reduce="first",
                keys=[DataKeys.X, DataKeys.HEIGHT, DataKeys.SEGMENT, DataKeys.INSTANCE],
                dst_pos_grid_key=DataKeys.POS_GRID,
                dst_inverse_key=DataKeys.INVERSE,
                random_first=True,
                allow_missing_keys=True,
            ),
        ]
    )


@register_model(
    "segment-any-tree.for-instance.nibio",
    task="instance-segmentation",
    input_keys=("x", "pos", "pos_grid", "batch"),
    weights=WeightsDict(
        url="hf://torch-pointcloud/segment-any-tree.for-instance.nibio/resolve/0000000000000000000000000000000000000000/model.safetensors",
        dataset="for-instance",
        metrics={"F1": 71.89, "precision": 65.56, "recall": 79.57, "MUCov": 78.17, "mIoU": 99.43},
        classes=("non_tree", "tree"),
        author="nibio",
        license="MIT",
    ),
    # The fourth feature is z above the scene's lowest point.
    transform=_forainet_transform(
        FORINSTANCE_TREE_LEARNING_MAP,
        T.AxisMinOffset(keys=DataKeys.POS, axis=2, dst_keys=DataKeys.HEIGHT),
    ),
    hparams=dict(
        in_channels=4,
        num_semantic_classes=2,
        stuff_classes=[0],
        head_act_kwargs=dict(negative_slope=0.2),
    ),
)
def segment_any_tree_for_instance(**hparams: Any) -> ForAINetInstanceSegmentation:
    return ForAINetInstanceSegmentation(**hparams)


@register_model(
    "forainet.for-instance.prs-eth",
    task="instance-segmentation",
    input_keys=("x", "pos", "pos_grid", "batch"),
    weights=WeightsDict(
        url="hf://torch-pointcloud/forainet.for-instance.prs-eth/resolve/0000000000000000000000000000000000000000/model.safetensors",
        dataset="for-instance",
        metrics={"F1": 79.64, "precision": 78.21, "recall": 81.11, "MUCov": 78.23, "mIoU": 74.42},
        classes=("low_vegetation", "terrain", "stem", "live_branches", "woody_branches"),
        author="prs-eth",
        license="BSD-3-Clause",
    ),
    # The fourth feature is the elevation itself.
    transform=_forainet_transform(
        FORINSTANCE_LEARNING_MAP,
        T.Slice(keys=DataKeys.POS, start=2, stop=3, dim=1, dst_keys=DataKeys.HEIGHT),
    ),
    hparams=dict(
        in_channels=4,
        num_semantic_classes=5,
        stuff_classes=[0, 1],
        head_act_kwargs=dict(negative_slope=0.2),
    ),
)
def forainet_for_instance(**hparams: Any) -> ForAINetInstanceSegmentation:
    return ForAINetInstanceSegmentation(**hparams)
