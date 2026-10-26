r"""Query-based TransFusion detection head shared by the BEV detectors (TransFusion-L, LION).

The training loss is [`TransFusionHeadLoss`][torch_pointcloud.losses.transfusion.TransFusionHeadLoss].
"""

from typing import Any, Callable, Dict, Literal, Optional, Sequence, TypedDict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import MLP
from torch_geometric.nn.dense.linear import Linear
from typing_extensions import NotRequired

from torch_pointcloud.layers.conv2d_blocks import Conv2dBlock
from torch_pointcloud.layers.transformer import TransformerDecoderLayer
from torch_pointcloud.ops.heatmap import decode_center_codes
from torch_pointcloud.utils.types import Detection3D, OptTensor


class TransFusionHeadOutput(TypedDict):
    r"""Raw per-query TransFusion head predictions plus the dense heatmap.

    Attributes:
        center: BEV center in feature-map cells, shape $(B, 2, Q)$.
        height: Absolute height code of the box, its center or top face (see `height_code`), shape $(B, 1, Q)$.
        dim: Log box size, shape $(B, 3, Q)$.
        rot: $(\sin\theta, \cos\theta)$, shape $(B, 2, Q)$.
        vel: BEV velocity, shape $(B, 2, Q)$.
        iou: IoU-rectification prediction in $[-1, 1]$, shape $(B, 1, Q)$; only from a head built with `use_iou`.
        heatmap: Per-query class logits, shape $(B, C, Q)$.
        query_heatmap_score: Dense-heatmap score gathered at each query cell, shape $(B, C, Q)$.
        query_labels: Initial class of each query, shape $(B, Q)$.
        dense_heatmap: Dense BEV class logits, shape $(B, C, H, W)$.
    """

    center: Tensor
    height: Tensor
    dim: Tensor
    rot: Tensor
    vel: Tensor
    iou: NotRequired[Tensor]
    heatmap: Tensor
    query_heatmap_score: Tensor
    query_labels: Tensor
    dense_heatmap: Tensor


class SeparateHeadTransfusion(nn.Module):
    r"""Per-attribute prediction head of TransFusion.

    One small MLP per box attribute, applied to the per-query features: `center` $(2)$, `height` $(1)$, `dim` $(3)$,
    `rot` $(2)$, `vel` $(2)$, a class `heatmap` and, with `use_iou`, an `iou` $(1)$ rectification score. The
    reference's $1 \times 1$ convolutions over $(B, C, Q)$ are equivalent to linear layers over the flattened query
    dim, so each branch is a plain `MLP`. The branch widths are fixed by the box parametrization; only the number of
    classes and the depths are configurable.

    Args:
        in_channels: Input feature channels.
        head_channels: Hidden channels of the per-attribute MLP.
        num_classes: Number of classes predicted by the heatmap branch.
        num_layers: Number of layers per box-attribute branch.
        num_heatmap_layers: Number of layers of the heatmap branch.
        init_bias: Bias initialization for the heatmap output layer.
        bias: Whether hidden layers carry a bias.
        use_iou: Add the IoU-rectification branch.

    Shape:
        - Input: $(B, C_\text{in}, Q)$ per-query features.
        - Output: dict of $(B, C_\text{attr}, Q)$ tensors keyed by attribute.

    Example:
        >>> head = SeparateHeadTransfusion(32, 16, num_classes=3)
        >>> out = head(torch.randn(2, 32, 10))
        >>> out["center"].shape, out["heatmap"].shape
        (torch.Size([2, 2, 10]), torch.Size([2, 3, 10]))
    """

    def __init__(
        self,
        in_channels: int,
        head_channels: int,
        num_classes: int,
        num_layers: int = 2,
        num_heatmap_layers: int = 2,
        init_bias: float = -2.19,
        bias: bool = False,
        use_iou: bool = False,
    ) -> None:
        super().__init__()
        self.use_iou = use_iou
        hidden = [head_channels] * (num_layers - 1)
        layer_bias = [bias] * (num_layers - 1) + [True]
        factory_kwargs: Dict[str, Any] = dict(act="relu", norm="batch_norm", plain_last=True)
        self.center = MLP([in_channels, *hidden, 2], bias=layer_bias, **factory_kwargs)
        self.height = MLP([in_channels, *hidden, 1], bias=layer_bias, **factory_kwargs)
        self.dim = MLP([in_channels, *hidden, 3], bias=layer_bias, **factory_kwargs)
        self.rot = MLP([in_channels, *hidden, 2], bias=layer_bias, **factory_kwargs)
        self.vel = MLP([in_channels, *hidden, 2], bias=layer_bias, **factory_kwargs)
        if use_iou:
            self.iou = MLP([in_channels, *hidden, 1], bias=layer_bias, **factory_kwargs)
        heatmap_hidden = [head_channels] * (num_heatmap_layers - 1)
        heatmap_bias = [bias] * (num_heatmap_layers - 1) + [True]
        self.heatmap = MLP([in_channels, *heatmap_hidden, num_classes], bias=heatmap_bias, **factory_kwargs)
        final_lin = self.heatmap.lins[-1]
        assert isinstance(final_lin, Linear) and final_lin.bias is not None
        final_lin.bias.data.fill_(init_bias)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        B, _, Q = x.shape
        flat = x.transpose(1, 2).reshape(B * Q, -1)
        outputs = {
            "center": self.center(flat),
            "height": self.height(flat),
            "dim": self.dim(flat),
            "rot": self.rot(flat),
            "vel": self.vel(flat),
            "heatmap": self.heatmap(flat),
        }
        if self.use_iou:
            outputs["iou"] = self.iou(flat)
        return {name: out.reshape(B, Q, -1).transpose(1, 2) for name, out in outputs.items()}


class TransFusionDecoderLayer(nn.Module):
    r"""Post-norm `TransformerDecoderLayer` over $(B, C, N)$ query and key maps.

    The flattened query and key positions are embedded by the layer's own MLPs and added to the attention queries,
    keys and values.

    Args:
        embed_dim: Model channels.
        num_heads: Number of attention heads.
        mlp_dim: FFN hidden width.
        dropout: Dropout probability.
        act: FFN activation.
        self_posembed: Position embedding applied to the flattened query positions (self-attention).
        cross_posembed: Position embedding applied to the flattened key positions (cross-attention).

    Shape:
        - Input: `query` $(B, C, Q)$, `key` $(B, C, K)$, `query_pos` $(B, Q, 2)$, `key_pos` $(B, K, 2)$ and an
          optional boolean `key_padding_mask` $(B, K)$ (`True` masks a key out).
        - Output: $(B, C, Q)$.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float,
        act: Union[str, Callable, None],
        self_posembed: nn.Module,
        cross_posembed: nn.Module,
    ) -> None:
        super().__init__()
        self.self_posembed = self_posembed
        self.cross_posembed = cross_posembed
        self.layer = TransformerDecoderLayer(
            embed_dim,
            num_heads,
            mlp_dim,
            dropout,
            act=act,
            norm_first=False,
            pos_in_value=True,
        )

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        query_pos: Tensor,
        key_pos: Tensor,
        key_padding_mask: OptTensor = None,
    ) -> Tensor:
        b, num_query, _ = query_pos.shape
        query_pos_embed = self.self_posembed(query_pos.reshape(b * num_query, -1)).reshape(b, num_query, -1)
        b, num_key, _ = key_pos.shape
        key_pos_embed = self.cross_posembed(key_pos.reshape(b * num_key, -1)).reshape(b, num_key, -1)
        query = self.layer(
            query.permute(2, 0, 1),
            key.permute(2, 0, 1),
            pos=key_pos_embed.transpose(0, 1),
            query_pos=query_pos_embed.transpose(0, 1),
            memory_key_padding_mask=key_padding_mask,
        )
        return query.permute(1, 2, 0)


class TransFusionHead(nn.Module):
    r"""Query-based TransFusion detection head.

    Reference: :arxiv: [TransFusion: Robust LiDAR-Camera Fusion for 3D Object Detection with Transformers](https://arxiv.org/abs/2203.11496) (Bai et al., 2022).
    Reference implementation: :github: [mit-han-lab/bevfusion](https://github.com/mit-han-lab/bevfusion).

    A shared conv produces a BEV feature map and a dense class heatmap; the top `num_proposals` heatmap peaks (after a
    local-max suppression) initialize object queries that attend to the BEV features, by self-attention and then
    cross-attention to the whole map or to a $(2r + 1)^2$ window around each query (`query_radius`), through a
    [`TransformerDecoderLayer`][torch_pointcloud.layers.transformer.TransformerDecoderLayer]. Per-query box
    attributes are regressed by a
    [`SeparateHeadTransfusion`][torch_pointcloud.layers.transfusion_head.SeparateHeadTransfusion].

    Args:
        in_channels: BEV feature channels feeding the shared conv.
        num_classes: Number of foreground classes.
        grid_size: Voxel grid extent $(n_x, n_y, n_z)$.
        point_cloud_range: Range $(x_\min, y_\min, z_\min, x_\max, y_\max, z_\max)$.
        voxel_size: Voxel size $(v_x, v_y, v_z)$.
        feature_map_stride: BEV stride relating feature pixels to metric coordinates.
        embed_dim: Query / decoder feature width.
        num_proposals: Number of object queries.
        num_heads: Decoder attention heads.
        nms_kernel_size: Heatmap local-max pooling kernel.
        mlp_dim: Decoder FFN width.
        dropout: Decoder dropout.
        act: Decoder FFN activation.
        num_heatmap_layers: Layers in the heatmap branch of the prediction head.
        bias: Whether the convs and linears followed by a norm carry a bias.
        use_iou: Add the IoU-rectification branch to the prediction head.
        query_radius: Half-width, in cells, of the local cross-attention window around each query; `None` attends
            to the whole BEV map.
        iou_rectifier: Exponent blending the class score with the predicted IoU at decode, for a head built with
            `use_iou`.
        height_code: Box point the `height` code regresses: the gravity center, or the top face
            ($z + d_z / 2$).
        local_max_classes: Class indices whose heatmap peaks skip the kernel local-max suppression (crowded small
            objects whose centers fall in neighboring cells).
        post_center_range: Box-center range $(x_\min, y_\min, z_\min, x_\max, y_\max, z_\max)$ kept at decode;
            `None` keeps every box.

    Shape:
        - Input: $(B, C, H, W)$ BEV features, with $H$ along $y$ and $W$ along $x$.
        - Output: a [`TransFusionHeadOutput`][torch_pointcloud.layers.transfusion_head.TransFusionHeadOutput].

    Example:
        >>> head = TransFusionHead(16, 3, (32, 64, 1), (-8.0, -16.0, -1.0, 8.0, 16.0, 1.0), (0.5, 0.5, 2.0),
        ...                        feature_map_stride=2, num_proposals=10).eval()
        >>> out = head(torch.randn(2, 16, 32, 16))
        >>> out["center"].shape, out["dense_heatmap"].shape
        (torch.Size([2, 2, 10]), torch.Size([2, 3, 32, 16]))
    """

    bev_pos: Tensor

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        grid_size: Sequence[int],
        point_cloud_range: Sequence[float],
        voxel_size: Sequence[float],
        *,
        feature_map_stride: int,
        embed_dim: int = 128,
        num_proposals: int = 200,
        num_heads: int = 8,
        nms_kernel_size: int = 3,
        mlp_dim: int = 256,
        dropout: float = 0.1,
        act: Union[str, Callable, None] = "relu",
        num_heatmap_layers: int = 2,
        bias: bool = False,
        use_iou: bool = False,
        query_radius: Optional[int] = None,
        iou_rectifier: Optional[float] = None,
        height_code: Literal["center", "top"] = "center",
        local_max_classes: Sequence[int] = (),
        post_center_range: Optional[Sequence[float]] = None,
    ) -> None:
        if iou_rectifier is not None and not use_iou:
            raise ValueError("`iou_rectifier` needs the `iou` branch of a head built with `use_iou`.")

        super().__init__()
        self.num_classes = num_classes
        self.grid_size = tuple(int(g) for g in grid_size)
        self.point_cloud_range = tuple(float(p) for p in point_cloud_range)
        self.voxel_size = tuple(float(v) for v in voxel_size)
        self.feature_map_stride = feature_map_stride
        self.num_proposals = num_proposals
        self.nms_kernel_size = nms_kernel_size
        self.use_iou = use_iou
        self.query_radius = query_radius
        self.iou_rectifier = iou_rectifier
        self.height_code = height_code
        self.local_max_classes = tuple(int(c) for c in local_max_classes)
        self.post_center_range = None if post_center_range is None else tuple(float(p) for p in post_center_range)

        self.shared_conv = nn.Conv2d(in_channels, embed_dim, kernel_size=3, padding=1)
        self.heatmap_head = nn.Sequential(
            Conv2dBlock(embed_dim, embed_dim, kernel_size=3, padding=1, act="relu", norm="batch_norm", bias=bias),
            nn.Conv2d(embed_dim, num_classes, kernel_size=3, padding=1),
        )
        self.class_encoding = nn.Linear(num_classes, embed_dim)
        self.decoder = TransFusionDecoderLayer(
            embed_dim,
            num_heads,
            mlp_dim,
            dropout,
            act,
            self_posembed=MLP([2, embed_dim, embed_dim], act="relu", norm="batch_norm", plain_last=True),
            cross_posembed=MLP([2, embed_dim, embed_dim], act="relu", norm="batch_norm", plain_last=True),
        )
        self.prediction_head = SeparateHeadTransfusion(
            embed_dim, 64, num_classes, num_heatmap_layers=num_heatmap_layers, bias=bias, use_iou=use_iou
        )
        self.reset_parameters()

        # The (x, y) cell center of every BEV position, in the row-major order of the flattened (H, W) map.
        width = self.grid_size[0] // feature_map_stride
        height = self.grid_size[1] // feature_map_stride
        ys, xs = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
        bev_pos = torch.stack([xs + 0.5, ys + 0.5], dim=-1).to(torch.float32)
        self.register_buffer("bev_pos", bev_pos.view(1, -1, 2), persistent=False)

    def reset_parameters(self) -> None:
        r"""Start the weight matrices of the transformer decoder Xavier-uniform."""
        for parameter in self.decoder.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

    def forward(self, x: Tensor) -> TransFusionHeadOutput:
        batch_size, _, height, width = x.shape
        bev_feat = self.shared_conv(x)
        bev_feat_flatten = bev_feat.view(batch_size, bev_feat.shape[1], -1)
        bev_pos = self.bev_pos.repeat(batch_size, 1, 1)

        dense_heatmap = self.heatmap_head(bev_feat)
        heatmap = dense_heatmap.detach().sigmoid()
        padding = self.nms_kernel_size // 2
        local_max = torch.zeros_like(heatmap)
        local_max_inner = F.max_pool2d(heatmap, kernel_size=self.nms_kernel_size, stride=1, padding=0)
        local_max[:, :, padding:(-padding), padding:(-padding)] = local_max_inner
        for cls_idx in self.local_max_classes:
            local_max[:, cls_idx] = heatmap[:, cls_idx]
        heatmap = heatmap * (heatmap == local_max)
        heatmap = heatmap.view(batch_size, heatmap.shape[1], -1)

        top_proposals = heatmap.view(batch_size, -1).argsort(dim=-1, descending=True)[..., : self.num_proposals]
        top_proposals_class = top_proposals // heatmap.shape[-1]
        top_proposals_index = top_proposals % heatmap.shape[-1]
        query_feat = bev_feat_flatten.gather(
            index=top_proposals_index[:, None, :].expand(-1, bev_feat_flatten.shape[1], -1), dim=-1
        )

        one_hot = F.one_hot(top_proposals_class, num_classes=self.num_classes).float()
        query_cat_encoding = self.class_encoding(one_hot).transpose(1, 2)
        query_feat = query_feat + query_cat_encoding
        query_pos = bev_pos.gather(index=top_proposals_index[:, :, None].expand(-1, -1, bev_pos.shape[-1]), dim=1)

        if self.query_radius is None:
            query_feat = self.decoder(query_feat, bev_feat_flatten, query_pos, bev_pos)
        else:
            query_feat = self._window_attention(
                query_feat, query_pos, top_proposals_index, bev_feat_flatten, bev_pos, height, width
            )

        res_layer = self.prediction_head(query_feat)
        out: TransFusionHeadOutput = {
            "center": res_layer["center"] + query_pos.permute(0, 2, 1),
            "height": res_layer["height"],
            "dim": res_layer["dim"],
            "rot": res_layer["rot"],
            "vel": res_layer["vel"],
            "heatmap": res_layer["heatmap"],
            "query_heatmap_score": heatmap.gather(
                index=top_proposals_index[:, None, :].expand(-1, self.num_classes, -1), dim=-1
            ),
            "query_labels": top_proposals_class,
            "dense_heatmap": dense_heatmap,
        }
        if self.use_iou:
            out["iou"] = res_layer["iou"]
        return out

    def _window_attention(
        self,
        query_feat: Tensor,
        query_pos: Tensor,
        query_index: Tensor,
        key_feat: Tensor,
        key_pos: Tensor,
        height: int,
        width: int,
    ) -> Tensor:
        r"""Run the decoder with each query attending to the $(2r + 1)^2$ BEV cells around its own cell.

        Every query becomes its own batch element whose keys are the window cells; the cells falling outside the map
        are masked out of the attention.

        Args:
            query_feat: Query features, shape $(B, C, Q)$.
            query_pos: Query cell centers, shape $(B, Q, 2)$.
            query_index: Flat row-major index of each query cell, shape $(B, Q)$.
            key_feat: Flattened BEV features, shape $(B, C, H \cdot W)$.
            key_pos: Flattened BEV cell centers, shape $(B, H \cdot W, 2)$.
            height: Rows of the BEV map.
            width: Columns of the BEV map.

        Returns:
            The decoded query features, shape $(B, C, Q)$.
        """
        assert self.query_radius is not None
        batch_size, channels, num_proposals = query_feat.shape
        offsets = torch.arange(-self.query_radius, self.query_radius + 1, device=query_feat.device)
        offset_h, offset_w = torch.meshgrid(offsets, offsets, indexing="ij")
        key_h = (query_index // width)[:, :, None] + offset_h.reshape(1, 1, -1)
        key_w = (query_index % width)[:, :, None] + offset_w.reshape(1, 1, -1)
        key_padding_mask = (key_h < 0) | (key_h >= height) | (key_w < 0) | (key_w >= width)
        key_index = (key_h.clamp(0, height - 1) * width + key_w.clamp(0, width - 1)).view(batch_size, -1)

        window_feat = key_feat.gather(index=key_index[:, None, :].expand(-1, channels, -1), dim=-1)
        window_feat = window_feat.view(batch_size, channels, num_proposals, -1)
        window_feat = window_feat.permute(0, 2, 1, 3).reshape(batch_size * num_proposals, channels, -1)
        window_pos = key_pos.gather(index=key_index[:, :, None].expand(-1, -1, key_pos.shape[-1]), dim=1)
        window_pos = window_pos.view(batch_size * num_proposals, -1, key_pos.shape[-1])

        query_feat = query_feat.permute(0, 2, 1).reshape(batch_size * num_proposals, channels, 1)
        query_pos = query_pos.reshape(batch_size * num_proposals, 1, -1)
        decoded = self.decoder(
            query_feat, window_feat, query_pos, window_pos, key_padding_mask.view(batch_size * num_proposals, -1)
        )
        return decoded.reshape(batch_size, num_proposals, channels).permute(0, 2, 1)

    @torch.no_grad()
    def decode(self, out: TransFusionHeadOutput) -> Detection3D:
        r"""Decode raw head predictions into raw candidate detections (no NMS).

        Multiplies the sigmoid query scores by the gathered dense-heatmap score, recovers oriented boxes, rescores
        each by its predicted IoU when the head has an `iou_rectifier`, and filters by the post-center range. The
        predicted BEV velocity $(v_x, v_y)$ of each kept box is returned under `velocity`. The full candidate set is
        returned, without NMS.

        Args:
            out: The dict returned by `forward`.

        Returns:
            Packed candidate detections `{"boxes": (K, 7), "scores": (K,), "labels": (K,), "batch": (K,)}`
            (PyG layout), plus `"velocity"` $(K, 2)$.
        """
        one_hot = F.one_hot(out["query_labels"], num_classes=self.num_classes).permute(0, 2, 1)
        scores, labels = (out["heatmap"].sigmoid() * out["query_heatmap_score"] * one_hot).max(1)
        boxes = decode_query_boxes(
            out, self.voxel_size, self.point_cloud_range, self.feature_map_stride, height_code=self.height_code
        )
        if self.iou_rectifier is not None:
            iou = ((out["iou"][:, 0] + 1) * 0.5).clamp(0.0, 1.0)
            scores = scores.pow(1 - self.iou_rectifier) * iou.pow(self.iou_rectifier)

        keep = torch.ones_like(scores, dtype=torch.bool)
        if self.post_center_range is not None:
            low, high = boxes.new_tensor(self.post_center_range).view(2, 3)
            keep &= (boxes[..., :3] >= low).all(-1) & (boxes[..., :3] <= high).all(-1)
        batch = torch.arange(scores.shape[0], device=scores.device).view(-1, 1).expand_as(scores)

        return {
            "boxes": boxes[keep],
            "scores": scores[keep],
            "labels": labels[keep],
            "batch": batch[keep],
            "velocity": out["vel"].transpose(1, 2)[keep],
        }


def decode_query_boxes(
    out: TransFusionHeadOutput,
    voxel_size: Sequence[float],
    point_cloud_range: Sequence[float],
    feature_map_stride: int,
    height_code: Literal["center", "top"] = "center",
) -> Tensor:
    r"""Oriented boxes of the TransFusion queries, from their box codes in feature-map cells.

    The `center` code is the absolute cell position, so it is decoded by
    [`decode_center_codes`][torch_pointcloud.ops.heatmap.decode_center_codes] from the map corner; the `rot` code is
    $(\sin\theta, \cos\theta)$.

    Args:
        out: Head output carrying the per-query `center`, `height`, `dim` and `rot` codes.
        voxel_size: Voxel size $(v_x, v_y, v_z)$ in metric units.
        point_cloud_range: Range $(x_\min, y_\min, z_\min, x_\max, y_\max, z_\max)$.
        feature_map_stride: Stride from voxel grid to feature map.
        height_code: Box point the `height` code regresses: the gravity center, or the top face ($z + d_z / 2$).

    Returns:
        Boxes $(c_x, c_y, c_z, d_x, d_y, d_z, \theta)$, shape $(B, Q, 7)$.

    Shape:
        - out: codes of shape $(B, \cdot, Q)$
        - output: $(B, Q, 7)$

    Example:
        >>> out = {"center": torch.full((1, 2, 1), 4.0), "height": torch.zeros(1, 1, 1),
        ...        "dim": torch.zeros(1, 3, 1), "rot": torch.tensor([[[0.0], [1.0]]])}
        >>> decode_query_boxes(out, (0.5, 0.5, 0.5), (-4.0, -4.0, -2.0, 4.0, 4.0, 2.0), 2)
        tensor([[[0., 0., 0., 1., 1., 1., 0.]]])
    """
    codes = torch.cat([out["center"], out["height"], out["dim"], out["rot"].flip(1)], dim=1).transpose(1, 2)
    boxes = decode_center_codes(
        codes, torch.zeros_like(codes[..., :2]), voxel_size, point_cloud_range, feature_map_stride
    )
    if height_code == "top":
        boxes[..., 2] = boxes[..., 2] - boxes[..., 5] * 0.5
    return boxes
