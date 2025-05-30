from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type

import pointops
import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax

from torch_pointcloud.layers import (
    ActLike,
    NormLike,
    PoolLike,
    create_act,
    create_cls_head,
    create_norm,
    create_pool,
    linear_block,
)
from torch_pointcloud.layers.dropouts import DropPath
from torch_pointcloud.utils.conversion import ensure_tuple, ensure_tuple_size
from torch_pointcloud.utils.imports import optional_import

# from torch_pointcloud.utils.ops import softmax, voxel_grid
from torch_pointcloud.utils.types import OptTensor, ValueCollection

if TYPE_CHECKING:
    from torch_cluster import fps, knn
    from torch_scatter import scatter_add

fps, _ = optional_import("torch_cluster", name="fps")
knn, _ = optional_import("torch_cluster", name="knn")
scatter_add, _ = optional_import("torch_scatter", name="scatter_add")


class PointTransformerConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_dim: int,
        num_groups: int = 8,
        norm: NormLike = "batch_norm1d",
        act: ActLike = "relu",
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.mid_channels = out_channels // 1
        self.out_channels = out_channels
        self.num_groups = num_groups

        self.lin_q = nn.Linear(in_channels, out_channels)
        self.lin_k = nn.Linear(in_channels, out_channels)
        self.lin_v = nn.Linear(in_channels, out_channels)
        self.lin_p = nn.Sequential(
            nn.Linear(spatial_dim, spatial_dim),
            # create_norm(norm, spatial_dim),
            create_act(act),
            nn.Linear(spatial_dim, out_channels),
        )
        self.lin_w = nn.Sequential(
            create_norm(norm, out_channels),
            create_act(act),
            nn.Linear(out_channels, out_channels // num_groups),
            create_norm(norm, out_channels // num_groups),
            create_act(act),
            nn.Linear(out_channels // num_groups, out_channels // num_groups),
        )

    def forward(self, coords: Tensor, features: Tensor, edge_index: Tensor) -> Tensor:
        x_q = self.lin_q(features)
        x_k = self.lin_k(features)
        x_v = self.lin_v(features)

        row, col = edge_index
        p_r = coords[col] - coords[row]
        p_r_encoded = self.lin_p(p_r)
        p_r_reduced = torch.sum(p_r_encoded.view(p_r_encoded.size(0), -1, self.mid_channels), dim=1)

        x_k_j = x_k[col]
        x_q_i = x_q[row]
        r_qk = x_k_j - x_q_i + p_r_reduced

        w = self.lin_w(r_qk)
        print(f"{w.shape = }")
        print(f"{col.shape = } | {col[:15] = }")
        print(f"{row.shape = } | {row[:15] = }")
        w = softmax(w, row, dim=0)
        x_v_j = x_v[col]
        x_v_j = x_v_j + p_r_encoded
        x_v_j = x_v_j.view(x_v_j.size(0), self.num_groups, -1)

        x_attn = torch.zeros((x_v_j.size(0), self.num_groups, x_v_j.size(2)), device=features.device)
        x_attn = x_v_j * w.unsqueeze(1)
        x_attn = x_attn.view(x_attn.size(0), -1)
        return scatter_add(x_attn, row, dim=0, dim_size=features.size(0))


class DownsampleBlock(nn.Module):
    def __init__(self, in_planes: int, out_planes: int, stride: int = 1, nsample: int = 16):
        super().__init__()
        self.stride, self.nsample = stride, nsample
        self.linear = nn.Linear(3 + in_planes, out_planes, bias=False)
        self.bn = nn.BatchNorm1d(out_planes)
        self.relu = nn.ReLU(inplace=True)


class TransitionDown(nn.Module):
    def __init__(self, in_planes: int, out_planes: int, stride: int = 1, nsample: int = 16):
        super().__init__()
        self.stride, self.nsample = stride, nsample
        if stride != 1:
            self.linear = nn.Linear(3 + in_planes, out_planes, bias=False)
            self.pool = nn.MaxPool1d(nsample)
        else:
            self.linear = nn.Linear(in_planes, out_planes, bias=False)
        self.bn = nn.BatchNorm1d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pxo: Tuple[Tensor, Tensor, Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        p, x, o = pxo  # (n, 3), (n, c), (b)
        if self.stride != 1:
            n_o, count = [o[0].item() // self.stride], o[0].item() // self.stride
            for i in range(1, o.shape[0]):
                count += (o[i].item() - o[i - 1].item()) // self.stride
                n_o.append(count)
            n_o = torch.cuda.IntTensor(n_o)
            idx = pointops.farthest_point_sampling(p, o, n_o)  # (m)
            n_p = p[idx.long(), :]  # (m, 3)
            x, _ = pointops.knn_query_and_group(
                x,
                p,
                offset=o,
                new_xyz=n_p,
                new_offset=n_o,
                nsample=self.nsample,
                with_xyz=True,
            )
            x = self.relu(self.bn(self.linear(x).transpose(1, 2).contiguous()))  # (m, c, nsample)
            x = self.pool(x).squeeze(-1)  # (m, c)
            p, o = n_p, n_o
        else:
            x = self.relu(self.bn(self.linear(x)))  # (n, c)
        return p, x, o


class TransitionUp(nn.Module):
    def __init__(self, in_planes: int, out_planes: Optional[int] = None):
        super().__init__()
        if out_planes is None:
            self.linear1 = nn.Sequential(
                nn.Linear(2 * in_planes, in_planes),
                nn.BatchNorm1d(in_planes),
                nn.ReLU(inplace=True),
            )
            self.linear2 = nn.Sequential(nn.Linear(in_planes, in_planes), nn.ReLU(inplace=True))
        else:
            self.linear1 = nn.Sequential(
                nn.Linear(out_planes, out_planes),
                nn.BatchNorm1d(out_planes),
                nn.ReLU(inplace=True),
            )
            self.linear2 = nn.Sequential(
                nn.Linear(in_planes, out_planes),
                nn.BatchNorm1d(out_planes),
                nn.ReLU(inplace=True),
            )

    def forward(
        self,
        pxo1: Tuple[Tensor, Tensor, Tensor],
        pxo2: Optional[Tuple[Tensor, Tensor, Tensor]] = None,
    ) -> Tensor:
        if pxo2 is None:
            _, x, o = pxo1  # (n, 3), (n, c), (b)
            x_tmp = []
            for i in range(o.shape[0]):
                if i == 0:
                    s_i, e_i, cnt = 0, o[0], o[0]
                else:
                    s_i, e_i, cnt = o[i - 1], o[i], o[i] - o[i - 1]
                x_b = x[s_i:e_i, :]
                x_b = torch.cat((x_b, self.linear2(x_b.sum(0, True) / cnt).repeat(cnt, 1)), 1)
                x_tmp.append(x_b)
            x = torch.cat(x_tmp, 0)
            x = self.linear1(x)
        else:
            p1, x1, o1 = pxo1
            p2, x2, o2 = pxo2
            x = self.linear1(x1) + pointops.interpolation(p2, p1, self.linear2(x2), o2, o1)
        return x


class Bottleneck(nn.Module):
    expansion: int = 1

    def __init__(self, in_planes: int, planes: int, share_planes: int = 8, nsample: int = 16):
        super().__init__()
        self.linear1 = nn.Linear(in_planes, planes, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.transformer = PointTransformerConv(planes, planes, share_planes, nsample)
        self.bn2 = nn.BatchNorm1d(planes)
        self.linear3 = nn.Linear(planes, planes * self.expansion, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pxo: Tuple[Tensor, Tensor, Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        p, x, o = pxo  # (n, 3), (n, c), (b)
        identity = x
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.relu(self.bn2(self.transformer([p, x, o])))
        x = self.bn3(self.linear3(x))
        x += identity
        x = self.relu(x)
        return p, x, o


class PointTransformerSeg(nn.Module):
    def __init__(self, block: Type[Bottleneck], blocks: List[int], in_channels: int = 6, num_classes: int = 13):
        super().__init__()
        self.in_channels = in_channels
        self.in_planes, planes = in_channels, [32, 64, 128, 256, 512]
        fpn_planes, fpnhead_planes, share_planes = 128, 64, 8
        stride, nsample = [1, 4, 4, 4, 4], [8, 16, 16, 16, 16]
        self.enc1 = self._make_enc(
            block,
            planes[0],
            blocks[0],
            share_planes,
            stride=stride[0],
            nsample=nsample[0],
        )  # N/1
        self.enc2 = self._make_enc(
            block,
            planes[1],
            blocks[1],
            share_planes,
            stride=stride[1],
            nsample=nsample[1],
        )  # N/4
        self.enc3 = self._make_enc(
            block,
            planes[2],
            blocks[2],
            share_planes,
            stride=stride[2],
            nsample=nsample[2],
        )  # N/16
        self.enc4 = self._make_enc(
            block,
            planes[3],
            blocks[3],
            share_planes,
            stride=stride[3],
            nsample=nsample[3],
        )  # N/64
        self.enc5 = self._make_enc(
            block,
            planes[4],
            blocks[4],
            share_planes,
            stride=stride[4],
            nsample=nsample[4],
        )  # N/256
        self.dec5 = self._make_dec(block, planes[4], 1, share_planes, nsample=nsample[4], is_head=True)  # transform p5
        self.dec4 = self._make_dec(block, planes[3], 1, share_planes, nsample=nsample[3])  # fusion p5 and p4
        self.dec3 = self._make_dec(block, planes[2], 1, share_planes, nsample=nsample[2])  # fusion p4 and p3
        self.dec2 = self._make_dec(block, planes[1], 1, share_planes, nsample=nsample[1])  # fusion p3 and p2
        self.dec1 = self._make_dec(block, planes[0], 1, share_planes, nsample=nsample[0])  # fusion p2 and p1
        self.cls = nn.Sequential(
            nn.Linear(planes[0], planes[0]),
            nn.BatchNorm1d(planes[0]),
            nn.ReLU(inplace=True),
            nn.Linear(planes[0], num_classes),
        )

    def _make_enc(
        self,
        block: Type[Bottleneck],
        planes: int,
        blocks: int,
        share_planes: int = 8,
        stride: int = 1,
        nsample: int = 16,
    ) -> nn.Sequential:
        layers: List[nn.Module] = [TransitionDown(self.in_planes, planes * block.expansion, stride, nsample)]
        self.in_planes = planes * block.expansion
        for _ in range(blocks):
            layers.append(block(self.in_planes, self.in_planes, share_planes, nsample=nsample))
        return nn.Sequential(*layers)

    def _make_dec(
        self,
        block: Type[Bottleneck],
        planes: int,
        blocks: int,
        share_planes: int = 8,
        nsample: int = 16,
        is_head: bool = False,
    ) -> nn.Sequential:
        layers: List[nn.Module] = [TransitionUp(self.in_planes, None if is_head else planes * block.expansion)]
        self.in_planes = planes * block.expansion
        for _ in range(blocks):
            layers.append(block(self.in_planes, self.in_planes, share_planes, nsample=nsample))
        return nn.Sequential(*layers)

    def forward(self, data_dict: Dict[str, Tensor]) -> Tensor:
        p0 = data_dict["coord"]
        x0 = data_dict["feat"]
        o0 = data_dict["offset"].int()
        p1, x1, o1 = self.enc1([p0, x0, o0])
        p2, x2, o2 = self.enc2([p1, x1, o1])
        p3, x3, o3 = self.enc3([p2, x2, o2])
        p4, x4, o4 = self.enc4([p3, x3, o3])
        p5, x5, o5 = self.enc5([p4, x4, o4])
        x5 = self.dec5[1:]([p5, self.dec5[0]([p5, x5, o5]), o5])[1]
        x4 = self.dec4[1:]([p4, self.dec4[0]([p4, x4, o4], [p5, x5, o5]), o4])[1]
        x3 = self.dec3[1:]([p3, self.dec3[0]([p3, x3, o3], [p4, x4, o4]), o3])[1]
        x2 = self.dec2[1:]([p2, self.dec2[0]([p2, x2, o2], [p3, x3, o3]), o2])[1]
        x1 = self.dec1[1:]([p1, self.dec1[0]([p1, x1, o1], [p2, x2, o2]), o1])[1]
        x = self.cls(x1)
        return x
