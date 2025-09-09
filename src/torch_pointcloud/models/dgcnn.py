import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch.nn import Linear
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP, DynamicEdgeConv, global_max_pool

from ._base import ClassificationModel, SegmentationModel
from ._registry import register_model


class DGCNNEncoder(nn.Module):
    def __init__(self):
        pass


class DGCNNDecoder(nn.Module):
    def __init__(self):
        pass


class DGCNNClassification(ClassificationModel):
    def __init__(self, out_channels, k=20, aggr="max"):
        super().__init__()
        self.conv1 = DynamicEdgeConv(MLP([2 * 3, 64, 64, 64]), k, aggr)
        self.conv2 = DynamicEdgeConv(MLP([2 * 64, 128]), k, aggr)
        self.lin1 = Linear(128 + 64, 1024)
        self.mlp = MLP([1024, 512, 256, out_channels], dropout=0.5, norm=None)

    def forward(self, data):
        pos, batch = data.pos, data.batch
        x1 = self.conv1(pos, batch)
        x2 = self.conv2(x1, batch)
        out = self.lin1(torch.cat([x1, x2], dim=1))
        out = global_max_pool(out, batch)
        out = self.mlp(out)
        return F.log_softmax(out, dim=1)


class DGCNNSegmentation(SegmentationModel):
    def __init__(self, out_channels, k=20, aggr="max"):
        super().__init__()
        self.conv1 = DynamicEdgeConv(MLP([2 * 6, 64, 64]), k, aggr)
        self.conv2 = DynamicEdgeConv(MLP([2 * 64, 64, 64]), k, aggr)
        self.conv3 = DynamicEdgeConv(MLP([2 * 64, 64, 64]), k, aggr)
        self.mlp = MLP([3 * 64, 1024, 256, 128, out_channels], dropout=0.5, norm=None)

    def forward(self, data):
        x, pos, batch = data.x, data.pos, data.batch
        x0 = torch.cat([x, pos], dim=-1)
        x1 = self.conv1(x0, batch)
        x2 = self.conv2(x1, batch)
        x3 = self.conv3(x2, batch)
        out = self.mlp(torch.cat([x1, x2, x3], dim=1))
        return F.log_softmax(out, dim=1)
