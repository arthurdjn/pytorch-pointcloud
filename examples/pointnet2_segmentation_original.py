from argparse import ArgumentParser
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch_points_kernels as tp
from torch.nn import (
    BatchNorm1d,
    BatchNorm2d,
    Conv1d,
    Conv2d,
    Dropout,
    LeakyReLU,
    Module,
    ModuleList,
    NLLLoss,
    Sequential,
)
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from torch_pointcloud.datasets import S3DIS
from torch_pointcloud.utils.utils import set_seed


def grouping_operation(features, idx):
    all_idx = idx.reshape(idx.shape[0], -1)
    all_idx = all_idx.unsqueeze(1).repeat(1, features.shape[1], 1)
    grouped_features = features.gather(2, all_idx)
    return grouped_features.reshape(idx.shape[0], features.shape[1], idx.shape[1], idx.shape[2])


class SharedMLP(Module):
    def __init__(self, channel_list: List[int], bias: bool = False, bn: bool = True, activation: Module | None = None):
        super().__init__()
        if len(channel_list) < 2:
            raise ValueError(f"The SharedMLP must have at least 2 channels. Got {len(channel_list)}.")
        activation = activation or LeakyReLU(negative_slope=0.01)

        self.layers = ModuleList()
        for i in range(len(channel_list) - 1):
            in_channels = channel_list[i]
            out_channels = channel_list[i + 1]
            block = []
            block.append(Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), bias=bias))
            if bn:
                block.append(BatchNorm2d(out_channels))
            block.append(activation)
            self.layers.append(Sequential(*block))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class SAModuleMSG(Module):
    def __init__(
        self,
        num_points: int,
        radius_list: List[int],
        samples_list: List[int],
        mlps: List[List[int]],
        use_pos: bool = True,
        normalize_pos: bool = False,
    ):
        super().__init__()
        if not (len(radius_list) == len(samples_list) == len(mlps)):
            raise ValueError(
                f"Invalid arguments. Expected len(radiuses) == len(samples) == len(mlps), but got {len(radius_list)=}, {len(samples_list)=}, {len(mlps)=}"
            )

        self.num_points = num_points
        self.radius_list = radius_list
        self.samples_list = samples_list
        self.use_pos = use_pos
        self.normalize_pos = normalize_pos

        self.mlps = ModuleList()
        for mlp_layer in mlps:
            self.mlps.append(SharedMLP(mlp_layer, bias=False))

    def forward(
        self, pos: torch.Tensor, features: Optional[torch.Tensor] = None, indices: Optional[torch.Tensor] = None
    ):
        B, N, d = pos.shape
        indices = indices if indices is not None else tp.furthest_point_sample(pos, self.num_points)
        indices = indices.unsqueeze(-1).repeat(1, 1, d).long()
        new_pos = pos.gather(1, indices)

        ms_x = []
        for r, num_samples, mlp in zip(self.radius_list, self.samples_list, self.mlps):
            # pos: (B, N, 3), new_pos: (B, Np, 3)
            radius_idx = tp.ball_query(r, num_samples, pos, new_pos)[0]
            pos_t = pos.transpose(1, 2).contiguous()  # (B, 3, N)
            grouped_pos = grouping_operation(pos_t, radius_idx)  # (B, 3, npoint, nsample)
            grouped_pos -= new_pos.transpose(1, 2).unsqueeze(-1)

            if self.normalize_pos:
                grouped_pos /= r

            if features is not None:
                # Select the features of the sampled points
                grouped_features = grouping_operation(features, radius_idx)  # (B, C, npoint, nsample)
                new_x = torch.cat([grouped_pos, grouped_features], dim=1) if self.use_pos else grouped_features
                # new_x: (B, 3 + C, npoint, nsample)
            else:
                assert self.use_pos, "Cannot have not features and not use xyz as a feature!"
                new_x = grouped_pos

            new_x = mlp(new_x)  # (B, mlp[-1], npoint, K)
            new_x = F.max_pool2d(new_x, kernel_size=[1, new_x.size(3)])  # (B, mlp[-1], npoint, 1)
            new_x = new_x.squeeze(-1)  # (B, mlp[-1], npoint)
            ms_x.append(new_x)

        new_x = torch.cat(ms_x, 1)
        # new_x: (B, sum(mlp[-1]), npoint)
        # print(f"{new_x.shape=}, {new_pos.shape=}, {indices.shape=}")
        return new_x, new_pos, indices


class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn, mode="max", bn=True):
        super().__init__()
        if mode not in ["mean", "max"]:
            raise ValueError(f"Unrecognized mode {mode!r} for the GlobalDenseBaseModule. " "Must be 'mean' or 'max'.")

        self.mode = mode
        self.nn = SharedMLP(nn, bn=bn, bias=False)

    def forward(self, x, pos):
        pos_flipped = pos.transpose(1, 2).contiguous()
        x = self.nn(torch.cat([x, pos_flipped], dim=1).unsqueeze(-1))

        if self.mode == "max":
            x = x.squeeze(-1).max(-1)[0]
        elif self.mode == "mean":
            x = x.squeeze(-1).mean(-1)
        else:
            raise ValueError(
                f"Unrecognized mode {self.mode!r} for the GlobalDenseBaseModule. " "Must be 'mean' or 'max'."
            )

        pos = pos.new_zeros((pos.size(0), 1, pos.size(2)))
        x = x.unsqueeze(-1)
        return x, pos


class DenseFPModule(Module):
    def __init__(self, channel_list, bn=True):
        super().__init__()
        self.nn = SharedMLP(channel_list, bn=bn, bias=False)

    def conv(self, pos, pos_skip, x):
        assert pos_skip.shape[2] == 3

        if pos is not None:
            dist, idx = tp.three_nn(pos_skip, pos)
            dist_recip = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_feats = tp.three_interpolate(x, idx, weight)
        else:
            interpolated_feats = x.expand(*(x.size()[0:2] + (pos_skip.size(1),)))

        return interpolated_feats

    def forward(self, x, pos, x_skip, pos_skip):
        new_features = self.conv(pos, pos_skip, x)

        if x_skip is not None:
            new_features = torch.cat([new_features, x_skip], dim=1)  # (B, C2 + C1, n)
        new_features = new_features.unsqueeze(-1)

        if hasattr(self, "nn"):
            new_features = self.nn(new_features)

        return new_features.squeeze(-1), pos_skip


class SegmentationNet(Module):
    def __init__(self, num_classes):
        super().__init__()

        self.sa1 = SAModuleMSG(
            num_points=1024,
            radius_list=[0.05, 0.1],
            samples_list=[16, 32],
            mlps=[[3 + 3, 16, 16, 32], [3 + 3, 32, 32, 64]],
        )
        self.sa2 = SAModuleMSG(
            num_points=256,
            radius_list=[0.1, 0.2],
            samples_list=[16, 32],
            mlps=[[32 + 64 + 3, 64, 64, 128], [32 + 64 + 3, 64, 96, 128]],
        )
        self.sa3 = SAModuleMSG(
            num_points=64,
            radius_list=[0.2, 0.4],
            samples_list=[16, 32],
            mlps=[[128 + 128 + 3, 128, 196, 256], [128 + 128 + 3, 128, 196, 256]],
        )
        self.sa4 = SAModuleMSG(
            num_points=16,
            radius_list=[0.4, 0.8],
            samples_list=[16, 32],
            mlps=[[256 + 256 + 3, 256, 256, 512], [256 + 256 + 3, 256, 384, 512]],
        )
        self.sag = GlobalSAModule([512 + 512 + 3, 512, 1024])

        self.fp4 = DenseFPModule([512 + 512 + 256 + 256, 256, 256])
        self.fp3 = DenseFPModule([128 + 128 + 256, 256, 256])
        self.fp2 = DenseFPModule([32 + 64 + 256, 256, 128])
        self.fp1 = DenseFPModule([128, 128, 128])

        self.conv1 = Conv1d(128, 128, 1)
        self.bn1 = BatchNorm1d(128)
        self.drop1 = Dropout(0.5)
        self.conv2 = Conv1d(128, num_classes, 1)

    def forward(self, xyz):
        l0_points = xyz[:, 3:6, :].contiguous()
        l0_xyz = xyz[:, :3, :].transpose(1, 2).contiguous()
        l1_points, l1_xyz, _ = self.sa1(l0_xyz, l0_points)
        l2_points, l2_xyz, _ = self.sa2(l1_xyz, l1_points)
        l3_points, l3_xyz, _ = self.sa3(l2_xyz, l2_points)
        l4_points, l4_xyz, _ = self.sa4(l3_xyz, l3_points)
        l3_points, _ = self.fp4(l4_points, l4_xyz, l3_points, l3_xyz)
        l2_points, _ = self.fp3(l3_points, l3_xyz, l2_points, l2_xyz)
        l1_points, _ = self.fp2(l2_points, l2_xyz, l1_points, l1_xyz)
        l0_points, _ = self.fp1(l1_points, l1_xyz, None, l0_xyz)

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x, l4_points


def get_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument(
        "--root", type=str, default="/home/arthur/Documents/Code/Github/pytorch-point3d-models/data/s3dis/processed"
    )
    parser.add_argument("--dataset", type=str, default="S3DIS")
    parser.add_argument("--num_classes", type=int, default=13)
    parser.add_argument("--num_points", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lr_decay", type=float, default=0.7)
    parser.add_argument("--lr_decay_step", type=int, default=10)
    parser.add_argument("--lr_clip", type=int, default=1e-5)
    parser.add_argument("--momentum", type=float, default=0.1)
    parser.add_argument("--momentum_decay", type=float, default=0.5)
    parser.add_argument("--momentum_decay_step", type=float, default=10)
    return parser


def train_one_epoch(
    model: Module,
    optimizer: Optimizer,
    criterion: Module,
    loader: DataLoader,
    device: str = "cuda",
    log_interval: int = 5,
) -> Dict[str, float]:
    model.train()

    total_correct = 0
    total_seen = 0
    loss_sum = 0

    pbar = tqdm(enumerate(loader), total=len(loader), desc="Training")
    for i, (points, target) in pbar:
        optimizer.zero_grad()

        pos = torch.tensor(points[:, :, :3], dtype=torch.float32).to(device)
        feats = torch.tensor(points[:, :, 3:6], dtype=torch.float32).to(device).transpose(1, 2)
        target = torch.tensor(target, dtype=torch.long).to(device)

        points = points.float().cuda().transpose(2, 1)
        seg_pred, _ = model(points)

        # seg_pred, _ = model(pos, feats)
        # seg_pred = seg_pred.transpose(1, 2).contiguous()
        B, N, C = seg_pred.size()
        seg_pred = seg_pred.contiguous().view(-1, C)
        target = target.view(-1, 1)[:, 0]
        loss = criterion(seg_pred, target)
        loss.backward()
        optimizer.step()

        pred_choice = seg_pred.cpu().data.max(1)[1].numpy()
        correct = np.sum(pred_choice == target.cpu().numpy())
        total_correct += correct
        total_seen += B * N
        loss_sum += loss

        if i % log_interval == 0:
            pbar.set_postfix({"train/loss_step": loss.item(), "train/acc_step": float(correct / (B * N))})

    return {"train/loss_epoch": loss_sum / len(loader), "train/acc": float(total_correct / float(total_seen))}


def main() -> None:
    parser = get_parser()
    args = parser.parse_args()
    set_seed(42)

    if args.dataset == "S3DIS":
        train_dataset = S3DIS(
            split="train",
            data_root=args.root,
            num_point=args.num_points,
            test_area=5,
            block_size=1.0,
            sample_rate=1.0,
            transform=None,
        )
    else:
        raise ValueError(f"Unrecognized dataset {args.dataset!r}. Must be 'S3DIS'.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = SegmentationNet(args.num_classes).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    weight = torch.Tensor(train_dataset.labelweights).to(args.device)
    criterion = NLLLoss(weight=weight)

    for epoch in range(args.epochs):
        lr = max(args.lr * (args.lr_decay ** (epoch // args.lr_decay_step)), args.lr_clip)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        momentum = args.momentum * (args.momentum_decay ** (epoch // args.momentum_decay_step))
        if momentum < 0.01:
            momentum = 0.01

        print(f"Epoch {epoch + 1}/{args.epochs}")
        metrics = {}
        train_metrics = train_one_epoch(model, optimizer, criterion, train_loader, args.device)
        metrics.update(train_metrics)

        print("Scores:", end=" ")
        print(" | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()]))


if __name__ == "__main__":
    main()
