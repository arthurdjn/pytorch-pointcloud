from argparse import ArgumentParser
from typing import Dict

import numpy as np
import torch
from torch.nn import Module, NLLLoss
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from torch_pointcloud.datasets import S3DIS
from torch_pointcloud.models.pointnet2 import PointNetSegmentation
from torch_pointcloud.utils.utils import set_seed


def get_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--root", type=str, default="data")
    parser.add_argument("--dataset", type=str, default="S3DIS")
    parser.add_argument("--num_classes", type=int, default=13)
    parser.add_argument("--num_points", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lr", type=float, default=0.001)
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
    for i, batch in pbar:
        optimizer.zero_grad()

        pos = torch.tensor(batch["xyz_shifted"], dtype=torch.float32).to(device)
        feats = torch.tensor(batch["rgb"], dtype=torch.float32).to(device).transpose(1, 2)
        target = torch.tensor(batch["semantic"], dtype=torch.long).to(device)

        seg_pred, _ = model(pos, feats)
        seg_pred = seg_pred.transpose(1, 2).contiguous()
        B, N, C = seg_pred.size()
        seg_pred = seg_pred.view(-1, C)
        batch_label = target.view(-1, 1)[:, 0].cpu().data.numpy()
        target = target.view(-1, 1)[:, 0]
        loss = criterion(seg_pred, target)
        loss.backward()
        optimizer.step()

        pred_choice = seg_pred.cpu().data.max(1)[1].numpy()
        correct = np.sum(pred_choice == batch_label)
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

    print("Loading datasets...")
    if args.dataset == "S3DIS":
        train_dataset = S3DIS(root=args.root)
    else:
        raise ValueError(f"Unrecognized dataset {args.dataset!r}. Must be 'S3DIS'.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print("Building model...")
    model = PointNetSegmentation(args.num_classes).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Compute weights for each class
    print("Computing class weights...")
    weight = torch.zeros(args.num_classes)
    for data in train_dataset:
        target = data["semantic"]
        count = torch.bincount(target, minlength=args.num_classes).float()
        weight += count

    weight = torch.pow(torch.max(weight) / weight, 1 / 3.0)
    print("Class weights:", weight)

    criterion = NLLLoss(weight=torch.tensor(weight).to(args.device))

    print("Starting training!")
    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")
        metrics = {}
        train_metrics = train_one_epoch(model, optimizer, criterion, train_loader, args.device)
        metrics.update(train_metrics)

        print("Scores:", end=" ")
        print(" | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()]))


if __name__ == "__main__":
    main()
