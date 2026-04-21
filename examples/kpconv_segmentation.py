from argparse import ArgumentParser, Namespace
from functools import partial
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

import torch_pointcloud.transforms as T
from torch_pointcloud.datasets import ShapeNetPart
from torch_pointcloud.models import KPConvNetSegmentation
from torch_pointcloud.utils.random import seed_everything


def main() -> None:
    args = parse_args()

    print(f"Seeding everything to {args.seed}!")
    seed_everything(args.seed)

    pre_transform = T.NormalizeScaled(keys="pos")
    transform = None

    print(f"Loading {args.dataset} dataset...")
    if args.dataset.lower() == "shapenetpart":
        train_dataset = ShapeNetPart(
            args.root,
            split="train",
            categories=args.categories,
            transform=transform,
            pre_transform=pre_transform,
        )
        test_dataset = ShapeNetPart(
            args.root,
            split="test",
            categories=args.categories,
            transform=transform,
            pre_transform=pre_transform,
        )
    else:
        raise ValueError(f"Unrecognized dataset {args.dataset!r}. Must be 'shapenetpart'.")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
    )

    print("Building model...")
    model = KPConvNetSegmentation(
        in_channels=3,
        num_classes=args.num_classes,
        stem_channels=32,
        stem_type="kpconv",
        encoder_depths=[1, 3, 3, 3],
        encoder_channels=[64, 128, 256, 512],
        encoder_num_neighbors=[20, 35, 40, 40],
        fp_channels=[[256], [128], [64], [32]],
        grid_sizes=[0.08, 0.16, 0.32],
        radii=[0.1, 0.2, 0.4, 0.8],
        kernel_size=15,
        kp_radius=[0.1, 0.2, 0.4, 0.8],
        kp_sigma=[0.05, 0.1, 0.2, 0.4],
        act="leaky_relu",
        norm=partial(torch.nn.BatchNorm1d, momentum=0.05),
    ).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    print("\nStarting training!\n")
    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")
        train_metrics = train_one_epoch(model, optimizer, train_dataloader, args.device)
        val_metrics = eval_one_epoch(model, test_dataloader, args.device)
        metrics = {**train_metrics, **val_metrics}

        print("Scores:", end=" ")
        print(" | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()]))


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--root", type=str, default="data")
    parser.add_argument("--dataset", type=str, default="ShapeNetPart")
    parser.add_argument("--num-classes", type=int, default=50)
    parser.add_argument("--categories", nargs="+", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lr", type=float, default=0.001)
    return parser.parse_args()


def train_one_epoch(
    model: Module,
    optimizer: Optimizer,
    dataloader: DataLoader,
    device: str = "cuda",
    log_interval: int = 5,
) -> Dict[str, float]:
    model.train()

    total_loss = total_correct = total_points = 0.0

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Training")
    for i, data in pbar:
        coords = data["pos"].to(device)
        target = data["label"].to(device)
        batch = data["batch"].to(device)

        optimizer.zero_grad()
        logits = model(None, coords, batch)
        logits = F.log_softmax(logits, dim=1)
        loss = F.nll_loss(logits, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct = logits.argmax(dim=1).eq(target).sum()
        total_correct += correct.item()
        total_points += len(target)

        if (i + 1) % log_interval == 0:
            metrics = {
                "train/loss_step": f"{loss.item():.3f}",
                "train/acc_step": f"{correct.item() / len(target):.3f}",
            }
            pbar.set_postfix(metrics)

    return {
        "train/loss_epoch": total_loss / len(dataloader),
        "train/acc_epoch": total_correct / total_points,
    }


def eval_one_epoch(model: Module, dataloader: DataLoader, device: str = "cuda") -> Dict[str, float]:
    model.eval()

    total_correct = total_points = 0.0
    for data in tqdm(dataloader, total=len(dataloader), desc="Evaluating"):
        coords = data["pos"].to(device)
        target = data["label"].to(device)
        batch = data["batch"].to(device)

        with torch.no_grad():
            logits = model(None, coords, batch)
            preds = logits.argmax(dim=1)

        total_correct += preds.eq(target).sum().item()
        total_points += len(target)

    return {"val/acc": total_correct / total_points}


def collate(data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    batch = torch.cat([torch.ones(len(d["pos"])) * i for i, d in enumerate(data_list)]).long()
    coords = torch.cat([d["pos"] for d in data_list]).float()
    target = torch.cat([d["segmentation"] for d in data_list])

    return {"pos": coords, "label": target, "batch": batch}


if __name__ == "__main__":
    main()
