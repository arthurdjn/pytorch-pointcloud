from argparse import ArgumentParser, Namespace
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

import torch_pointcloud.transforms as T
from torch_pointcloud.datasets import ShapeNetPart
from torch_pointcloud.models import SPVCNNSegmentation
from torch_pointcloud.utils.random import seed_everything


def main() -> None:
    args = parse_args()

    print(f"Seeding everything to {args.seed}!")
    seed_everything(args.seed)

    pre_transform = T.NormalizeScaled(keys="coords")
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

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
    )

    print("Building model...")
    model = SPVCNNSegmentation(
        in_channels=3,
        num_classes=args.num_classes,
        encoder_channels=(32, 64, 128, 256),
        encoder_depths=(2, 2, 2, 2),
        decoder_channels=(256, 128, 96, 96),
        decoder_depths=(2, 2, 2, 2),
        stem_channels=32,
    ).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    print("\nStarting training!\n")
    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")
        train_metrics = train_one_epoch(model, optimizer, train_loader, args.device)
        val_metrics = eval_one_epoch(model, test_loader, args.device)
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
    loader: DataLoader,
    device: str = "cuda",
    log_interval: int = 5,
) -> Dict[str, float]:
    model.train()

    total_loss = total_correct = total_points = 0.0

    pbar = tqdm(enumerate(loader), total=len(loader), desc="Training")
    for i, data in pbar:
        coords = data["coords"].to(device)
        target = data["target"].to(device)
        batch = data["batch"].to(device)

        grid_size = 0.01
        grid_coords = torch.div(coords - coords.min(0)[0], grid_size, rounding_mode="trunc").int()

        optimizer.zero_grad()
        logits = model(None, grid_coords, batch)
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
        "train/loss_epoch": total_loss / len(loader),
        "train/acc_epoch": total_correct / total_points,
    }


def eval_one_epoch(model: Module, loader: DataLoader, device: str = "cuda") -> Dict[str, float]:
    model.eval()

    total_correct = total_points = 0.0
    for data in tqdm(loader, total=len(loader), desc="Evaluating"):
        coords = data["coords"].to(device)
        target = data["target"].to(device)
        batch = data["batch"].to(device)

        grid_size = 0.01
        grid_coords = torch.div(coords - coords.min(0)[0], grid_size, rounding_mode="trunc").int()

        with torch.no_grad():
            logits = model(None, grid_coords, batch)
            preds = logits.argmax(dim=1)

        total_correct += preds.eq(target).sum().item()
        total_points += len(target)

    return {"val/acc": total_correct / total_points}


def collate(data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    batch = torch.cat([torch.ones(len(d["coords"])) * i for i, d in enumerate(data_list)]).long()
    coords = torch.cat([d["coords"] for d in data_list]).float()
    target = torch.cat([d["segmentation"] for d in data_list])

    return {"coords": coords, "target": target, "batch": batch}


if __name__ == "__main__":
    main()
