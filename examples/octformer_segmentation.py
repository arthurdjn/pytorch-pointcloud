from argparse import ArgumentParser, Namespace
from typing import TYPE_CHECKING, Any, Dict, List

import torch
import torch.nn.functional as F
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

import torch_pointcloud.transforms as T
from torch_pointcloud import create_model
from torch_pointcloud.datasets import ShapeNetPart
from torch_pointcloud.utils.imports import optional_import
from torch_pointcloud.utils.random import seed_everything

if TYPE_CHECKING:
    from ocnn.octree import Octree, Points

Octree, _ = optional_import("ocnn.octree", "Octree")
Points, _ = optional_import("ocnn.octree", "Points")


OCTREE_SCALE_FACTOR = 10.24
OCTREE_DEPTH = 11
OCTREE_FULL_DEPTH = 2


def main() -> None:
    args = parse_args()
    seed_everything(42)

    pre_transform = T.NormalizeScaled(keys="pos")
    transform = None
    train_dataset: Dataset
    test_dataset: Dataset
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

    # Limit the size of the datasets, if specified
    if args.limit_train_batches is not None:
        train_dataset = Subset(train_dataset, range(args.limit_train_batches * args.batch_size))
    if args.limit_test_batches is not None:
        test_dataset = Subset(test_dataset, range(args.limit_test_batches * args.batch_size))

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

    model = create_model(
        name="octformer-base",
        num_classes=args.num_classes,
        in_channels=3,
        task="segmentation",
    ).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")
        train_metrics = train_one_epoch(model, optimizer, train_dataloader, args.device)
        val_metrics = eval_one_epoch(model, test_dataloader, args.device)
        metrics = {**train_metrics, **val_metrics}

        print("Scores:", end=" ")
        print(" | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()]))


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--root", type=str, default="data")
    parser.add_argument("--dataset", type=str, default="ShapeNetPart")
    parser.add_argument("--num-classes", type=int, default=50)
    parser.add_argument("--categories", nargs="+", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--limit-train-batches", type=int, default=None)
    parser.add_argument("--limit-test-batches", type=int, default=None)
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
        batch_size = batch.max().item() + 1

        points = Points(
            points=coords / OCTREE_SCALE_FACTOR,
            # normal=normal,
            features=coords,
            batch_id=batch.unsqueeze(-1),
            batch_size=batch_size,
        )
        octree = Octree(
            depth=OCTREE_DEPTH,
            full_depth=OCTREE_FULL_DEPTH,
            batch_size=batch_size,
            device=coords.device,
        )
        octree.build_octree(points)
        octree.construct_all_neigh()

        optimizer.zero_grad()
        logits = model(None, octree, coords, batch)
        logits = F.log_softmax(logits, dim=1)
        loss = F.nll_loss(logits, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct = logits.argmax(dim=1).eq(target).sum()
        total_correct += correct.item()
        total_points += len(target)

        if i % log_interval == 0:
            pbar.set_postfix({"train/loss_step": loss.item(), "train/acc_step": correct.item() / len(target)})

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
        batch_size = batch.max().item() + 1

        points = Points(
            points=coords / OCTREE_SCALE_FACTOR,
            # normal=normal,
            # features=features,
            batch_id=batch.unsqueeze(-1),
            batch_size=batch_size,
        )
        octree = Octree(
            depth=OCTREE_DEPTH,
            full_depth=OCTREE_FULL_DEPTH,
            batch_size=batch_size,
            device=coords.device,
        )
        octree.build_octree(points)
        octree.construct_all_neigh()

        with torch.no_grad():
            logits = model(None, octree, coords, batch)
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
