from argparse import ArgumentParser, Namespace
from typing import TYPE_CHECKING, Dict

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

import torch_pointcloud.transforms as T
from torch_pointcloud import create_model
from torch_pointcloud.config import DATA_DIR
from torch_pointcloud.datasets import S3DIS, ShapeNetPart
from torch_pointcloud.utils.data import DataKeys, collate
from torch_pointcloud.utils.imports import optional_import
from torch_pointcloud.utils.random import seed_everything

if TYPE_CHECKING:
    from ocnn.octree import Octree, Points

Octree, _ = optional_import("ocnn.octree", "Octree")
Points, _ = optional_import("ocnn.octree", "Points")


def main() -> None:
    args = parse_args()

    print(f"Seeding everything to {args.seed}!")
    seed_everything(args.seed)

    print(f"Loading {args.dataset} dataloaders...", end=" ")
    train_dataloader, test_dataloader = configure_dataloaders(args)
    print("Done!")

    print("Loading model, optimizer, and scheduler...", end=" ")
    model = create_model(
        name=args.model,
        in_channels=4,
        num_classes=args.num_classes,
        task="segmentation",
    ).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        pct_start=0.05,
        anneal_strategy="cos",
        div_factor=10.0,
        final_div_factor=1000.0,
        total_steps=len(train_dataloader) * args.epochs,
    )
    print("Done!")

    print("\nStarting training!\n")
    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")
        train_metrics = train_one_epoch(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            dataloader=train_dataloader,
            device=args.device,
        )
        val_metrics = eval_one_epoch(
            model=model,
            dataloader=test_dataloader,
            num_classes=args.num_classes,
            device=args.device,
        )
        metrics = {**train_metrics, **val_metrics}

        print("Scores:", end=" ")
        print(" | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()]))


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--root", type=str, default=DATA_DIR)
    parser.add_argument("--dataset", type=str, default="shapenetpart", choices=["shapenetpart", "s3dis"])
    parser.add_argument("--model", type=str, default="octformer-base.sm", choices=["octformer-base.sm"])
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
    scheduler: LRScheduler,
    dataloader: DataLoader,
    device: str = "cuda",
    log_interval: int = 5,
) -> Dict[str, float]:
    model.train()

    total_loss = total_correct = total_points = 0.0

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Training")
    for i, data in pbar:
        points = data[DataKeys.POINTS].to(device)
        octree = data[DataKeys.OCTREE].to(device)
        target = data[DataKeys.SEGMENT].to(device)
        x = octree.get_input_feature("ND", nempty=True)

        optimizer.zero_grad()
        logits = model(x, octree, points.points, points.batch_id.squeeze())
        logits = F.log_softmax(logits, dim=1)
        loss = F.nll_loss(logits, target)
        loss.backward()
        optimizer.step()

        scheduler.step()
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


def eval_one_epoch(model: Module, dataloader: DataLoader, num_classes: int, device: str = "cuda") -> Dict[str, float]:
    model.eval()

    total_correct = total_points = 0.0
    for data in tqdm(dataloader, total=len(dataloader), desc="Evaluating"):
        octree = data[DataKeys.OCTREE].to(device)
        points = data[DataKeys.POINTS].to(device)
        target = data[DataKeys.SEGMENT].to(device)
        x = octree.get_input_feature("ND", nempty=True)

        with torch.no_grad():
            logits = model(x, octree, points.points, points.batch_id.squeeze())
            preds = logits.argmax(dim=1).detach()

        total_correct += preds.eq(target).sum().item()
        total_points += len(target)

    return {"val/acc": total_correct / total_points}


def configure_dataloaders(args: Namespace) -> tuple[DataLoader, DataLoader]:
    train_dataset: Dataset
    test_dataset: Dataset

    if args.dataset.lower() == "shapenetpart":
        transform = T.Compose(
            [
                T.Shift(keys=DataKeys.POS, method="bbox"),
                T.Divide(keys=DataKeys.POS, divisor=10.24),
                T.AlignAxis(keys=DataKeys.POS, dim=-1),
                T.BuildOctree(
                    pos_key=DataKeys.POS,
                    normal_key=DataKeys.NORMAL,
                    label_key=DataKeys.SEGMENT,
                    points_key=DataKeys.POINTS,
                    octree_key=DataKeys.OCTREE,
                    depth=11,
                    full_depth=2,
                    batch_size=1,
                ),
            ]
        )

        train_dataset = ShapeNetPart(
            args.root,
            split="train",
            categories=args.categories,
            transform=transform,
        )
        test_dataset = ShapeNetPart(
            args.root,
            split="test",
            categories=args.categories,
            transform=transform,
        )
    elif args.dataset.lower() == "s3dis":
        transform = T.Compose(
            [
                T.Shift(keys=DataKeys.POS, method="bbox"),
                T.Rescale(keys=DataKeys.POS, method="bbox"),
                T.Divide(keys=DataKeys.POS, divisor=10.24),
                T.AlignAxis(keys=DataKeys.POS, dim=-1),
                T.BuildOctree(
                    pos_key=DataKeys.POS,
                    normal_key=DataKeys.NORMAL,
                    label_key=DataKeys.SEGMENT,
                    points_key=DataKeys.POINTS,
                    octree_key=DataKeys.OCTREE,
                    depth=11,
                    full_depth=2,
                    batch_size=1,
                ),
            ]
        )

        train_dataset = S3DIS(
            args.root,
            areas=["Area_1", "Area_2", "Area_3", "Area_4", "Area_6"],
            transform=transform,
        )
        test_dataset = S3DIS(
            args.root,
            areas=["Area_5"],
            transform=transform,
        )
    else:
        raise ValueError(f"Unrecognized dataset {args.dataset!r}. Must be 'shapenetpart'.")

    # Limit the size of the dataset if specified
    if args.limit_train_batches is not None:
        n = min(args.limit_train_batches * args.batch_size, len(train_dataset))
        train_dataset = Subset(train_dataset, range(args.limit_train_batches * args.batch_size))
    if args.limit_test_batches is not None:
        n = min(args.limit_test_batches * args.batch_size, len(test_dataset))
        test_dataset = Subset(test_dataset, range(n))

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

    return train_dataloader, test_dataloader


def compute_intersection_union(
    preds: Tensor,
    target: Tensor,
    num_classes: int,
    ignore_index: int = -1,
) -> tuple[Tensor, Tensor]:
    valid_mask = target != ignore_index
    preds = preds[valid_mask]
    target = target[valid_mask]

    confusion_matrix = torch.zeros(num_classes, num_classes, device=preds.device)
    indices = num_classes * target + preds
    confusion_matrix = confusion_matrix.view(-1)
    confusion_matrix.index_add_(0, indices, torch.ones_like(indices, dtype=torch.float))
    confusion_matrix = confusion_matrix.view(num_classes, num_classes)

    intersection = torch.diag(confusion_matrix)
    union = confusion_matrix.sum(dim=0) + confusion_matrix.sum(dim=1) - intersection

    return intersection, union


if __name__ == "__main__":
    main()
