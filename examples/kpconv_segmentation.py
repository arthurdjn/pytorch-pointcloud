from argparse import ArgumentParser, Namespace
from functools import partial
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

import torch_pointcloud.transforms as T
from torch_pointcloud.config import DATA_DIR
from torch_pointcloud.datasets import S3DIS, ShapeNetPart
from torch_pointcloud.models import KPConvNetSegmentation
from torch_pointcloud.utils.data import DataKeys, collate
from torch_pointcloud.utils.random import seed_everything


def main() -> None:
    args = parse_args()

    print(f"Seeding everything to {args.seed}!")
    seed_everything(args.seed)

    print(f"Loading {args.dataset} dataloaders...", end=" ")
    train_dataloader, test_dataloader = configure_dataloaders(args)
    print("Done!")

    print("Loading model, optimizer, and scheduler...", end=" ")
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
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
        train_metrics = train_one_epoch(model, optimizer, scheduler, train_dataloader, args.device)
        val_metrics = eval_one_epoch(model, test_dataloader, args.num_classes, args.device)
        metrics = {**train_metrics, **val_metrics}

        print("Scores:", end=" ")
        print(" | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()]))


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--root", type=str, default=DATA_DIR)
    parser.add_argument("--dataset", type=str, default="shapenetpart", choices=["shapenetpart", "s3dis"])
    parser.add_argument("--num-classes", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lr", type=float, default=0.006)
    parser.add_argument("--weight-decay", type=float, default=0.05)
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
) -> dict[str, float]:
    model.train()
    total_loss = 0.0

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Training")
    for i, data in pbar:
        pos = data[DataKeys.POS].to(device)
        target = data[DataKeys.SEGMENT].to(device)
        batch = data[DataKeys.BATCH].to(device)

        optimizer.zero_grad()
        logits = model(None, pos, batch)
        logits = F.log_softmax(logits, dim=1)
        loss = F.nll_loss(logits, target)

        loss.backward()
        optimizer.step()

        scheduler.step()
        total_loss += loss.item()

        if (i + 1) % log_interval == 0:
            loss_step = loss.item()
            metrics = {"train/loss_step": f"{loss_step:.3f}"}
            pbar.set_postfix(metrics)

    return {"train/loss_epoch": total_loss / len(dataloader)}


def eval_one_epoch(
    model: Module,
    dataloader: DataLoader,
    num_classes: int,
    device: str = "cuda",
) -> dict[str, float]:
    model.eval()

    val_intersection: Any = []
    val_union: Any = []

    for data in tqdm(dataloader, total=len(dataloader), desc="Evaluating"):
        pos = data[DataKeys.POS].to(device)
        target = data[DataKeys.SEGMENT].to(device)
        batch = data[DataKeys.BATCH].to(device)

        with torch.no_grad():
            logits = model(None, pos, batch)
            preds = logits.argmax(dim=1)

        intersection, union = compute_intersection_union(
            preds,
            target,
            num_classes=num_classes,
            ignore_index=-1,
        )

        val_intersection.append(intersection)
        val_union.append(union)

    val_union = torch.stack(val_union).sum(dim=0)
    val_intersection = torch.stack(val_intersection).sum(dim=0)

    iou_class = val_intersection / (val_union + 1e-10)
    m_iou = iou_class.mean()

    return {"val/mIoU": m_iou}


def configure_dataloaders(args: Namespace) -> tuple[DataLoader, DataLoader]:
    train_dataset: Dataset
    test_dataset: Dataset
    transform: Callable

    if args.dataset.lower() == "shapenetpart":
        transform = T.NormalizeScaled(keys=DataKeys.POS)
        train_dataset = ShapeNetPart(
            args.root,
            split="train",
            transform=transform,
            num_workers=args.num_workers,
        )
        test_dataset = ShapeNetPart(
            args.root,
            split="test",
            transform=transform,
            num_workers=args.num_workers,
        )
    elif args.dataset.lower() == "s3dis":
        transform = T.Compose(
            [
                T.NormalizeScaled(keys=DataKeys.POS),
                T.RandomSampled(
                    keys=[DataKeys.POS, DataKeys.COLOR, DataKeys.SEGMENT, DataKeys.INSTANCE],
                    num_samples=4096,
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
