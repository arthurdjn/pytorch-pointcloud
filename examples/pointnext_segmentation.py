from argparse import ArgumentParser, Namespace
from typing import Callable

import torch
import torch.nn.functional as F
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

import torch_pointcloud.transforms as T
from torch_pointcloud.config import DATA_DIR
from torch_pointcloud.datasets import S3DIS, ShapeNetPart
from torch_pointcloud.metrics import confusion_matrix, intersection_over_union
from torch_pointcloud.models import create_model
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
    model = create_model(
        args.model,
        in_channels=3,
        num_classes=args.num_classes,
        task="semantic-segmentation",
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
    parser.add_argument(
        "--model",
        type=str,
        default="pointnext-base",
        choices=["pointnext-base", "pointnext-sm", "pointnext-lg", "pointnext-xl"],
    )
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--categories", nargs="+", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lr", type=float, default=0.006)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--limit-train-batches", type=int, default=None)
    parser.add_argument("--limit-test-batches", type=int, default=None)
    args = parser.parse_args()
    if args.num_classes is None:
        args.num_classes = 50 if args.dataset == "shapenetpart" else 13
    return args


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

    cm = torch.zeros(num_classes, num_classes, dtype=torch.long, device=device)

    for data in tqdm(dataloader, total=len(dataloader), desc="Evaluating"):
        pos = data[DataKeys.POS].to(device)
        target = data[DataKeys.SEGMENT].to(device)
        batch = data[DataKeys.BATCH].to(device)

        with torch.no_grad():
            logits = model(None, pos, batch)
            preds = logits.argmax(dim=1)

        cm += confusion_matrix(preds, target, num_classes, ignore_index=-1)

    return {"val/mIoU": intersection_over_union(cm)}


def configure_dataloaders(args: Namespace) -> tuple[DataLoader, DataLoader]:
    train_dataset: Dataset
    test_dataset: Dataset
    transform: Callable

    if args.dataset.lower() == "shapenetpart":
        transform = T.Rescale(keys=DataKeys.POS)
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
                T.Rescale(keys=DataKeys.POS),
                T.RandomSample(
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
        train_dataset = Subset(train_dataset, range(n))
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


if __name__ == "__main__":
    main()
