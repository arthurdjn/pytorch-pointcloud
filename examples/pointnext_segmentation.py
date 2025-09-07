from argparse import ArgumentParser, Namespace
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import torch_pointcloud.transforms as T
from torch_pointcloud.datasets import S3DIS, ShapeNetPart
from torch_pointcloud.models import create_model
from torch_pointcloud.utils.random import seed_everything


def main() -> None:
    args = parse_args()

    print(f"Seeding everything to {args.seed}!")
    seed_everything(args.seed)

    pre_transform = T.NormalizeScaled(keys="coords")
    transform = None

    print(f"Loading {args.dataset} dataset...")
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
    elif args.dataset.lower() == "s3dis":
        train_dataset = S3DIS(
            args.root,
            areas=["Area_1", "Area_2", "Area_3", "Area_4", "Area_6"],
            transform=transform,
            pre_transform=pre_transform,
        )
        test_dataset = S3DIS(
            args.root,
            areas=["Area_5"],
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
    model = create_model(
        args.model,
        in_channels=3,
        num_classes=args.num_classes,
        task="segmentation",
    ).to(args.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        pct_start=0.05,
        anneal_strategy="cos",
        div_factor=10.0,
        final_div_factor=1000.0,
        total_steps=len(train_loader) * args.epochs,
    )

    print("\nStarting training!\n")
    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")
        train_metrics = train_one_epoch(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            loader=train_loader,
            device=args.device,
        )
        val_metrics = eval_one_epoch(
            model=model,
            loader=test_loader,
            num_classes=args.num_classes,
            device=args.device,
        )
        metrics = {**train_metrics, **val_metrics}

        print("Scores:", end=" ")
        print(" | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()]))


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--root", type=str, default="data")
    parser.add_argument("--dataset", type=str, default="ShapeNetPart", choices=["ShapeNetPart", "S3DIS"])
    parser.add_argument(
        "--model",
        type=str,
        default="pointnext-base",
        choices=["pointnext-base", "pointnext-sm", "pointnext-lg", "pointnext-xl"],
    )
    parser.add_argument("--num-classes", type=int, default=50)
    parser.add_argument("--categories", nargs="+", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lr", type=float, default=0.006)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    return parser.parse_args()


def train_one_epoch(
    model: Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    loader: DataLoader,
    device: str = "cuda",
    log_interval: int = 5,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0

    pbar = tqdm(enumerate(loader), total=len(loader), desc="Training")
    for i, data in pbar:
        coords = data["coords"].to(device)
        # colors = data["colors"].to(device)
        target = data["target"].to(device)
        batch = data["batch"].to(device)

        optimizer.zero_grad()
        logits = model(None, coords, batch)
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

    return {"train/loss_epoch": total_loss / len(loader)}


def eval_one_epoch(
    model: Module,
    loader: DataLoader,
    num_classes: int,
    device: str = "cuda",
) -> Dict[str, float]:
    model.eval()

    val_intersection: Any = []
    val_union: Any = []

    for data in tqdm(loader, total=len(loader), desc="Evaluating"):
        coords = data["coords"].to(device)
        # colors = data["colors"].to(device)
        target = data["target"].to(device)
        batch = data["batch"].to(device)

        with torch.no_grad():
            logits = model(None, coords, batch)
            preds = logits.argmax(dim=1)

        intersection, union = compute_intersection_union_stats(
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


def collate(data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    batch = torch.cat([torch.ones(len(d["coords"])) * i for i, d in enumerate(data_list)]).long()
    coords = torch.cat([d["coords"] for d in data_list]).float()
    target = torch.cat([d["segmentation"] if "segmentation" in d else d["semantic"] for d in data_list])
    # colors = torch.cat([d["colors"] for d in data_list]).int() / 255.0

    return {"coords": coords, "target": target, "batch": batch}


def compute_intersection_union_stats(
    preds: Tensor,
    target: Tensor,
    num_classes: int,
    ignore_index: int = -1,
) -> Tuple[Tensor, Tensor]:
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
