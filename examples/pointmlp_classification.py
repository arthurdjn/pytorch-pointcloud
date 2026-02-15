from argparse import ArgumentParser, Namespace
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

import torch_pointcloud.transforms as T
from torch_pointcloud.datasets import ModelNet10, ModelNet40
from torch_pointcloud.models import create_model
from torch_pointcloud.utils.random import seed_everything


def main() -> None:
    args = parse_args()

    print(f"Seeding everything to {args.seed}!")
    seed_everything(args.seed)

    pre_transform = T.NormalizeScaled(keys="coords")
    transform = T.Compose(
        [
            T.RandomSampleFaceVerticesd(
                keys="coords",
                face_key="faces",
                normal_key="normals",
                num_samples=args.num_points,
            )
        ]
    )

    train_dataset: Dataset
    test_dataset: Dataset
    if args.dataset.lower() == "modelnet10":
        train_dataset = ModelNet10(
            args.root,
            True,
            transform=transform,
            pre_transform=pre_transform,
            download=True,
            num_workers=args.num_workers,
        )
        test_dataset = ModelNet10(
            args.root,
            False,
            transform=transform,
            pre_transform=pre_transform,
            download=True,
            num_workers=args.num_workers,
        )
    elif args.dataset.lower() == "modelnet40":
        train_dataset = ModelNet40(
            args.root,
            True,
            transform=transform,
            pre_transform=pre_transform,
            download=True,
            num_workers=args.num_workers,
        )
        test_dataset = ModelNet40(
            args.root,
            False,
            transform=transform,
            pre_transform=pre_transform,
            download=True,
            num_workers=args.num_workers,
        )
    else:
        raise ValueError(f"Unrecognized dataset {args.dataset!r}. Must be 'ModelNet10' or 'ModelNet40'.")

    # Limit the size of the dataset if specified
    if args.limit_train_batches is not None:
        train_dataset = Subset(train_dataset, range(args.limit_train_batches * args.batch_size))
    if args.limit_test_batches is not None:
        test_dataset = Subset(test_dataset, range(args.limit_test_batches * args.batch_size))

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

    model = create_model(
        args.model,
        in_channels=6,
        num_classes=args.num_classes,
        task="classification",
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

    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")
        train_metrics = train_one_epoch(model, optimizer, train_loader, args.device)
        val_metrics = eval_one_epoch(model, test_loader, args.device)
        metrics = {**train_metrics, **val_metrics}
        scheduler.step()

        print("Scores:", end=" ")
        print(" | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()]))


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--root", type=str, default="data")
    parser.add_argument("--dataset", type=str, default="modelnet10", choices=["modelnet10", "modelnet40"])
    parser.add_argument("--model", type=str, default="pointmlp-base", choices=["pointmlp-base", "pointmlp-elite"])
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--limit-train-batches", type=int, default=None)
    parser.add_argument("--limit-test-batches", type=int, default=None)
    return parser.parse_args()


def train_one_epoch(
    model: Module,
    optimizer: Optimizer,
    loader: DataLoader,
    device: str = "cuda",
    log_interval: int = 5,
) -> Dict[str, float]:
    total_correct = total_loss = 0.0
    model.train()

    pbar = tqdm(enumerate(loader), total=len(loader), desc="Training")
    for i, data in pbar:
        coords = data["coords"].to(device)
        normals = data["normals"].to(device)
        target = data["target"].to(device)
        batch = data["batch"].to(device)
        features = torch.cat([coords, normals], dim=1)

        optimizer.zero_grad()
        logits = model(features, coords, batch)
        probs = F.log_softmax(logits, dim=1)

        loss = F.nll_loss(probs, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        correct = logits.argmax(dim=1).eq(target).sum()
        total_correct += correct.item()

        if i % log_interval == 0:
            loss_step = loss.item()
            acc_step = correct.item() / len(target)
            pbar.set_postfix({"train/loss_step": f"{loss_step:.3f}", "train/acc_step": f"{acc_step:.3f}"})

    return {
        "train/loss_epoch": total_loss / len(loader),
        "train/acc_epoch": int(total_correct) / len(loader.dataset),  # type: ignore[arg-type]
    }


def eval_one_epoch(model: Module, loader: DataLoader, device: str = "cuda") -> Dict[str, float]:
    model.eval()
    correct = 0
    for data in tqdm(loader, total=len(loader), desc="Evaluating"):
        coords = data["coords"].to(device)
        normals = data["normals"].to(device)
        target = data["target"].to(device)
        batch = data["batch"].to(device)
        features = torch.cat([coords, normals], dim=1)

        with torch.no_grad():
            preds = model(features, coords, batch).max(1)[1]
        correct += preds.eq(target).sum().item()

    return {"val/acc": correct / len(loader.dataset)}  # type: ignore[arg-type]


def collate(data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    batch = torch.cat([torch.ones(len(d["coords"])) * i for i, d in enumerate(data_list)]).long()
    coords = torch.cat([d["coords"] for d in data_list]).float()
    normals = torch.cat([d["normals"] for d in data_list]).float()
    target = torch.stack([d["target"] for d in data_list])

    return {"coords": coords, "normals": normals, "target": target, "batch": batch}


if __name__ == "__main__":
    main()
