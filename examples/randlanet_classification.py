from argparse import ArgumentParser, Namespace
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

import torch_pointcloud.transforms as T
from torch_pointcloud.datasets import ModelNet10
from torch_pointcloud.models import RandLANetClassification
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

    print(f"Loading {args.dataset} dataset...")
    if args.dataset == "ModelNet10":
        train_dataset = ModelNet10(args.root, True, transform=transform, pre_transform=pre_transform, download=True)
        test_dataset = ModelNet10(args.root, False, transform=transform, pre_transform=pre_transform, download=True)
    else:
        raise ValueError(f"Unrecognized dataset {args.dataset!r}. Must be 'ModelNet10'.")

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
    model = RandLANetClassification(
        in_channels=3,
        num_classes=10,
        stem_channels=8,
        encoder_channels=[32, 128],
        num_neighbors=[16, 16],
        aggr_channels=128,
        decimation=4,
    ).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

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
    parser.add_argument("--dataset", type=str, default="ModelNet10")
    parser.add_argument("--num_points", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=6)
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
    cum_loss = 0.0
    model.train()

    pbar = tqdm(enumerate(loader), total=len(loader), desc="Training")
    for i, data in pbar:
        coords = data["coords"].to(device)
        target = data["target"].to(device)
        features = data["normals"].to(device)
        batch = data["batch"].to(device)

        optimizer.zero_grad()
        logits = model(features, coords, batch)
        preds = F.log_softmax(logits, dim=1)
        loss = F.nll_loss(preds, target)
        loss.backward()
        optimizer.step()
        cum_loss += loss.item()

        if i % log_interval == 0:
            pbar.set_postfix({"train/loss_step": loss.item()})

    return {"train/loss_epoch": cum_loss / len(loader)}


def eval_one_epoch(model: Module, loader: DataLoader, device: str = "cuda") -> Dict[str, float]:
    model.eval()
    correct = 0
    for data in tqdm(loader, total=len(loader), desc="Evaluating"):
        coords = data["coords"].to(device)
        target = data["target"].to(device)
        features = data["normals"].to(device)
        batch = data["batch"].to(device)

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
