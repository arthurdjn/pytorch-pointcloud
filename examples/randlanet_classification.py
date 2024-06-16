from argparse import ArgumentParser
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

import torch_pointcloud.transforms as T2
from torch_pointcloud.datasets import ModelNet10
from torch_pointcloud.models.randlanet import RandLANetClassification
from torch_pointcloud.utils.utils import set_seed


def get_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--root", type=str, default="data")
    parser.add_argument("--dataset", type=str, default="ModelNet10")
    parser.add_argument("--num_points", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lr", type=float, default=0.001)
    return parser


def train_one_epoch(
    model: Module,
    optimizer: Optimizer,
    loader: DataLoader,
    device: str = "cuda",
    log_interval: int = 5,
) -> Dict[str, float]:
    model.train()

    cum_loss, total = 0.0, 0
    with tqdm(total=len(loader), desc="Training") as pbar:
        for i, batch in pbar:
            optimizer.zero_grad()
            pos = batch["pos"].to(device)
            y = batch["target"].to(device)
            preds = model(pos, None)
            loss = F.nll_loss(preds, y)
            loss.backward()
            optimizer.step()
            cum_loss += loss.item()
            total += len(y)

            if i % log_interval == 0:
                pbar.set_postfix({"train/loss_step": loss.item()})

    return {"train/loss_epoch": cum_loss / total}


@torch.no_grad()
def eval_one_epoch(model: Module, loader: DataLoader, device: str = "cuda") -> Dict[str, float]:
    model.eval()

    correct, total = 0, 0
    for batch in tqdm(loader, total=len(loader), desc="Evaluating"):
        pos = batch["pos"].to(device)
        y = batch["target"].to(device)
        preds = model(pos, None).max(1)[1]
        correct += preds.eq(y).sum().item()
        total += len(y)

    return {"val/acc": correct / total}


def main() -> None:
    parser = get_parser()
    args = parser.parse_args()
    set_seed(42)

    pre_transform = T2.NormalizeScale()
    transform = T2.Compose([T2.SampleMeshPoints(args.num_points)])
    if args.dataset == "ModelNet10":
        train_dataset = ModelNet10(args.root, True, transform=transform, pre_transform=pre_transform, download=True)
        test_dataset = ModelNet10(args.root, False, transform=transform, pre_transform=pre_transform, download=True)
    else:
        raise ValueError(f"Unrecognized dataset {args.dataset!r}. Must be 'ModelNet10'.")

    def collate(data_list: List[Any]) -> Dict[str, Any]:
        return {
            "pos": torch.stack([d["pos"] for d in data_list]),
            "target": torch.cat([d["target"] for d in data_list]),
        }

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

    model = RandLANetClassification(num_features=3, num_classes=10).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")
        metrics = {}
        train_metrics = train_one_epoch(model, optimizer, train_loader, args.device)
        val_metrics = eval_one_epoch(model, test_loader, args.device)
        metrics.update(train_metrics)
        metrics.update(val_metrics)

        print("Scores:", end=" ")
        print(" | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()]))


if __name__ == "__main__":
    main()
