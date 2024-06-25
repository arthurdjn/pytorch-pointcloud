from argparse import ArgumentParser
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

import torch_pointcloud.transforms as T
from torch_pointcloud.datasets import ShapeNet
from torch_pointcloud.models.randlanet import RandLANetSegmentation
from torch_pointcloud.utils.utils import set_seed


def get_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--root", type=str, default="data")
    parser.add_argument("--num_points", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lr", type=float, default=0.01)
    return parser


def train_one_epoch(
    model: Module,
    optimizer: Optimizer,
    loader: DataLoader,
    device: str = "cuda",
    log_interval: int = 5,
) -> Dict[str, float]:
    model.train()

    total_loss = correct = total = 0.0
    with tqdm(total=len(loader), desc="Training") as pbar:
        for i, batch in enumerate(loader):
            optimizer.zero_grad()
            xyz = batch["xyz"].to(device)
            target = batch["segmentation_target"].to(device)
            logits = model(xyz, None)
            preds = logits.log_softmax(dim=-1)
            loss = F.nll_loss(preds, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += preds.argmax(dim=1).eq(target).sum().item()
            total += target.size(0) * target.size(1)
            pbar.update(1)

            if i % log_interval == 0:
                pbar.set_postfix({"train/loss_step": loss.item(), "train/acc": correct / total})

    return {"train/loss_epoch": total_loss / total, "train/acc": correct / total}


@torch.no_grad()
def eval_one_epoch(model: Module, loader: DataLoader, device: str = "cuda") -> Dict[str, float]:
    model.eval()

    correct, total = 0, 0
    for batch in tqdm(loader, total=len(loader), desc="Evaluating"):
        xyz = batch["xyz"].to(device)
        target = batch["segmentation_target"].to(device)
        preds = model(xyz, None).max(1)[1]
        correct += preds.eq(target).sum().item()
        total += len(target)

    return {"val/acc": correct / len(loader.dataset)}  # type: ignore


def main() -> None:
    parser = get_parser()
    args = parser.parse_args()
    set_seed(42)

    pre_transform = T.NormalizeScale()
    transform = T.Compose([T.SampleRandomPoints(args.num_points, keys=("xyz", "segmentation_target"))])
    train_dataset = ShapeNet(
        args.root,
        split="train",
        categories=["Airplane"],
        transform=transform,
        pre_transform=pre_transform,
        download=True,
    )
    test_dataset = ShapeNet(
        args.root,
        split="test",
        categories=["Airplane"],
        transform=transform,
        pre_transform=pre_transform,
        download=True,
    )

    def collate(data_list: List[Any]) -> Dict[str, Any]:
        return {
            "xyz": torch.stack([d["xyz"] for d in data_list]),
            "segmentation_target": torch.stack([d["segmentation_target"] for d in data_list]),
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

    model = RandLANetSegmentation(in_channels=3, num_classes=50).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

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
