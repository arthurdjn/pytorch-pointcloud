from argparse import ArgumentParser, Namespace
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import torch_pointcloud.transforms as T
from torch_pointcloud.datasets import ModelNet10, ModelNet40
from torch_pointcloud.models import PointTransformerV3Classification
from torch_pointcloud.utils.random import seed_everything


def main() -> None:
    args = parse_args()
    seed_everything(42)

    pre_transform = T.NormalizeScaled(keys="pos")
    transform = T.Compose(
        [
            T.RandomSampleFaceVerticesd(
                keys="pos",
                face_key="face",
                normal_key="normal",
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

    model = PointTransformerV3Classification(
        num_classes=args.num_classes,
        in_channels=6,
        serialization_orders=("z", "z-trans", "hilbert", "hilbert-trans"),
        shuffle_serialization_orders=True,
        stride=(2, 2, 2, 2),
        encoder_depths=(2, 2, 2, 6, 2),
        encoder_channels=(32, 64, 128, 256, 512),
        encoder_num_head=(2, 4, 8, 16, 32),
        encoder_patch_size=(1024, 1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        with_rpe=False,
        with_flash_attn=True,
        upcast_attention=False,
        upcast_softmax=False,
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

    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")
        train_metrics = train_one_epoch(model, optimizer, train_dataloader, args.device)
        val_metrics = eval_one_epoch(model, test_dataloader, args.device)
        metrics = {**train_metrics, **val_metrics}
        scheduler.step()

        print("Scores:", end=" ")
        print(" | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()]))


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--root", type=str, default="data")
    parser.add_argument("--dataset", type=str, default="modelnet10", choices=["modelnet10", "modelnet40"])
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def train_one_epoch(
    model: Module,
    optimizer: Optimizer,
    dataloader: DataLoader,
    device: str = "cuda",
    log_interval: int = 5,
) -> Dict[str, float]:
    total_correct = total_loss = 0.0
    model.train()

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Training")
    for i, data in pbar:
        coords = data["pos"].to(device)
        normal = data["normal"].to(device)
        target = data["label"].to(device)
        batch = data["batch"].to(device)

        grid_size = 0.01
        grid_coords = torch.div(coords - coords.min(0)[0], grid_size, rounding_mode="trunc").int()
        features = torch.cat([coords, normal], dim=1)

        optimizer.zero_grad()
        logits = model(features, grid_coords, batch)
        probs = F.log_softmax(logits, dim=1)

        loss = F.nll_loss(probs, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        correct = logits.argmax(dim=1).eq(target).sum()
        total_correct += correct.item()

        if i % log_interval == 0:
            pbar.set_postfix({"train/loss_step": loss.item(), "train/acc_step": correct.item() / len(target)})

    return {
        "train/loss_epoch": total_loss / len(dataloader),
        "train/acc_epoch": int(total_correct) / len(dataloader.dataset),  # type: ignore[arg-type]
    }


def eval_one_epoch(model: Module, dataloader: DataLoader, device: str = "cuda") -> Dict[str, float]:
    model.eval()
    total_correct = 0
    for data in tqdm(dataloader, total=len(dataloader), desc="Evaluating"):
        coords = data["pos"].to(device)
        normal = data["normal"].to(device)
        target = data["label"].to(device)
        batch = data["batch"].to(device)

        grid_size = 0.01
        grid_coords = torch.div(coords - coords.min(0)[0], grid_size, rounding_mode="trunc").int()
        features = torch.cat([coords, normal], dim=1)

        with torch.no_grad():
            logits = model(features, grid_coords, batch)

        correct = logits.argmax(1).eq(target).sum()
        total_correct += correct.item()

    return {"val/acc": total_correct / len(dataloader.dataset)}  # type: ignore[arg-type]


def collate(data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    batch = torch.cat([torch.ones(len(d["pos"])) * i for i, d in enumerate(data_list)]).long()
    coords = torch.cat([d["pos"] for d in data_list]).float()
    normal = torch.cat([d["normal"] for d in data_list]).float()
    target = torch.stack([d["label"] for d in data_list])

    return {"pos": coords, "normal": normal, "label": target, "batch": batch}


if __name__ == "__main__":
    main()
