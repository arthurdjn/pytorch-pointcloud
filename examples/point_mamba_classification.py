from argparse import ArgumentParser, Namespace

import torch
import torch.nn.functional as F
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

import torch_pointcloud.transforms as T
from torch_pointcloud.config import DATA_DIR
from torch_pointcloud.datasets import ModelNet10, ModelNet40
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
        total_steps=len(train_dataloader) * args.epochs,
    )

    print("\nStarting training!\n")
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--root", type=str, default=DATA_DIR)
    parser.add_argument("--dataset", type=str, default="modelnet10", choices=["modelnet10", "modelnet40"])
    parser.add_argument(
        "--model",
        type=str,
        default="point-mamba-base.modelnet40",
        choices=["point-mamba-base.modelnet40"],
    )
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
    dataloader: DataLoader,
    device: str = "cuda",
    log_interval: int = 5,
) -> dict[str, float]:
    total_correct = total_loss = 0.0
    model.train()

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Training")
    for i, data in pbar:
        pos = data[DataKeys.POS].to(device)
        normal = data[DataKeys.NORMAL].to(device)
        target = data[DataKeys.LABEL].to(device)
        batch = data[DataKeys.BATCH].to(device)
        x = torch.cat([pos, normal], dim=1)

        optimizer.zero_grad()
        logits = model(x, pos, batch)
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
        "train/loss_epoch": total_loss / len(dataloader),
        "train/acc_epoch": int(total_correct) / len(dataloader.dataset),  # type: ignore[arg-type]
    }


def eval_one_epoch(model: Module, dataloader: DataLoader, device: str = "cuda") -> dict[str, float]:
    model.eval()
    correct = 0
    for data in tqdm(dataloader, total=len(dataloader), desc="Evaluating"):
        pos = data[DataKeys.POS].to(device)
        normal = data[DataKeys.NORMAL].to(device)
        target = data[DataKeys.LABEL].to(device)
        batch = data[DataKeys.BATCH].to(device)
        x = torch.cat([pos, normal], dim=1)

        with torch.no_grad():
            preds = model(x, pos, batch).max(1)[1]
        correct += preds.eq(target).sum().item()

    return {"val/acc": correct / len(dataloader.dataset)}  # type: ignore[arg-type]


def configure_dataloaders(args: Namespace) -> tuple[DataLoader, DataLoader]:
    transform = T.Compose(
        [
            T.Rescale(keys=DataKeys.POS),
            T.RandomSampleFaceVertices(
                keys=DataKeys.POS,
                face_key=DataKeys.FACE,
                normal_key=DataKeys.NORMAL,
                num_samples=args.num_points,
            ),
        ]
    )

    train_dataset: Dataset
    test_dataset: Dataset
    if args.dataset.lower() == "modelnet10":
        train_dataset = ModelNet10(
            args.root,
            True,
            transform=transform,
            download=True,
            num_workers=args.num_workers,
        )
        test_dataset = ModelNet10(
            args.root,
            False,
            transform=transform,
            download=True,
            num_workers=args.num_workers,
        )
    elif args.dataset.lower() == "modelnet40":
        train_dataset = ModelNet40(
            args.root,
            True,
            transform=transform,
            download=True,
            num_workers=args.num_workers,
        )
        test_dataset = ModelNet40(
            args.root,
            False,
            transform=transform,
            download=True,
            num_workers=args.num_workers,
        )
    else:
        raise ValueError(f"Unrecognized dataset {args.dataset!r}. Must be 'ModelNet10' or 'ModelNet40'.")

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
