"""Train DGCNN on random ScanNet blocks, following dgcnn.pytorch's `main_semseg_scannet.py`."""

import os
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

import torch_pointcloud.transforms as T
from torch_pointcloud.config import DATA_DIR
from torch_pointcloud.datasets import RepeatSampler, ScanNet20
from torch_pointcloud.inferers import Inferer, SlidingWindowInferer
from torch_pointcloud.metrics import confusion_matrix, intersection_over_union
from torch_pointcloud.models import create_model
from torch_pointcloud.optim import bn_momentum, set_bn_momentum
from torch_pointcloud.utils.data import DataKeys, PointCloudDataLoader
from torch_pointcloud.utils.random import seed_everything

CPU_COUNT = os.cpu_count()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = CPU_COUNT // 2 if CPU_COUNT is not None else 0
MODELS = ["dgcnn.scannet20.an-tao"]
NUM_CLASSES = 20
IGNORE_INDEX = 255
NUM_POINTS = 8192
BLOCK_SIZE = 1.5
EPOCHS = 200
BATCH_SIZE = 36
LR = 1e-3
WEIGHT_DECAY = 1e-4
MIN_LR = 1e-5
BN_MOMENTUM = 0.1
BN_DECAY_RATE = 0.5
BN_DECAY_STEP = 10
BN_MOMENTUM_CLIP = 0.01
SW_BATCH_SIZE = 8
POINT_KEYS = [DataKeys.POS, DataKeys.COLOR, DataKeys.SEGMENT]
TRAIN_TRANSFORM = T.Compose(
    [
        T.KeepItems(keys=POINT_KEYS),
        T.Relabel(keys=DataKeys.SEGMENT, labels=list(range(1, 21)), default=IGNORE_INDEX),
        T.ToTensor(keys=DataKeys.SEGMENT, dtype=torch.long),
        T.Reduce(keys=DataKeys.POS, op="max", dst_keys=DataKeys.SCENE_MAX),  # the whole scene, before the cut
        # The reference redraws the block up to ten times until 70 % of its points are labeled and it fills 2 %
        # of its voxels; here any block holding points is accepted.
        T.RandomBlockCrop(
            pos_key=DataKeys.POS,
            keys=[DataKeys.COLOR, DataKeys.SEGMENT],
            block_size=BLOCK_SIZE,
            min_nodes=0,
            max_tries=10,
            dst_center_key=DataKeys.BLOCK_CENTER,
        ),
        T.RandomSample(keys=POINT_KEYS, num_samples=NUM_POINTS),
        T.CopyItems(keys=DataKeys.POS, dst_keys=DataKeys.NORM_POS),
        T.DivideItems(keys=DataKeys.NORM_POS, div_keys=DataKeys.SCENE_MAX),
        T.SubtractItems(keys=DataKeys.POS, sub_keys=DataKeys.BLOCK_CENTER, axes=[0, 1]),
        T.Divide(keys=DataKeys.COLOR, divisor=255),
        T.Cat(keys=[DataKeys.POS, DataKeys.COLOR], dst_key=DataKeys.X),
    ]
)
VAL_TRANSFORM = T.Compose(
    [
        T.Relabel(keys=DataKeys.SEGMENT, labels=list(range(1, 21)), default=IGNORE_INDEX),
        T.ToTensor(keys=DataKeys.SEGMENT, dtype=torch.long),
        T.Reduce(keys=DataKeys.POS, op="max", dst_keys=DataKeys.SCENE_MAX),
    ]
)
INFERER_TRANSFORM = T.Compose(
    [
        T.BBoxCenter(keys=DataKeys.BLOCK_BBOX, dst_keys=DataKeys.BLOCK_CENTER),
        T.CopyItems(keys=DataKeys.POS, dst_keys=DataKeys.NORM_POS),
        T.DivideItems(keys=DataKeys.NORM_POS, div_keys=DataKeys.SCENE_MAX),
        T.SubtractItems(keys=DataKeys.POS, sub_keys=DataKeys.BLOCK_CENTER, axes=[0, 1]),
        T.Divide(keys=DataKeys.COLOR, divisor=255),
        T.Cat(keys=[DataKeys.POS, DataKeys.COLOR], dst_key=DataKeys.X),
        T.DivisiblePad(num_samples=NUM_POINTS, pad_fill="random", dst_inverse_key=DataKeys.INVERSE),
    ]
)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    train_loader, val_loader = configure_dataloaders(args)
    model = configure_model(args)
    optimizer, scheduler = configure_optimizers(args, model)
    criterion = configure_criterion(args)
    inferer = configure_inferer(args)

    num_train, num_val = len(train_loader.sampler), len(val_loader.dataset)  # type: ignore[arg-type]
    print(f"Training {args.model!r} on {num_train} blocks per epoch, validating on {num_val} scenes.")
    output: Optional[Path] = Path(args.output) if args.output else None
    best_loss = float("inf")
    for epoch in range(args.epochs):
        momentum = bn_momentum(
            epoch, init=BN_MOMENTUM, decay_rate=BN_DECAY_RATE, decay_step=BN_DECAY_STEP, clip=BN_MOMENTUM_CLIP
        )
        set_bn_momentum(model, momentum)
        loss = train_one_epoch(model, criterion, optimizer, train_loader, args.device)
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        print(f"Epoch {epoch + 1}/{args.epochs}  lr={lr:.2e}  bn_momentum={momentum:.3f}  train/loss {loss:.4f}")

        if (epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs:
            val = evaluate(model, val_loader, inferer, args.device)
            print("  " + " | ".join(f"val/{key} {value * 100:.2f}" for key, value in val.items()))

        if output is not None:
            output.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), output / "last.pt")
            if loss < best_loss:
                best_loss = loss
                torch.save(model.state_dict(), output / "best.pt")


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Train DGCNN semantic segmentation on ScanNet.")
    parser.add_argument("--model", default=MODELS[0], choices=MODELS)
    parser.add_argument("--root", default=DATA_DIR, help="Dataset root directory.")
    parser.add_argument("--output", default=None, help="Directory for the checkpoints (default: none saved).")
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--epochs", default=EPOCHS, type=int)
    parser.add_argument("--batch-size", default=BATCH_SIZE, type=int)
    parser.add_argument("--lr", default=LR, type=float)
    parser.add_argument("--weight-decay", default=WEIGHT_DECAY, type=float)
    parser.add_argument("--sw-batch-size", default=SW_BATCH_SIZE, type=int, help="Validation blocks per forward.")
    parser.add_argument("--num-workers", default=NUM_WORKERS, type=int)
    parser.add_argument("--eval-every", default=1, type=int, help="Validate every this many epochs.")
    parser.add_argument("--limit-train-batches", default=None, type=int)
    parser.add_argument("--limit-val-batches", default=None, type=int)
    parser.add_argument("--download", action="store_true", help="Download the dataset if missing.")
    parser.add_argument("--force-process", action="store_true", help="Force re-processing the dataset.")
    return parser.parse_args()


def configure_datasets(args: Namespace) -> tuple[Dataset, Dataset]:
    common: Dict[str, Any] = dict(
        root=args.root,
        download=args.download,
        force_process=args.force_process,
        num_workers=args.num_workers,
        use_axis_alignment=False,
    )

    train_dataset: Dataset
    val_dataset: Dataset
    train_dataset = ScanNet20(split="train", transform=TRAIN_TRANSFORM, **common)
    val_dataset = ScanNet20(split="val", transform=VAL_TRANSFORM, **common)

    if args.limit_val_batches is not None:
        val_dataset = Subset(val_dataset, range(min(args.limit_val_batches, len(val_dataset))))

    return train_dataset, val_dataset


def configure_dataloaders(args: Namespace) -> tuple[DataLoader, DataLoader]:
    train_dataset, val_dataset = configure_datasets(args)

    # Every scene is drawn in proportion to its points, a new random block each time.
    draws = draws_per_scene(train_dataset)
    if args.limit_train_batches is not None:
        draws = first_draws(draws, args.limit_train_batches * args.batch_size)
    sampler = RepeatSampler(draws)

    train_loader = PointCloudDataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        drop_last=len(sampler) >= args.batch_size,
        num_workers=args.num_workers,
    )
    # Scenes go through the sliding window one at a time.
    val_loader = PointCloudDataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    return train_loader, val_loader


def configure_model(args: Namespace) -> nn.Module:
    return create_model(args.model, task="semantic-segmentation", pretrained=False).to(args.device)


def configure_optimizers(args: Namespace, model: nn.Module) -> tuple[Optimizer, LRScheduler]:
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=MIN_LR)
    return optimizer, scheduler


def configure_criterion(args: Namespace) -> nn.Module:
    return nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)


def configure_inferer(args: Namespace) -> Inferer:
    return SlidingWindowInferer(
        block_size=BLOCK_SIZE,
        overlap=0.5,
        dims=(0, 1),
        padding=1e-8,
        roi_num_points=NUM_POINTS,
        sw_batch_size=args.sw_batch_size,
        softmax=False,
        aggregate="mean",
        transform=INFERER_TRANSFORM,
        inverse_key=DataKeys.INVERSE,
        seed=args.seed,
    )


def train_one_epoch(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: Optimizer,
    loader: DataLoader,
    device: str,
) -> float:
    model.train()
    total_loss = 0.0
    for data in tqdm(loader, desc="Training", leave=False):
        data = {key: value.to(device) for key, value in data.items() if isinstance(value, Tensor)}
        logits = model(data[DataKeys.X], data[DataKeys.NORM_POS], data[DataKeys.BATCH])
        loss = criterion(logits, data[DataKeys.SEGMENT])

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, inferer: Inferer, device: str) -> dict[str, float]:
    """Whole scenes scored through the sliding-block protocol of the benchmark."""
    model.eval()
    confusion = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)
    for data in tqdm(loader, desc="Evaluating", leave=False):
        data = {key: value.to(device) if isinstance(value, Tensor) else value for key, value in data.items()}
        logits = inferer(data, predictor=lambda d: model(d[DataKeys.X], d[DataKeys.NORM_POS], d[DataKeys.BATCH]))
        preds = logits.argmax(dim=1).cpu()
        confusion += confusion_matrix(preds, data[DataKeys.SEGMENT].cpu(), NUM_CLASSES, ignore_index=IGNORE_INDEX)

    present = confusion.sum(dim=1) > 0
    accuracy = confusion.diag().sum() / confusion.sum()
    mean_accuracy = (confusion.diag() / confusion.sum(dim=1).clamp_min(1))[present].mean()
    return {"mIoU": float(intersection_over_union(confusion)), "OA": accuracy.item(), "mAcc": mean_accuracy.item()}


def draws_per_scene(dataset: Dataset) -> List[int]:
    """Blocks drawn from each scene per epoch: about one per `NUM_POINTS` of its points, at least one."""
    scenes = dataset
    while not isinstance(scenes, ScanNet20):
        scenes = scenes.dataset  # type: ignore[attr-defined]

    draws = []
    for entry in scenes.data:
        if isinstance(entry, dict):
            num_points = entry[DataKeys.POS].shape[0]
        else:
            num_points = np.load(entry / "pos.npy", mmap_mode="r").shape[0]
        draws.append(max(1, round(num_points / NUM_POINTS)))  # the reference rounds, leaving tiny scenes undrawn
    return draws


def first_draws(draws: List[int], budget: int) -> List[int]:
    """The per-scene draws truncated to the first `budget` draws, scenes in order."""
    kept = []
    for count in draws:
        kept.append(min(count, budget))
        budget -= kept[-1]
    return kept


if __name__ == "__main__":
    main()
