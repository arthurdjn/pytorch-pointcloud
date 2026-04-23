import os
from argparse import ArgumentParser, Namespace
from typing import Dict

import torch
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from torch_pointcloud.config import DATA_DIR
from torch_pointcloud.datasets import S3DIS
from torch_pointcloud.models import create_model
from torch_pointcloud.utils.data import DataKeys
from torch_pointcloud.utils.data import collate as collate_packed
from torch_pointcloud.utils.metrics import confusion_matrix
from torch_pointcloud.utils.random import seed_everything

CUDA_AVAILABLE = torch.cuda.is_available()
CPU_COUNT = os.cpu_count()
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"
NUM_WORKERS = CPU_COUNT // 2 if CPU_COUNT is not None else 0
BATCH_SIZE = 1
SEED = 42


def main() -> None:
    args = parse_args()

    print(f"Seeding everything to {args.seed}!")
    seed_everything(args.seed)
    print(f"Benchmarking model {args.model!r} on S3DIS areas {list(args.areas)}!")

    model, info = create_model(args.model, task="segmentation", pretrained=args.pretrained, return_info=True)
    num_classes: int = int(model.num_classes)

    transform = info.get("transforms")
    if transform is None:
        raise ValueError(
            f"Model {args.model!r} has no registered `transforms`. "
            "Use a registered S3DIS KP-FCNN checkpoint (e.g. 'kpfcnn-base.s3dis')."
        )

    test_dataset: Dataset = S3DIS(
        root=args.root,
        areas=list(args.areas),
        download=args.download,
        transform=transform,
        show_progress=False,
    )
    if args.limit is not None:
        n = min(int(args.limit), len(test_dataset))  # type: ignore[arg-type]
        test_dataset = Subset(test_dataset, range(n))
        print(f"Evaluating on a subset of the first {n} rooms.")

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_packed,
    )

    print("Evaluating (metrics on subsampled points after the model transform)...")
    metrics = evaluate(model, test_dataloader, args.device, num_classes=num_classes)

    print("\nScores:", end=" ")
    print(" | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()]))


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Evaluate a pretrained KP-FCNN model on S3DIS semantic segmentation.")
    pretrained_group = parser.add_mutually_exclusive_group()
    pretrained_group.add_argument(
        "--pretrained",
        dest="pretrained",
        action="store_true",
        help="Load registry weights (default).",
    )
    pretrained_group.add_argument(
        "--no-pretrained",
        dest="pretrained",
        action="store_false",
        help="Skip loading checkpoint (architecture-only smoke test).",
    )
    parser.set_defaults(pretrained=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--root", type=str, default=DATA_DIR, help="Dataset root (contains S3DIS/).")
    parser.add_argument(
        "--areas",
        nargs="+",
        default=["Area_5"],
        help="Which S3DIS areas to load (default: Area_5, the usual held-out test area).",
    )
    parser.add_argument("--model", type=str, default="kpfcnn-base.s3dis", help="Registered segmentation model name.")
    parser.add_argument("--download", action="store_true", help="Download S3DIS if missing.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--device", type=str, default=DEVICE)
    parser.add_argument("--limit", type=int, default=None, help="Evaluate at most this many rooms (debug).")
    return parser.parse_args()


@torch.no_grad()
def evaluate(model: Module, dataloader: DataLoader, device: str, *, num_classes: int) -> Dict[str, float]:
    model.to(device).eval()
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)

    pbar = tqdm(dataloader, total=len(dataloader), desc="Testing")
    for data in pbar:
        x = data[DataKeys.X].to(device)
        pos = data[DataKeys.POS].to(device)
        target = data[DataKeys.LABEL].to(device)
        batch = data[DataKeys.BATCH].to(device)

        logits = model(x, pos, batch)
        preds = logits.argmax(dim=1)

        cm += confusion_matrix(preds.cpu(), target.cpu(), num_classes)
        oa = cm.diag().sum().float() / cm.sum().float().clamp_min(1)
        pbar.set_postfix({"oa": f"{oa.item():.4f}"})

    intersection = cm.diag().float()
    union = cm.sum(dim=1).float() + cm.sum(dim=0).float() - intersection
    iou_per_class = intersection / union.clamp_min(1e-10)
    miou = iou_per_class.nan_to_num(nan=0.0).mean()
    oa = cm.diag().sum().float() / cm.sum().float()
    return {"test/mIoU": miou.item(), "test/oa": oa.item()}


if __name__ == "__main__":
    main()
