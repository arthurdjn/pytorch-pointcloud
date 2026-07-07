import os
from argparse import ArgumentParser, Namespace
from typing import TYPE_CHECKING, Any, Dict

import torch
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from torch_pointcloud.config import DATA_DIR
from torch_pointcloud.datasets import ScanNet20, ScanNet200
from torch_pointcloud.models._registry import create_model
from torch_pointcloud.utils.data import DataKeys, collate
from torch_pointcloud.utils.imports import _OCNN_GITHUB_URL, optional_import
from torch_pointcloud.utils.metrics import confusion_matrix
from torch_pointcloud.utils.random import seed_everything

if TYPE_CHECKING:
    from ocnn.octree import Octree, Points

Octree, _ = optional_import("ocnn.octree", "Octree", url=_OCNN_GITHUB_URL)
Points, _ = optional_import("ocnn.octree", "Points", url=_OCNN_GITHUB_URL)

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
    print(f"Benchmarking model {args.model!r} on ScanNet (split={args.split!r})!")

    model, info = create_model(args.model, task="segmentation", pretrained=args.pretrained, return_info=True)
    num_classes: int = int(model.num_classes)

    transform = info.get("transforms")
    if transform is None:
        raise ValueError(
            f"Model {args.model!r} has no registered `transforms`. "
            "Use a registered ScanNet OctFormer checkpoint (e.g. 'octformer-base.scannet20.octree-nn')."
        )

    test_dataset: Dataset
    if "scannet200" in args.model:
        test_dataset = ScanNet200(
            root=args.root,
            split=args.split,
            download=args.download,
            transform=transform,
            force_process=args.force_process,
            num_workers=args.num_workers,
        )
    else:
        test_dataset = ScanNet20(
            root=args.root,
            split=args.split,
            download=args.download,
            transform=transform,
            force_process=args.force_process,
            num_workers=args.num_workers,
            use_axis_alignment=False,
        )

    if args.limit is not None:
        n = min(int(args.limit), len(test_dataset))
        test_dataset = Subset(test_dataset, range(n))
        print(f"Evaluating on a subset of the first {n} scenes.")

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
    )

    print(f"Test set: {len(test_dataset)} scenes")
    print("Evaluating...")
    metrics = evaluate(model, test_dataloader, args.device, num_classes=num_classes)

    print("\nResults:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<20} {v:.4f}")
        else:
            print(f"  {k:<20} {v}")


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Benchmark OctFormer segmentation on ScanNet.")
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
    parser.add_argument("--force-process", action="store_true", help="Force re-processing the dataset.")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--root", type=str, default=DATA_DIR, help="Dataset root (contains ScanNet/).")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument(
        "--model",
        type=str,
        default="octformer-base.scannet20.octree-nn",
        choices=["octformer-base.scannet20.octree-nn", "octformer-base.scannet200.octree-nn"],
    )
    parser.add_argument("--download", action="store_true", help="Download ScanNet if missing.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--device", type=str, default=DEVICE)
    parser.add_argument("--limit", type=int, default=None, help="Evaluate at most this many scenes (debug).")
    return parser.parse_args()


@torch.no_grad()
def evaluate(
    model: Module,
    dataloader: DataLoader,
    device: str,
    *,
    num_classes: int,
    ignore_index: int = 0,
) -> Dict[str, Any]:
    model.to(device).eval()
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)

    pbar = tqdm(dataloader, total=len(dataloader), desc="Testing")
    for data in pbar:
        octree = data[DataKeys.OCTREE].to(device)
        pos = data[DataKeys.POS].to(device)
        batch = data[DataKeys.BATCH].to(device)
        x = data[DataKeys.X].to(device)
        target = data[DataKeys.SEGMENT].to(device)

        logits = model(x, octree, octree.depth, pos, batch)
        preds = logits.argmax(dim=1)

        cm += confusion_matrix(preds.cpu(), target.cpu(), num_classes, ignore_index=ignore_index)
        oa = cm.diag().sum().float() / cm.sum().float().clamp_min(1)
        pbar.set_postfix({"oa": f"{oa.item():.4f}"})

    intersection = cm.diag().float()
    union = cm.sum(dim=1).float() + cm.sum(dim=0).float() - intersection
    iou_per_class = intersection / union.clamp_min(1e-10)

    valid = torch.ones(num_classes, dtype=torch.bool)
    valid[ignore_index] = False
    miou = iou_per_class[valid].mean()
    oa = cm.diag().sum().float() / cm.sum().float()

    return {
        "test/mIoU": miou.item(),
        "test/oa": oa.item(),
        "test/iou_per_class": iou_per_class,
    }


if __name__ == "__main__":
    main()
