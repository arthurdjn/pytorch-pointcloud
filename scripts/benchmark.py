import os
from argparse import ArgumentParser, Namespace
from typing import Any, Dict, List

import torch
from dotenv import load_dotenv
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from torch_pointcloud.datasets import ModelNetNormalResampled, ScanObjectNN
from torch_pointcloud.models import create_model
from torch_pointcloud.utils.random import seed_everything

load_dotenv()

from torch_pointcloud.config import DATA_DIR  # noqa: E402

CUDA_AVAILABLE = torch.cuda.is_available()
CPU_COUNT = os.cpu_count()
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"
NUM_WORKERS = CPU_COUNT // 2 if CPU_COUNT is not None else 0
BATCH_SIZE = 32
SEED = 42


def main() -> None:
    args = parse_args()

    print(f"Seeding everything to {args.seed}!")
    seed_everything(args.seed)
    print(f"Benchmarking model {args.model!r} on dataset {args.dataset}!")

    model, info = create_model(args.model, task="classification", pretrained=True, return_info=True)

    test_dataset: Dataset
    if args.dataset == "modelnet40":
        test_dataset = ModelNetNormalResampled(
            root=args.root,
            variant="40",
            train=False,
            transform=info["transforms"],
            download=False,
            num_workers=args.num_workers,
        )
    elif args.dataset == "scanobjectnn":
        test_dataset = ScanObjectNN(
            root=args.root,
            train=False,
            split="main",
            background=True,
            download=False,
            transform=info["transforms"],
        )
    elif args.dataset == "scanobjectnn-nobg":
        test_dataset = ScanObjectNN(
            root=args.root,
            train=False,
            split="main",
            background=False,
            download=False,
            transform=info["transforms"],
        )
    elif args.dataset == "scanobjectnn-augmentedrot-scale75":
        test_dataset = ScanObjectNN(
            root=args.root,
            train=False,
            split="main",
            variant="augmentedrot_scale75",
            background=True,
            download=False,
            transform=info["transforms"],
        )
    else:
        raise ValueError(f"Unrecognized dataset {args.dataset!r}. Must be 'modelnet40'.")

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
    )

    print("Evaluating...")
    metrics = evaluate(model, test_dataloader, args.device)

    print("\nScores:", end=" ")
    print(" | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()]))


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--root", type=str, default=DATA_DIR, help="The root directory to store the data.")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--device", type=str, default=DEVICE)
    return parser.parse_args()


@torch.no_grad()
def evaluate(model: Module, dataloader: DataLoader, device: str = "cuda") -> Dict[str, float]:
    correct = 0
    model.to(device).eval()
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Testing")
    for i, data in pbar:
        x = data["x"].to(device) if "x" in data else None
        pos = data["pos"].to(device)
        target = data["target"].to(device)
        batch = data["batch"].to(device)

        preds = model(x, pos, batch).max(1)[1]
        correct += preds.eq(target).sum().item()
        pbar.set_postfix({"acc": correct / (i + 1)})

    return {"test/acc": correct / len(dataloader.dataset)}  # type: ignore[arg-type]


def collate(data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    first_data = data_list[0]
    keys = set(first_data.keys()) - {"batch"}

    batch_data: Dict[str, Any] = {}
    for key in keys:
        first_value = first_data[key]
        if isinstance(first_value, torch.Tensor) and first_value.ndim > 0:
            batch_data[key] = torch.cat([d[key] for d in data_list])
        elif isinstance(first_value, torch.Tensor) and first_value.ndim == 0:
            batch_data[key] = torch.tensor([d[key] for d in data_list])
        elif isinstance(first_value, (int, float, bool)):
            batch_data[key] = torch.tensor([d[key] for d in data_list])
        else:
            batch_data[key] = [d[key] for d in data_list]

    if "pos" in keys:
        batch_data["batch"] = torch.cat([torch.ones(len(d["pos"])) * i for i, d in enumerate(data_list)]).long()
    else:
        raise ValueError(
            f"Could not determine batch ids from the input data with the following keys: {', '.join(keys)} "
            f"Make sure the input data to collate contains dictionary data with at least one tensor with shape (N, ...)."
        )

    return batch_data


if __name__ == "__main__":
    main()
