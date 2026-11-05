"""Benchmark Cylinder3D semantic segmentation on SemanticKITTI in a single pass, scored on every point.

NOTE: 67.01 is the training-log validation of the LaserMix/PolarMix file; mmdetection3d fe25f7a5 scores it at 67.54:

```python
# mmdetection3d v1.4.0, every raw scan of sequence 08, one scan per forward, no test-time augmentation
model = init_model("configs/cylinder3d/cylinder3d_8xb2-laser-polar-mix-3x_semantickitti.py", checkpoint)
data = model.data_preprocessor({"inputs": {"points": [points]}, "data_samples": [Det3DDataSample()]}, False)
logits = model.decode_head.predict(model.extract_feat(data["inputs"]), data["inputs"], data["data_samples"])[0]
seg_eval(gt_labels, predictions, label2cat, ignore_index=19)
```
"""

import os
from argparse import ArgumentParser, Namespace
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

import torch_pointcloud.transforms as T
from torch_pointcloud.config import DATA_DIR
from torch_pointcloud.datasets import SemanticKITTI
from torch_pointcloud.inferers import Inferer, SimpleInferer
from torch_pointcloud.metrics import accuracy, confusion_matrix, intersection_over_union
from torch_pointcloud.models import SemanticSegmentationModel, create_model, model_info
from torch_pointcloud.utils.data import DataKeys, PointCloudDataLoader
from torch_pointcloud.utils.random import seed_everything, set_determinism

CUDA_AVAILABLE = torch.cuda.is_available()
CPU_COUNT = os.cpu_count()
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"
NUM_WORKERS = CPU_COUNT // 2 if CPU_COUNT is not None else 0
SEED = 42
IGNORE_INDEX = -1


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    set_determinism(tf32=False)

    print(f"Benchmarking model {args.model!r} on SemanticKITTI!")
    model = configure_model(args)
    num_classes = int(model.num_classes)
    inferer = configure_inferer(args)
    dataloader = configure_dataloader(args)

    num_scenes = len(dataloader.dataset)  # type: ignore[arg-type]
    print(f"Test set: {num_scenes} scans  (single pass, scored on every point)")
    metrics = evaluate(model, dataloader, inferer, args.device, num_classes)
    print("\nResults:")
    for key, value in metrics.items():
        print(f"  {key:<24} {value:.4f}")


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Benchmark Cylinder3D semantic segmentation on SemanticKITTI.")
    parser.add_argument(
        "--model", default="cylinder3d.semantickitti.mmdet3d", help="Registered segmentation model name"
    )
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--root", default=DATA_DIR, help="Dataset root directory.")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--seed", default=SEED, type=int)
    parser.add_argument("--num-workers", default=NUM_WORKERS, type=int)
    parser.add_argument("--limit", default=None, type=int, help="Evaluate at most this many scans.")
    return parser.parse_args()


def configure_transform(args: Namespace) -> T.Transform:
    """The registered inference transform of the model."""
    transform = model_info(args.model, task="semantic-segmentation")["transform"]
    if transform is None:
        raise ValueError(f"{args.model!r} registers no transform.")

    return transform


def configure_dataset(args: Namespace) -> Dataset:
    dataset: Dataset
    transform = configure_transform(args)
    dataset = SemanticKITTI(root=args.root, split=args.split, transform=transform)
    if args.limit is not None:
        n = min(int(args.limit), len(dataset))
        dataset = Subset(dataset, range(n))
        print(f"Evaluating on a subset of the first {n} scans.")

    return dataset


def configure_dataloader(args: Namespace) -> DataLoader:
    dataset = configure_dataset(args)
    return PointCloudDataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)


def configure_model(args: Namespace) -> SemanticSegmentationModel:
    return create_model(args.model, task="semantic-segmentation", pretrained=True)


def configure_inferer(args: Namespace) -> Inferer:
    return SimpleInferer()


@torch.no_grad()
def evaluate(
    model: SemanticSegmentationModel, dataloader: DataLoader, inferer: Inferer, device: str, num_classes: int
) -> dict[str, Any]:
    model.to(device).eval()
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)

    pbar = tqdm(dataloader, total=len(dataloader), desc="Testing")
    for data in pbar:
        data = {key: value.to(device) if torch.is_tensor(value) else value for key, value in data.items()}
        scores = inferer(data, predictor=lambda d: model(d[DataKeys.X], d[DataKeys.POS], d[DataKeys.BATCH]))
        preds = scores.argmax(dim=1)
        cm += confusion_matrix(preds.cpu(), data[DataKeys.SEGMENT].cpu(), num_classes, ignore_index=IGNORE_INDEX)
        pbar.set_postfix({"oa": f"{accuracy(cm):.4f}"})

    return {
        "test/mIoU": intersection_over_union(cm),
        "test/oa": accuracy(cm),
    }


if __name__ == "__main__":
    main()
