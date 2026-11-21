"""Dataset wrapper that gives every sample the same random draws each time it is read."""

from typing import Any

import torch
from torch.utils.data import Dataset

from torch_pointcloud.utils.data import _random_objects


class SeededDataset(Dataset):
    """Re-seeds the seeded random objects of a dataset from the index of the sample being read.

    A seeded random transform draws from one stream that advances with every sample, and `PointCloudDataLoader`
    re-seeds it in every worker and epoch, so a sample gets new draws each time it is read. Wrapped in this class, a
    sample gets the draws of its own index, whatever the epoch, the number of workers and the order of the reads: a
    validation split is scored on the same crops and points at every epoch.

    The random objects are the ones `torch_pointcloud.utils.data.set_random_states` re-seeds (seeded transforms, a
    seeded `MixDataset`), looked up once at construction. Those drawing from the global generator (`seed=None`) keep
    doing so. An index read twice gets the same draws twice: to draw a sample several times per epoch, list its index
    once per draw in a `torch.utils.data.Subset` and wrap the subset.

    Args:
        dataset: The dataset to read from, with seeded random transforms.
        seed: Seed of the draws; sample `index` draws from `seed + index`.

    Raises:
        ValueError: If `dataset` holds no seeded random object.

    Example:
        ```python
        import torch

        import torch_pointcloud.transforms as T
        from torch_pointcloud.datasets import SeededDataset, ShapeNetPart

        transform = T.RandomSample(keys=["pos", "segment"], num_samples=1024, replace=True, seed=0)
        dataset = SeededDataset(ShapeNetPart(root="data", split="test", transform=transform))
        assert torch.equal(dataset[0]["pos"], dataset[0]["pos"])
        ```
    """

    def __init__(self, dataset: Dataset, seed: int = 0) -> None:
        random_objects = [item for item in _random_objects(dataset) if getattr(item, "R", None) is not None]
        if not random_objects:
            raise ValueError("`dataset` holds no seeded random object: pass a `seed` to its random transforms.")

        self.dataset = dataset
        self.seed = seed
        self.random_objects = random_objects

    def __len__(self) -> int:
        return len(self.dataset)  # type: ignore[arg-type]

    def __getitem__(self, index: int) -> Any:
        generator = torch.Generator().manual_seed(self.seed + index)
        seeds = torch.randint(2**62, (len(self.random_objects),), generator=generator)
        for item, seed in zip(self.random_objects, seeds.tolist()):
            item.set_random_state(seed=seed)
        return self.dataset[index]
