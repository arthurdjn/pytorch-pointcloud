"""Dataset wrapper that gives every sample the same random draws each time it is read."""

from typing import Any

import numpy as np
from torch.utils.data import Dataset

from torch_pointcloud.utils.data import set_random_states


class SeededDataset(Dataset):
    """Re-seeds the seeded random objects of a dataset from the index of the sample being read.

    A seeded random transform draws from one stream that advances with every sample, and `PointCloudDataLoader`
    re-seeds it in every worker and epoch, so a sample gets new draws each time it is read. Wrapped in this class, a
    sample gets the draws of its own index, whatever the epoch, the number of workers and the order of the reads: a
    validation split is scored on the same crops and points at every epoch.

    The random objects are the ones `torch_pointcloud.utils.data.set_random_states` re-seeds (seeded transforms, a
    seeded `MixDataset`), re-seeded in place before every read: do not share them with another dataset. Those drawing
    from the global generator (`seed=None`) keep doing so. An index read twice gets the same draws twice: to draw a
    sample several times per epoch, list its index once per draw in a `torch.utils.data.Subset` and wrap the subset.

    Args:
        dataset: The dataset to read from, with seeded random transforms.
        seed: Seed of the draws, combined with the index of the sample being read.

    Raises:
        ValueError: If `dataset` holds no seeded random object.

    Example:
        ```python
        import torch
        from torch.utils.data import Dataset

        import torch_pointcloud.transforms as T
        from torch_pointcloud.datasets import SeededDataset

        class Clouds(Dataset):
            def __init__(self) -> None:
                self.transform = T.RandomSample(keys="pos", num_samples=16, seed=0)

            def __len__(self) -> int:
                return 4

            def __getitem__(self, index: int) -> dict:
                return self.transform({"pos": torch.arange(64 * 3.0).reshape(64, 3)})

        dataset = SeededDataset(Clouds())
        assert torch.equal(dataset[0]["pos"], dataset[0]["pos"])
        ```
    """

    def __init__(self, dataset: Dataset, seed: int = 0) -> None:
        if set_random_states(dataset, seed=seed) == seed:
            raise ValueError("`dataset` holds no seeded random object: pass a `seed` to its random transforms.")

        self.dataset = dataset
        self.seed = seed

    def __len__(self) -> int:
        return len(self.dataset)  # type: ignore[arg-type]

    def __getitem__(self, index: int) -> Any:
        index = range(len(self))[index]
        sample_seed = np.random.SeedSequence([self.seed, index]).generate_state(1, dtype=np.uint64)[0]
        set_random_states(self.dataset, seed=int(sample_seed))
        return self.dataset[index]
