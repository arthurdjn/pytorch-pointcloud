"""Dataset wrapper that lengthens an epoch by repeating an underlying dataset."""

from typing import Any, Iterator, Optional, Sequence

import torch
from torch.utils.data import Dataset, Sampler

from torch_pointcloud.utils.conversion import ensure_list


class RepeatDataset(Dataset):
    """Repeats a dataset `k` times so one epoch iterates it `k` times.

    Lengthens an epoch (and thus the number of optimizer steps per epoch)
    without otherwise changing training.

    Args:
        dataset: The dataset to repeat.
        k: Number of times the dataset is iterated per epoch.
    """

    def __init__(self, dataset: Dataset, k: int) -> None:
        self.dataset = dataset
        self.k = k

    def __len__(self) -> int:
        return len(self.dataset) * self.k  # type: ignore[arg-type]

    def __getitem__(self, index: int) -> Any:
        return self.dataset[index % len(self.dataset)]  # type: ignore[arg-type]


class RepeatSampler(Sampler[int]):
    r"""Draws every sample of a dataset its own number of times per epoch.

    An epoch yields index $i$ `counts[i]` times, shuffled unless `shuffle` is off, so unevenly sized samples
    can be visited in proportion to their size (large scenes more often than small ones).

    Args:
        counts: Number of draws per epoch of every sample, one non-negative count per dataset index.
        shuffle: Shuffle the draws; `False` yields them grouped by sample, in index order.
        generator: Random generator of the shuffle; `None` uses the global one.

    Example:
        ```pycon
        >>> sampler = RepeatSampler([2, 0, 3, 1], shuffle=False)
        >>> list(sampler)
        [0, 0, 2, 2, 2, 3]
        >>> len(sampler)
        6

        ```
    """

    def __init__(
        self,
        counts: Sequence[int],
        shuffle: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        self.counts = torch.tensor(ensure_list(counts), dtype=torch.long)
        if (self.counts < 0).any():
            raise ValueError("`counts` must be non-negative.")

        self.shuffle = shuffle
        self.generator = generator

    def __iter__(self) -> Iterator[int]:
        indices = torch.repeat_interleave(torch.arange(self.counts.numel()), self.counts)
        if self.shuffle:
            indices = indices[torch.randperm(indices.numel(), generator=self.generator)]
        return iter(indices.tolist())

    def __len__(self) -> int:
        return int(self.counts.sum())
