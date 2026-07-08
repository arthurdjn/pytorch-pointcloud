"""Concatenate several datasets into one flat index space for multi-dataset joint training."""

from bisect import bisect_right
from typing import Any, Iterator, List, Optional, Sequence

import torch
from torch.utils.data import Dataset, Sampler


class ConcatDataset(Dataset):
    """Concatenates several datasets into one flat index space.

    Each child dataset keeps its own `transform`, so datasets from different domains (e.g. ScanNet and
    S3DIS) train jointly while stamping their own condition key and mapping to their native label space.
    Pair it with `SingleDatasetBatchSampler` to keep every batch single-domain so per-dataset
    normalization statistics (BatchNorm, PDNorm) stay clean.

    Args:
        datasets: The datasets to concatenate, in order. The first is treated as the main dataset by
            `SingleDatasetBatchSampler` (its exhaustion ends the epoch).

    Example:
        ```python
        from torch_pointcloud.datasets import ConcatDataset, S3DIS, ScanNet20

        dataset = ConcatDataset([ScanNet20(root, split="train"), S3DIS(root, areas=["Area_1"])])
        len(dataset)  # len(scannet) + len(s3dis)
        dataset.sizes  # [len(scannet), len(s3dis)]
        ```
    """

    def __init__(self, datasets: Sequence[Dataset]) -> None:
        if len(datasets) == 0:
            raise ValueError("ConcatDataset requires at least one dataset.")
        self.datasets = list(datasets)
        self.sizes = [len(d) for d in self.datasets]  # type: ignore[arg-type]
        self.cumulative_sizes: List[int] = []
        total = 0
        for size in self.sizes:
            total += size
            self.cumulative_sizes.append(total)

    def __len__(self) -> int:
        return self.cumulative_sizes[-1]

    def __getitem__(self, index: int) -> Any:
        if index < 0:
            index += len(self)
        dataset_index = bisect_right(self.cumulative_sizes, index)
        start = self.cumulative_sizes[dataset_index - 1] if dataset_index > 0 else 0
        return self.datasets[dataset_index][index - start]


class SingleDatasetBatchSampler(Sampler[List[int]]):
    """Yields batches whose global indices all come from one dataset of a `ConcatDataset`.

    Given the per-dataset sizes and one positive integer ratio per dataset, it partitions each dataset's
    indices into batches of `batch_size` and interleaves the datasets round-robin weighted by the ratios.
    Within a round the first dataset yields `ratios[0]` batches, the second `ratios[1]`, and so on. The
    first (main) dataset drives the epoch length: when it
    is exhausted the epoch ends, while the other datasets restart (reshuffled) as needed. Because every
    yielded batch is drawn from a single dataset, per-batch normalization (BatchNorm, PDNorm) sees a
    single domain.

    Args:
        sizes: Number of samples in each child dataset, in `ConcatDataset` order.
        ratios: One positive integer sampling weight per dataset, aligned with `sizes`.
        batch_size: Number of indices per batch.
        shuffle: Shuffle each dataset's indices, reshuffling on restart.
        drop_last: Drop each dataset's trailing partial batch.
        generator: Optional `torch.Generator` for shuffling.

    Shape:
        Each yielded value is a `List[int]` of length `batch_size` (or fewer for a trailing batch when
        `drop_last` is `False`).

    Example:
        ```python
        from torch.utils.data import DataLoader

        from torch_pointcloud.datasets import ConcatDataset, SingleDatasetBatchSampler

        dataset = ConcatDataset([scannet, s3dis])
        sampler = SingleDatasetBatchSampler(dataset.sizes, ratios=[2, 1], batch_size=4)
        loader = DataLoader(dataset, batch_sampler=sampler)
        ```
    """

    def __init__(
        self,
        sizes: Sequence[int],
        ratios: Sequence[int],
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        if len(sizes) != len(ratios):
            raise ValueError(f"sizes and ratios must have the same length, got {len(sizes)} and {len(ratios)}.")
        if any(r <= 0 for r in ratios):
            raise ValueError(f"ratios must be positive integers, got {list(ratios)}.")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}.")

        self.sizes = list(sizes)
        self.ratios = list(ratios)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.generator = generator
        self.offsets: List[int] = [sum(self.sizes[:i]) for i in range(len(self.sizes))]

    def _num_batches(self, size: int) -> int:
        if self.drop_last:
            return size // self.batch_size
        return (size + self.batch_size - 1) // self.batch_size

    def _batches(self, dataset_index: int) -> List[List[int]]:
        size = self.sizes[dataset_index]
        offset = self.offsets[dataset_index]
        if self.shuffle:
            order = torch.randperm(size, generator=self.generator).tolist()
        else:
            order = list(range(size))

        indices = [offset + i for i in order]
        batches = [indices[start : start + self.batch_size] for start in range(0, size, self.batch_size)]

        if self.drop_last and batches and len(batches[-1]) < self.batch_size:
            batches.pop()
        return batches

    def __len__(self) -> int:
        main = self._num_batches(self.sizes[0])
        if main == 0:
            return 0

        full_rounds, remainder = divmod(main, self.ratios[0])
        per_round = self.ratios[0] + sum(
            self.ratios[i] for i in range(1, len(self.sizes)) if self._num_batches(self.sizes[i]) > 0
        )
        return full_rounds * per_round + remainder

    def __iter__(self) -> Iterator[List[int]]:
        iterators = [iter(self._batches(i)) for i in range(len(self.sizes))]
        while True:
            for dataset_index, ratio in enumerate(self.ratios):
                for _ in range(ratio):
                    batch = next(iterators[dataset_index], None)
                    if batch is None:
                        if dataset_index == 0:
                            return

                        iterators[dataset_index] = iter(self._batches(dataset_index))
                        batch = next(iterators[dataset_index], None)
                        if batch is None:
                            break

                    yield batch
