"""Dataset wrapper that lengthens an epoch by repeating an underlying dataset."""

import bisect
import itertools
from typing import Any, List, Optional, Sequence, Union

from torch.utils.data import Dataset


class RepeatDataset(Dataset):
    """Repeats a dataset `k` times so one epoch iterates it `k` times, or each sample its own number of times.

    Lengthens an epoch (and thus the number of optimizer steps per epoch)
    without otherwise changing training. With one count per sample, an epoch draws sample $i$ `k[i]` times, in
    order (e.g. to draw large scenes more often than small ones).

    Args:
        dataset: The dataset to repeat.
        k: Number of times the dataset is iterated per epoch, or the number of times each sample is drawn per
            epoch (one non-negative count per sample).
    """

    def __init__(self, dataset: Dataset, k: Union[int, Sequence[int]]) -> None:
        self.dataset = dataset
        self.k = k
        self._ends: Optional[List[int]] = None
        if not isinstance(k, int):
            counts = [int(count) for count in k]
            if len(counts) != len(dataset):  # type: ignore[arg-type]
                raise ValueError(f"`k` lists {len(counts)} counts for a dataset of {len(dataset)} samples.")  # type: ignore[arg-type]

            if any(count < 0 for count in counts):
                raise ValueError("`k` counts must be non-negative.")

            self._ends = list(itertools.accumulate(counts))

    def __len__(self) -> int:
        if self._ends is not None:
            return self._ends[-1] if self._ends else 0
        return len(self.dataset) * self.k  # type: ignore[arg-type, operator]

    def __getitem__(self, index: int) -> Any:
        if self._ends is not None:
            if not 0 <= index < len(self):
                raise IndexError(f"Index {index} out of range for {len(self)} samples.")
            return self.dataset[bisect.bisect_right(self._ends, index)]
        return self.dataset[index % len(self.dataset)]  # type: ignore[arg-type]
