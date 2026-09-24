"""Dataset wrapper that lengthens an epoch by repeating an underlying dataset."""

from typing import Any

from torch.utils.data import Dataset


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
