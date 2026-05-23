from typing import Iterable, Optional, Union

import lightning.pytorch as L
from torch.utils.data import DataLoader, Dataset, Sampler

from torch_pointcloud.utils.data import collate


class PointCloudDataModule(L.LightningDataModule):
    """LightningDataModule wrapping point cloud datasets with the packed-batch collate.

    Each dataset is passed through as-is. To lengthen an epoch (Pointcept's `loop`),
    wrap the training dataset with `torch_pointcloud.datasets.RepeatDataset(dataset, loop=k)`
    before passing it in.

    The DataLoader kwargs that make sense for point-cloud training are exposed as
    constructor arguments and forwarded to each loader. `shuffle` is forced to
    `True` for train and `False` for val/test, and `collate_fn` is locked to the
    packed-batch `torch_pointcloud.utils.data.collate`.

    Args:
        train_dataset: Dataset for the training loop.
        val_dataset: Dataset for the validation loop.
        test_dataset: Dataset for the test loop.
        batch_size: Number of point clouds per batch.
        num_workers: Number of worker processes for data loading.
        pin_memory: Pin tensors in pinned (page-locked) memory before transfer.
        drop_last: Drop the last incomplete batch. Applied to the *train* loader only;
            val/test always set `drop_last=False`.
        timeout: Timeout for collecting a batch from the workers.
        prefetch_factor: Number of batches each worker prefetches ahead.
        persistent_workers: Keep workers alive between epochs.
        pin_memory_device: Target device for `pin_memory` (e.g. `"cuda"`).
    """

    def __init__(
        self,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        test_dataset: Optional[Dataset] = None,
        *,
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last: bool = True,
        timeout: float = 0.0,
        prefetch_factor: Optional[int] = None,
        persistent_workers: bool = False,
        pin_memory_device: str = "",
    ) -> None:
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.timeout = timeout
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers
        self.pin_memory_device = pin_memory_device

    def configure_dataloader(
        self,
        dataset: Optional[Dataset],
        *,
        shuffle: bool,
        drop_last: bool,
        sampler: Optional[Union[Sampler, Iterable]] = None,
        batch_sampler: Optional[Union[Sampler, Iterable]] = None,
    ) -> DataLoader:
        if dataset is None:
            raise ValueError("Requested a dataloader for a dataset that was not provided.")

        # `sampler` and `shuffle` are mutually exclusive in torch.utils.data.DataLoader.
        effective_shuffle = False if sampler is not None or batch_sampler is not None else shuffle

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=effective_shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=self.num_workers,
            collate_fn=collate,
            pin_memory=self.pin_memory,
            drop_last=drop_last,
            timeout=self.timeout,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.persistent_workers,
            pin_memory_device=self.pin_memory_device,
        )

    def train_dataloader(self) -> DataLoader:
        return self.configure_dataloader(self.train_dataset, shuffle=True, drop_last=self.drop_last)

    def val_dataloader(self) -> DataLoader:
        return self.configure_dataloader(self.val_dataset, shuffle=False, drop_last=False)

    def test_dataloader(self) -> DataLoader:
        return self.configure_dataloader(self.test_dataset, shuffle=False, drop_last=False)
