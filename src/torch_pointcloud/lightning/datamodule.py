from typing import TYPE_CHECKING, Iterable, Optional, Sequence, Union

from torch.utils.data import DataLoader, Dataset, Sampler

from torch_pointcloud.utils.data import PointCloudDataLoader
from torch_pointcloud.utils.imports import optional_import

if TYPE_CHECKING:
    from lightning.pytorch import LightningDataModule
else:
    LightningDataModule, _ = optional_import("lightning.pytorch", "LightningDataModule")


class PointCloudDataModule(LightningDataModule):
    """LightningDataModule wrapping point cloud datasets with the packed-batch collate.

    Each dataset is passed through as-is. To lengthen an epoch (Pointcept's `loop`),
    wrap the training dataset with `torch_pointcloud.datasets.RepeatDataset(dataset, loop=k)`
    before passing it in.

    Loaders are built with `torch_pointcloud.utils.data.PointCloudDataLoader`, which collates to the
    packed-batch `torch_pointcloud.utils.data.collate`. Collation specs are never read off the dataset
    (transforms rewrite keys downstream); pass `stack_keys` / `cat_keys` to control how per-scene ground
    truth is batched (e.g. `cat_keys=("box",)` for a detection dataset). `shuffle` is forced to `True`
    for train and `False` for val/test.

    Args:
        train_dataset: Dataset for the training loop.
        val_dataset: Dataset for the validation loop.
        test_dataset: Dataset for the test loop.
        stack_keys: Keys collated by stacking to a leading batch dim instead of concatenating.
        cat_keys: Packed keys that additionally emit a `batch_<key>` per-element scene index.
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
        stack_keys: Optional[Sequence[str]] = None,
        cat_keys: Optional[Sequence[str]] = None,
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

        self.stack_keys = stack_keys
        self.cat_keys = cat_keys
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.timeout = timeout
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers
        self.pin_memory_device = pin_memory_device

    def setup(self, stage: str) -> None:
        """Apply the model's registered evaluation transform to any dataset that has none.

        The LightningModule (built from the registry) carries its `eval_transform`; an experiment leaves
        a dataset's `transform` as `None` to use it, or sets one explicitly for custom augmentation.
        """
        trainer = getattr(self, "trainer", None)
        lit_model = getattr(trainer, "lightning_module", None)
        eval_transform = getattr(lit_model, "eval_transform", None)
        if eval_transform is None:
            return
        for dataset in (self.train_dataset, self.val_dataset, self.test_dataset):
            if dataset is not None and getattr(dataset, "transform", None) is None:
                dataset.transform = eval_transform  # type: ignore[attr-defined]

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

        return PointCloudDataLoader(
            dataset,
            stack_keys=self.stack_keys,
            cat_keys=self.cat_keys,
            batch_size=self.batch_size,
            shuffle=effective_shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=self.num_workers,
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
        # Fall back to the validation set when no dedicated test set is given: for these benchmarks the
        # held-out split is the validation set, so `Trainer.test` (e.g. a pretrained-weight benchmark)
        # evaluates on it without the experiment having to duplicate the dataset as `test_dataset`.
        dataset = self.test_dataset if self.test_dataset is not None else self.val_dataset
        return self.configure_dataloader(dataset, shuffle=False, drop_last=False)
