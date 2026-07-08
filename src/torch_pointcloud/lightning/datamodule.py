from typing import TYPE_CHECKING, Iterable, Optional, Sequence, Union

from torch.utils.data import DataLoader, Dataset, Sampler

from torch_pointcloud.datasets.concat import SingleDatasetBatchSampler
from torch_pointcloud.utils.data import PointCloudDataLoader
from torch_pointcloud.utils.imports import _LIGHTNING_GITHUB_URL, optional_import

if TYPE_CHECKING:
    from lightning.pytorch import LightningDataModule
else:
    LightningDataModule, _ = optional_import("lightning.pytorch", "LightningDataModule", url=_LIGHTNING_GITHUB_URL)


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
        train_ratios: One positive integer sampling weight per child dataset of a `ConcatDataset` train
            set. When set, the train loader draws single-dataset batches interleaved by these ratios via
            `torch_pointcloud.datasets.SingleDatasetBatchSampler`, so every batch stays single-domain
            (required by per-dataset normalization such as PDNorm). Leave `None` for a single dataset.
        val_dataset: Dataset for the validation loop.
        test_dataset: Dataset for the test loop.
        stack_keys: Keys collated by stacking to a leading batch dim instead of concatenating.
        cat_keys: Packed keys that additionally emit a `batch_<key>` per-element scene index.
        batch_size: Number of point clouds per batch.
        eval_batch_size: Batch size of the val/test loaders; defaults to `batch_size`. Evaluation often
            runs full-resolution scenes while training runs crops, so the two memory envelopes differ
            (a benchmark protocol is typically one scene per batch).
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
        train_ratios: Optional[Sequence[int]] = None,
        stack_keys: Optional[Sequence[str]] = None,
        cat_keys: Optional[Sequence[str]] = None,
        batch_size: int = 1,
        eval_batch_size: Optional[int] = None,
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

        self.train_ratios = train_ratios
        self.stack_keys = stack_keys
        self.cat_keys = cat_keys
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size if eval_batch_size is not None else batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.timeout = timeout
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers
        self.pin_memory_device = pin_memory_device

    def setup(self, stage: str) -> None:
        """Apply the model's registered evaluation transform to any dataset that has none.

        The LightningModule (built from the registry) carries its `transform`; an experiment leaves
        a dataset's `transform` as `None` to use it, or sets one explicitly for custom augmentation.
        """
        trainer = getattr(self, "trainer", None)
        lit_model = getattr(trainer, "lightning_module", None)
        transform = getattr(lit_model, "transform", None)
        if transform is None:
            return

        for dataset in (self.train_dataset, self.val_dataset, self.test_dataset):
            if dataset is not None and getattr(dataset, "transform", None) is None:
                dataset.transform = transform  # type: ignore[attr-defined]

    def configure_dataloader(
        self,
        dataset: Optional[Dataset],
        *,
        shuffle: bool,
        drop_last: bool,
        batch_size: Optional[int] = None,
        sampler: Optional[Union[Sampler, Iterable]] = None,
        batch_sampler: Optional[Union[Sampler, Iterable]] = None,
    ) -> DataLoader:
        if dataset is None:
            raise ValueError("Requested a dataloader for a dataset that was not provided.")

        # `sampler` and `shuffle` are mutually exclusive in torch.utils.data.DataLoader.
        effective_shuffle = False if sampler is not None or batch_sampler is not None else shuffle

        # A `batch_sampler` already yields whole batches, so DataLoader requires batch_size 1,
        # no separate sampler, and drop_last off.
        if batch_sampler is not None:
            batch_size = 1
            drop_last = False
            sampler = None

        return PointCloudDataLoader(
            dataset,
            stack_keys=self.stack_keys,
            cat_keys=self.cat_keys,
            batch_size=batch_size if batch_size is not None else self.batch_size,
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
        batch_sampler: Optional[SingleDatasetBatchSampler] = None
        if self.train_ratios is not None:
            sizes = getattr(self.train_dataset, "sizes", None)
            if sizes is None:
                raise ValueError("train_ratios requires train_dataset to be a ConcatDataset exposing `sizes`.")
            batch_sampler = SingleDatasetBatchSampler(
                sizes, ratios=self.train_ratios, batch_size=self.batch_size, shuffle=True, drop_last=self.drop_last
            )
        return self.configure_dataloader(
            self.train_dataset, shuffle=True, drop_last=self.drop_last, batch_sampler=batch_sampler
        )

    def val_dataloader(self) -> DataLoader:
        return self.configure_dataloader(
            self.val_dataset, shuffle=False, drop_last=False, batch_size=self.eval_batch_size
        )

    def test_dataloader(self) -> DataLoader:
        # Fall back to the validation set when no dedicated test set is given: for these benchmarks the
        # held-out split is the validation set, so `Trainer.test` (e.g. a pretrained-weight benchmark)
        # evaluates on it without the experiment having to duplicate the dataset as `test_dataset`.
        dataset = self.test_dataset if self.test_dataset is not None else self.val_dataset
        return self.configure_dataloader(dataset, shuffle=False, drop_last=False, batch_size=self.eval_batch_size)
