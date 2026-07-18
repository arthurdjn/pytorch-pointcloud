from collections import Counter
from typing import Any, Dict

import pytest
import torch
from torch.utils.data import Dataset

from torch_pointcloud.datasets import ConcatDataset, SingleDatasetBatchSampler


class _DomainDataset(Dataset):
    """Each sample carries its domain id and its within-dataset index, so a collated batch reveals
    which source dataset every index came from."""

    def __init__(self, domain: int, size: int) -> None:
        self.domain = domain
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return {"domain": self.domain, "index": index}


def test_concat_len_is_sum() -> None:
    dataset = ConcatDataset([_DomainDataset(0, 7), _DomainDataset(1, 3)])
    assert len(dataset) == 10
    assert dataset.sizes == [7, 3]


def test_concat_maps_flat_index_to_child() -> None:
    dataset = ConcatDataset([_DomainDataset(0, 7), _DomainDataset(1, 3)])
    assert dataset[0] == {"domain": 0, "index": 0}
    assert dataset[6] == {"domain": 0, "index": 6}
    assert dataset[7] == {"domain": 1, "index": 0}
    assert dataset[9] == {"domain": 1, "index": 2}


def test_concat_negative_index() -> None:
    dataset = ConcatDataset([_DomainDataset(0, 7), _DomainDataset(1, 3)])
    assert dataset[-1] == {"domain": 1, "index": 2}
    assert dataset[-10] == {"domain": 0, "index": 0}


def test_concat_out_of_range_index_raises() -> None:
    dataset = ConcatDataset([_DomainDataset(0, 7), _DomainDataset(1, 3)])
    with pytest.raises(IndexError):
        _ = dataset[10]
    with pytest.raises(IndexError):
        _ = dataset[-11]


def test_sampler_batches_are_single_dataset() -> None:
    sizes = [40, 20]
    sampler = SingleDatasetBatchSampler(sizes, ratios=[2, 1], batch_size=4)
    offsets = [0, 40]
    batches = list(sampler)
    assert len(batches) > 0
    for batch in batches:
        assert len(batch) == 4
        which = [0 if i < offsets[1] else 1 for i in batch]
        assert len(set(which)) == 1


def test_sampler_ratio_proportions() -> None:
    """With drop_last batches, both datasets are fully covered per epoch and the batch counts follow the
    ratio (ScanNet-like 40 samples : S3DIS-like 20 samples, batch 4 -> 10 : 5 batches, ratio 2:1)."""
    sizes = [40, 20]
    sampler = SingleDatasetBatchSampler(sizes, ratios=[2, 1], batch_size=4, generator=torch.Generator().manual_seed(0))
    domains = Counter(0 if batch[0] < 40 else 1 for batch in sampler)
    assert domains[0] == 10
    assert domains[1] == 5
    assert len(sampler) == domains[0] + domains[1]


def test_sampler_main_dataset_drives_epoch() -> None:
    """The first (main) dataset's exhaustion ends the epoch; the second restarts to keep the 2:1 mix."""
    sizes = [40, 8]
    sampler = SingleDatasetBatchSampler(sizes, ratios=[2, 1], batch_size=4, generator=torch.Generator().manual_seed(0))
    domains = Counter(0 if batch[0] < 40 else 1 for batch in sampler)
    assert domains[0] == 10
    assert domains[1] == 5
    assert len(sampler) == 15


def test_sampler_len_matches_iteration() -> None:
    for sizes, ratios in ([40, 20], [2, 1]), ([37, 11], [3, 1]), ([16, 40], [1, 2]):
        sampler = SingleDatasetBatchSampler(
            sizes, ratios=ratios, batch_size=4, generator=torch.Generator().manual_seed(1)
        )
        assert len(sampler) == sum(1 for _ in sampler)
