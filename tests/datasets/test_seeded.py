from typing import Any, Dict, List

import pytest
import torch
from torch import Tensor
from torch.utils.data import Dataset, Subset

import torch_pointcloud.transforms as T
from torch_pointcloud.datasets import MixDataset, SeededDataset
from torch_pointcloud.utils.data import DataKeys, PointCloudDataLoader


class DummySegmentationDataset(Dataset):
    def __init__(self, transform: T.Transform) -> None:
        self.transform = transform

    def __len__(self) -> int:
        return 6

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if not 0 <= index < len(self):
            raise IndexError(index)

        pos = torch.arange(64 * 3, dtype=torch.float32).reshape(64, 3) + 1000 * index
        return self.transform({DataKeys.POS: pos, DataKeys.SEGMENT: torch.arange(64)})


def _seeded_transform() -> T.Compose:
    return T.Compose(
        [
            T.RandomSample(keys=[DataKeys.POS, DataKeys.SEGMENT], num_samples=8, seed=0),
            T.RandomJitter(keys=DataKeys.POS, sigma=1.0, clip=None, seed=0),
        ]
    )


def _one_pass(dataset: Dataset, num_workers: int) -> List[Tensor]:
    loader = PointCloudDataLoader(dataset, batch_size=2, num_workers=num_workers)
    return [torch.cat([batch[DataKeys.POS], batch[DataKeys.SEGMENT][:, None]], dim=1) for batch in loader]


@pytest.mark.parametrize(
    "num_workers",
    [
        pytest.param(0, id="main-process"),
        pytest.param(2, id="two-workers"),
    ],
)
def test_seeded_dataset_yields_the_same_batches_at_every_pass(num_workers: int) -> None:
    dataset = SeededDataset(DummySegmentationDataset(_seeded_transform()))
    first = _one_pass(dataset, num_workers)
    second = _one_pass(dataset, num_workers)
    assert len(first) == 3
    assert all(torch.equal(a, b) for a, b in zip(first, second))


def test_seeded_dataset_draws_do_not_depend_on_the_number_of_workers() -> None:
    dataset = SeededDataset(DummySegmentationDataset(_seeded_transform()))
    assert all(torch.equal(a, b) for a, b in zip(_one_pass(dataset, 0), _one_pass(dataset, 2)))


def test_seeded_dataset_draws_do_not_depend_on_the_read_order() -> None:
    dataset = SeededDataset(DummySegmentationDataset(_seeded_transform()))
    expected = [dataset[index][DataKeys.POS] for index in range(len(dataset))]
    for index in (4, 1, 4, 5, 0):
        assert torch.equal(dataset[index][DataKeys.POS], expected[index])


def test_seeded_dataset_gives_each_index_and_each_seed_its_own_draws() -> None:
    dataset = SeededDataset(DummySegmentationDataset(_seeded_transform()), seed=0)
    other = SeededDataset(DummySegmentationDataset(_seeded_transform()), seed=100)
    segments = [dataset[index][DataKeys.SEGMENT].tolist() for index in range(len(dataset))]
    assert len({tuple(segment) for segment in segments}) == len(dataset)
    assert segments[0] != other[0][DataKeys.SEGMENT].tolist()


def test_seeded_dataset_gives_each_draw_of_a_repeated_sample_its_own_draws() -> None:
    draws = Subset(DummySegmentationDataset(_seeded_transform()), [0, 0, 0, 3])
    dataset = SeededDataset(draws)
    segments = [dataset[index][DataKeys.SEGMENT].tolist() for index in range(len(dataset))]
    assert len({tuple(segment) for segment in segments}) == 4
    assert segments == [dataset[index][DataKeys.SEGMENT].tolist() for index in range(len(dataset))]


def test_seeded_dataset_reseeds_a_seeded_mix_dataset() -> None:
    mix = T.Mix3D(keys=(DataKeys.POS, DataKeys.SEGMENT), instance_key=None)
    dataset = SeededDataset(MixDataset(DummySegmentationDataset(_seeded_transform()), mix=mix, seed=0))
    first = [dataset[index][DataKeys.POS] for index in range(len(dataset))]
    second = [dataset[index][DataKeys.POS] for index in range(len(dataset))]
    assert all(torch.equal(a, b) for a, b in zip(first, second))


def test_seeded_dataset_leaves_unseeded_transforms_on_the_global_generator() -> None:
    crop = T.RandomSample(keys=[DataKeys.POS, DataKeys.SEGMENT], num_samples=8, seed=0)
    jitter = T.RandomJitter(keys=DataKeys.POS, sigma=1.0, clip=None)
    dataset = SeededDataset(DummySegmentationDataset(T.Compose([crop, jitter])))
    first, second = dataset[2], dataset[2]
    assert jitter.R is None
    assert torch.equal(first[DataKeys.SEGMENT], second[DataKeys.SEGMENT])
    assert not torch.equal(first[DataKeys.POS], second[DataKeys.POS])


def test_seeded_dataset_without_a_seeded_random_object_raises() -> None:
    transform = T.RandomSample(keys=[DataKeys.POS, DataKeys.SEGMENT], num_samples=8)
    with pytest.raises(ValueError, match="no seeded random object"):
        SeededDataset(DummySegmentationDataset(transform))


def test_seeded_dataset_len_and_index_errors_follow_the_dataset() -> None:
    dataset = SeededDataset(DummySegmentationDataset(_seeded_transform()))
    assert len(dataset) == 6
    with pytest.raises(IndexError):
        dataset[6]


@pytest.mark.parametrize(
    "num_workers",
    [
        pytest.param(0, id="main-process"),
        pytest.param(2, id="two-workers"),
    ],
)
def test_seeded_transforms_alone_draw_new_samples_at_every_pass(num_workers: int) -> None:
    dataset = DummySegmentationDataset(_seeded_transform())
    first = _one_pass(dataset, num_workers)
    second = _one_pass(dataset, num_workers)
    assert not all(torch.equal(a, b) for a, b in zip(first, second))
