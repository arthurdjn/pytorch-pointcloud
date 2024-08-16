import random
from pathlib import Path
from typing import Sequence

import numpy as np
import pytest
import torch
from pytest_mock import MockerFixture

from torch_pointcloud.datasets import ModelNet10
from torch_pointcloud.utils import PATH_LIKE, load_off, save_off


def mock_modelnet_dataset(
    data_dir: PATH_LIKE,
    out_dir: PATH_LIKE,
    num_samples_per_class: int = 2,
    num_points: int = 100,
) -> None:
    """Mocks the ModelNet dataset by sub-sampling the OFF files. This function is useful for testing purposes, and needs to be run only once.
    The mocked dataset is saved in the `tests/data` directory.

    Args:
        data_dir: The directory containing the ModelNet dataset.
        out_dir: The directory where the mocked dataset will be saved.
        num_samples_per_class: The number of samples to keep per class. Defaults to 2.
        num_points: The number of points to keep per sample. Defaults to 100.
    """

    def subsample_off_files(out_dir: PATH_LIKE, off_files: Sequence[PATH_LIKE], num_samples: int) -> None:
        num_samples = min(num_samples, len(off_files))
        selected_files = random.sample(off_files, num_samples)

        for off_file in selected_files:
            vertices, faces = load_off(off_file)

            if vertices.shape[0] > num_points:
                subsample_indices = np.random.choice(vertices.shape[0], num_points, replace=False)
                vertices = vertices[subsample_indices]

            out_path = Path(out_dir) / Path(off_file).relative_to(data_dir)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            save_off(out_path, vertices, faces)

    # Initialize paths
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)

    # Subsample the dataset
    classes = [path.name for path in data_dir.iterdir() if path.is_dir()]
    for class_name in classes:
        train_off_files = list(Path(data_dir, class_name, "train").rglob("*.off"))
        test_off_files = list(Path(data_dir, class_name, "test").rglob("*.off"))

        subsample_off_files(out_dir, train_off_files, num_samples_per_class)
        subsample_off_files(out_dir, test_off_files, num_samples_per_class)


@pytest.fixture
def modelnet10_train(data_dir: Path) -> ModelNet10:
    return ModelNet10(root=data_dir, train=True, download=False)


@pytest.fixture
def modelnet10_test(data_dir: Path) -> ModelNet10:
    return ModelNet10(root=data_dir, train=False, download=False)


def test_class_to_idx(modelnet10_train: ModelNet10) -> None:
    expected_classes = (
        "bathtub",
        "bed",
        "chair",
        "desk",
        "dresser",
        "monitor",
        "night_stand",
        "sofa",
        "table",
        "toilet",
    )
    expected_class_to_idx = {label: i for i, label in enumerate(expected_classes)}
    assert modelnet10_train.class_to_idx == expected_class_to_idx


def test_modelnet_lengths(modelnet10_train: ModelNet10, modelnet10_test: ModelNet10) -> None:
    assert len(modelnet10_train) == 10
    assert len(modelnet10_test) == 10


def test_getitem(modelnet10_train: ModelNet10) -> None:
    sample = modelnet10_train[0]

    assert "xyz" in sample and "face" in sample and "target" in sample

    # Check that the data types are correct
    assert isinstance(sample["xyz"], torch.Tensor), "'xyz' should be a torch.Tensor"
    assert isinstance(sample["face"], torch.Tensor), "'face' should be a torch.Tensor"
    assert isinstance(sample["target"], torch.Tensor), "'target' should be a torch.Tensor"

    # Check that xyz and faces are of the expected shapes
    assert sample["xyz"].ndim == 2 and sample["xyz"].size(1) == 3, "'xyz' should be a 2D tensor with shape [N, 3]"
    assert sample["face"].ndim == 2 and sample["face"].size(1) == 3, "'face' should be a 2D tensor with shape [M, 3]"


def test_data_processing(modelnet10_train: ModelNet10) -> None:
    assert len(modelnet10_train.data) > 0, "The training dataset should not be empty after processing."


def test_no_download(modelnet10_train: ModelNet10, mocker: MockerFixture) -> None:
    mocked_download = mocker.patch.object(ModelNet10, "download", autospec=True)
    _ = ModelNet10(root=modelnet10_train.root, train=True, download=False)
    mocked_download.assert_not_called()


def test_process_call(modelnet10_train: ModelNet10, mocker: MockerFixture) -> None:
    mocked_process = mocker.patch.object(ModelNet10, "process", autospec=True)
    _ = ModelNet10(root=modelnet10_train.root, train=True, download=False)
    mocked_process.assert_called_once()
