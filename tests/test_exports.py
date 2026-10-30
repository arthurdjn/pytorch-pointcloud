import importlib
import inspect
import subprocess
import sys

import pytest

from torch_pointcloud.utils.imports import _LIGHTNING_AVAILABLE

PACKAGES = [
    pytest.param("torch_pointcloud.datasets"),
    pytest.param("torch_pointcloud.inferers"),
    pytest.param("torch_pointcloud.layers"),
    pytest.param(
        "torch_pointcloud.lightning",
        marks=pytest.mark.skipif(not _LIGHTNING_AVAILABLE, reason="lightning is not installed"),
    ),
    pytest.param("torch_pointcloud.losses"),
    pytest.param("torch_pointcloud.losses.functional"),
    pytest.param("torch_pointcloud.metrics"),
    pytest.param("torch_pointcloud.models"),
    pytest.param("torch_pointcloud.ops"),
    pytest.param("torch_pointcloud.optim"),
    pytest.param("torch_pointcloud.transforms"),
    pytest.param("torch_pointcloud.transforms.functional"),
    pytest.param("torch_pointcloud.utils"),
]

OPS_MODULES = [
    "anchors",
    "box3d",
    "cluster",
    "density",
    "geometry",
    "heatmap",
    "hilbert",
    "masking",
    "math",
    "octree",
    "serialization",
    "voxelization",
]

TRANSFORMS_MODULES = [
    "augmentation",
    "base",
    "box",
    "geometry",
    "masking",
    "mixing",
    "octree",
    "sampling",
    "scaling",
    "utils",
    "voxelization",
]


@pytest.mark.parametrize("name", PACKAGES)
def test_all_names_resolve(name: str) -> None:
    module = importlib.import_module(name)
    assert isinstance(module.__all__, list)
    for attr in module.__all__:
        getattr(module, attr)


@pytest.mark.parametrize("name", PACKAGES)
def test_all_lists_every_public_name(name: str) -> None:
    module = importlib.import_module(name)
    public = {attr for attr, value in vars(module).items() if not attr.startswith("_") and not inspect.ismodule(value)}
    assert public <= set(module.__all__)


def test_transforms_exports_every_module_class() -> None:
    transforms = importlib.import_module("torch_pointcloud.transforms")
    expected = [
        name
        for module in TRANSFORMS_MODULES
        for name in importlib.import_module(f"torch_pointcloud.transforms.{module}").__all__
    ]
    assert sorted(transforms.__all__) == sorted(expected)


def test_ops_namespace_in_fresh_interpreter() -> None:
    code = "import torch_pointcloud as tp; print(*tp.ops.__all__); print(tp.ops.box3d.nms3d.__name__)"
    stdout = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True).stdout
    assert stdout.split("\n")[:2] == [" ".join(OPS_MODULES), "nms3d"]
