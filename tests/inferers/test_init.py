import importlib
import re
from pathlib import Path

import pytest
from packaging.version import Version

import torch_pointcloud.inferers as inferers

PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"


def _source_version() -> Version:
    """The `pyproject.toml` version, which a release bump edits before the package is reinstalled."""
    match = re.search(r'^version = "(?P<version>[^"]+)"$', PYPROJECT.read_text(), re.MULTILINE)
    assert match is not None, PYPROJECT
    return Version(match["version"])


@pytest.mark.parametrize(
    "module,former_name,name",
    [
        pytest.param("sliding_window", "SlidingWindowInferer", "SlidingWindowSegmentationInferer"),
        pytest.param("sliding_window", "sliding_window_inference", "sliding_window_segmentation_inference"),
        pytest.param("knn_window", "KNNWindowInferer", "KNNWindowSegmentationInferer"),
        pytest.param("knn_window", "knn_window_inference", "knn_window_segmentation_inference"),
        pytest.param("potential_sphere", "PotentialSphereInferer", "PotentialSphereSegmentationInferer"),
        pytest.param("potential_sphere", "potential_sphere_inference", "potential_sphere_segmentation_inference"),
        pytest.param("voxel_partition", "VoxelPartitionInferer", "VoxelPartitionSegmentationInferer"),
        pytest.param("tta", "TTAInferer", "TTASegmentationInferer"),
    ],
)
def test_former_inferer_names_resolve_until_their_removal(module: str, former_name: str, name: str) -> None:
    """Fails on the version bump that drops the former names: delete each module's `_RENAMED` and this test."""
    assert _source_version() < Version("0.0.20"), f"{former_name!r} was removed in 0.0.20; drop its alias"
    submodule = importlib.import_module(f"torch_pointcloud.inferers.{module}")
    for namespace in (inferers, submodule):
        with pytest.warns(FutureWarning, match=f"`{former_name}` is deprecated since 0.0.19.*use `{name}`"):
            assert getattr(namespace, former_name) is getattr(submodule, name)
    assert former_name not in inferers.__all__


def test_unknown_inferer_name_raises_attribute_error() -> None:
    with pytest.raises(AttributeError, match="has no attribute 'NoSuchInferer'"):
        inferers.NoSuchInferer  # noqa: B018
